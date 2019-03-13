#!/usr/bin/env python

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Dot, Flatten, Concatenate, Dropout, Lambda, Subtract

from tensorflow.keras.initializers import Initializer, RandomNormal, Zeros, Ones
from tensorflow.keras.constraints import Constraint, UnitNorm, MinMaxNorm

from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework import dtypes

from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np

import semeval_eval
import yamane_evaluator

from collections import Sequence


def get_embeddings_model(embedding_matrix):
    hypo_input = Input(shape=(1,))
    hyper_input = Input(shape=(1,))

    word_embedding = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], name='TermEmbedding',
                               embeddings_constraint = UnitNorm(axis=1))

    hypo_embedding = word_embedding(hypo_input)
    hyper_embedding = word_embedding(hyper_input)

    embedding_model = Model(inputs=[hypo_input, hyper_input], outputs=[hypo_embedding, hyper_embedding])

    # inject pre-trained embeddings into this mini, resusable model/layer
    embedding_model.get_layer(name='TermEmbedding').set_weights([embedding_matrix])
    embedding_model.get_layer(name='TermEmbedding').trainable = False
    
    return embedding_model

class ForceToOne (Constraint):    
    def __call__(self, w):
        w /= w
        return w


class YamaneCluster():
    def __init__(self, args):
        self.embeddings_layer = args['embeddings_layer']
        self.lr = args['lr']
        
        # initialise variables             
        self.epoch_count = 0
        # perisists losses across epochs
        self.loss = []
        self.test_loss = []               
        
        # used to calculate loss during "current" epoch
        self._loss = 0.
        self._test_loss = 0.        
        self._train_update_count = 0
        self._test_update_count = 0
        
        self.model = self.build_yamane_cluster()
    
    # makes sense to re-purpose the call to trigger training the cluster
    def __call__(self):
        return self.model
            
    def before_epoch(self):
        self._loss = 0.
        self._test_loss = 0.
        self._train_update_count = 0
        self._test_update_count = 0
                
    def after_epoch(self):
        #print ("train update count: %d; train loss: %0.5f" % (self._train_update_count, self._loss))
        #print ("test update count: %d; test loss: %0.5f" % (self._test_update_count, self._test_loss))
        
        if self._train_update_count > 0:
            self.loss.append(round(self._loss/self._train_update_count, 5))
        else:
            self.loss.append(0.)
        
        if self._test_update_count > 0:
            self.test_loss.append(round(self._test_loss/self._test_update_count, 5))
        else:
            self.test_loss.append(0.)
    
    
    def build_yamane_cluster(self, phi_k=1):
        hypo_input  = Input(shape=(1,), name='Hyponym')    
        hyper_input = Input(shape=(1,), name='Hypernym')

        #def random_identity(shape, dtype="float32", partition_info=None):    
        #    identity = K.eye(shape[-1], dtype='float32')
        #    return identity 

        hypo_embedding, hyper_embedding = self.embeddings_layer([hypo_input, hyper_input])                

        phi_layer = []
        for i in range(phi_k):
            phi_layer.append(Dense(300, activation=None, use_bias=False,                                                               
                                   name='Phi%d' % (i+1)) (hypo_embedding))

        # either merge phi layers in 1 or flatten single phi projection
        if phi_k == 1:
            # flatten tensors
            phi = Flatten(name='Flatten_Phi')(phi_layer[0])        
        else:
            phi = Concatenate(axis=1, name='Concat_Phi')(phi_layer)


        # compute hypernym similarity to each projection
        phi_hyper  = Dot(axes=-1, normalize=True, name='SimHyper')([phi, hyper_embedding])        

        prediction = Dense(1, activation="sigmoid", name='Prediction',
                           use_bias=True, 
                           kernel_constraint= ForceToOne(),
                           bias_initializer=Zeros(),                                                              
                           ) (phi_hyper)

        model = Model(inputs=[hypo_input, hyper_input], outputs=prediction)    

        adam = Adam(lr=self.lr)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        return model
        

class YamaneEnsemble(Sequence):
    def __init__(self, args):
        # contains important bits such as the tokeniser, batch augmentation logic, etc.
        self.data = args['data'] 
        # this object generates the predictions from a model's learned parameters
        self.evaluator = yamane_evaluator.YamaneEvaluator(self.data, self.ensemble) 
        
        self.reset_ensemble(args)        
                
            
    def reset_ensemble(self, args):        
        self.cluster_args = {k:v for k, v in args.items() if k in ['embeddings_layer', 'lr']}
        
        # pass embeddings layer to avoid having to allocate large embeddings for every cluster        
        self.lambda_c = args['lambda_c']
        self.negative_sample_n = args['negative_sample_n']
        self.epochs = args['epochs'] 
        self.lr = args['lr']
        self.save_path = args['save_path']
        self.patience = args['patience']
                
        # initialise internal list of clusters
        self.ensemble = []        
        self.append(YamaneCluster(self.cluster_args))
                
        self.evaluator.set_ensemble(self.ensemble)
        
    
    def __len__(self):
        return len(self.ensemble)
 
    def append(self, item):
        self.ensemble.append(item)
 
    def remove(self, item):
        self.ensemble.remove(item)
 
    def __repr__(self):
        return str(self.ensemble)
 
    def __getitem__(self, sliced):
        return self.ensemble[sliced]
            
    
    def get_best_cluster(self, index, query_input, hyper_input, is_test=False):
        z_i = 0
        
        sim = list(map(lambda x: x().predict([query_input, hyper_input]).flatten()[0], self.ensemble))                    
        max_sim = np.argmax(sim)
        
        if is_test:
            return max_sim
        
        if (sim[max_sim] < self.lambda_c):
            # similarity is less than a threshold; create new cluster            
            self.append(YamaneCluster(self.cluster_args))
            #print ("Created cluster #: %d" % (len(self.ensemble)))
            z_i = len(self.ensemble) - 1
            self.sample_clusters[index] = z_i
        else:
            z_i = max_sim
            self.sample_clusters[index] = z_i                
        
        return z_i
    
    
    def get_best_test_loss(self, query_input, hyper_input, y_input):        
        
        losses = list(map(lambda x: x().test_on_batch([query_input, hyper_input], y_input)[0], self.ensemble))
        min_loss_cluster = np.argmin(losses)
        # return cluster registering lowest loss and loss value
        return min_loss_cluster, losses[min_loss_cluster]
                
    
    # do NOT call fit twice unless ensemble is instantiated afresh
    # it will re-allocate all samples to the first cluster 
    def fit(self, train_data, test_data):

        # maintain internal scorer to validate test_data
        test_tuples = self.data.token_to_words(test_data)
        scorer = semeval_eval.HypernymEvaluation(test_tuples)

        print ("Fitting model with following parameters: lambda_c=%0.2f; epochs=%d; negative_count=%d; lr=%0.5f" %\
              (self.lambda_c, self.epochs, self.negative_sample_n, self.lr))
        
        # this list stores which cluster each training sequence pertains to
        self.sample_clusters = np.zeros(len(train_data), dtype='int32')
        
        indices = np.arange(len(train_data))    
        validation_indices = np.arange(len(test_data))
        
        # initialise each training sample to cluster 0
        self.sample_clusters[indices] = 0        
        
        # initialise history object.  We maintain both cluster and ensemble-wide statistics
        self.history = {metric:[] for metric in ['epoch', 'loss', 'test_loss','clusters','MAP','MRR']}
        
        self.best_MAP = 0.
        self.no_gain_n = 0
                
        for epoch in range(self.epochs):
            # reset loss and update counts
            for c in self.ensemble:
                c.before_epoch()            

            # shuffle training samples in place
            np.random.shuffle(indices)    
            #shuffled_pairs = train_data[indices]        

            for idx, i in enumerate(indices): 
                # print status
                if (idx + 1) % 1000 == 0:
                    print ("Processed ", idx+1, "samples...")                                
                                                                
                # complement single +ve sample with negatives
                query_input, hyper_input, _, y_input =\
                    self.data.get_augmented_batch(train_data[i:i+1], self.negative_sample_n, 1)                    
                    
                # find the cluster to which to assign sample
                # the first element in query_input, hyper_input is the +ve sample
                z_i = self.get_best_cluster(i, query_input[0], hyper_input[0]) 
                cluster = self.ensemble[z_i]
                cluster._loss += cluster().train_on_batch([query_input, hyper_input], y_input)[0]
                cluster._train_update_count += 1
                

            # compute validation loss after first epoch of training is complete
            if (1==0):
                for idx, i in enumerate(validation_indices):                

                    # complement +ve sample with negatives to compare loss with training phase
                    query_input, hyper_input, _, y_input =\
                        self.data.get_augmented_batch(test_data[i:i+1], self.negative_sample_n, 1)                    

                    # calculate the test loss on a test batch augmented with negative samples
                    z_i, min_loss = self.get_best_test_loss(query_input, hyper_input, y_input)
                    cluster = self.ensemble[z_i]
                    cluster._test_loss += min_loss
                    cluster._test_update_count += 1                                    

            # calculate average loss per cluster
            for c in self.ensemble:
                c.after_epoch()            
                                
            # compute MAP on validation set
            predictions = self.evaluator.predict(test_data)
            score_names, all_scores = scorer.get_evaluation_scores(predictions)
            
            scores = {s:0.0 for s in score_names }
            for k in range(len(score_names)):    
                scores[score_names[k]] = float('%.5f' % (sum([score_list[k] for score_list in all_scores]) / len(all_scores)))    

            epoch_test_map = scores['MAP']
            epoch_test_mrr = scores['MRR']
            self.history['MAP'].append(epoch_test_map)
            self.history['MRR'].append(epoch_test_mrr)
                        
            # calculate mean train and test loss across all clusters                        
            self.history['epoch'].append(epoch)
            self.history['clusters'].append(len(self.ensemble))
            
            epoch_training_loss = round(np.mean([c.loss[::-1][0] for c in self.ensemble]), 5)
            self.history['loss'].append(epoch_training_loss)
            
            epoch_test_loss = round(np.mean([c.test_loss[::-1][0] for c in self.ensemble]), 5)
            self.history['test_loss'].append(epoch_test_loss)
                        
            print ("Epoch: %d; Clusters: %d; Training Loss: %0.5f; Test MAP: %0.5f; Test MRR: %0.5f" %\
                       (epoch+1, len(self.ensemble), epoch_training_loss, epoch_test_map, epoch_test_mrr))
            
            # check whether to stop early
            if (epoch_test_map > self.best_MAP):
                self.best_MAP = epoch_test_map
                self.save_model()
                self.no_gain_n = 0
            # execute at least 3 epochs
            elif (epoch >= 3):
                self.no_gain_n += 1
            
            
            if (self.no_gain_n >= self.patience):
                print ("Early Stop invoked at epoch %d" % (epoch+1))
                # load last best model
                self.load_model()
                break
                
            
        print ("Done!")
        
    def save_model(self):
        # save all weights except embeddings which do not change and so don't need to be persistd
        weight_dict = {idx:c.model.get_weights()[1:] for idx, c in enumerate(self.ensemble)}
        np.savez_compressed(self.save_path, weight_dict=weight_dict, clusters=self.sample_clusters)
    
    def load_model(self):
        weight_dict = np.load(self.save_path)
        
        # initialise internal list of clusters
        self.ensemble = []        
        for k, v in weight_dict['weight_dict'].tolist().items():
            # create fresh cluster
            self.append(YamaneCluster(self.cluster_args))
            weights = [self.data.embeddings_matrix] + v
            self.ensemble[k]().set_weights(weights)
        
        self.sample_clusters = weight_dict['clusters']
                                 
        # this object generates the predictions from a model's learned parameters
        self.evaluator.set_ensemble(self.ensemble)
        