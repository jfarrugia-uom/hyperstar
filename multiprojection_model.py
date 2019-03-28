#!/usr/bin/env python

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Dot, Flatten, Concatenate, Dropout, Lambda, Subtract

from tensorflow.keras.initializers import Initializer, RandomNormal, Zeros, Ones
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.constraints import UnitNorm, MinMaxNorm

from tensorflow.keras.optimizers import Adam, Adadelta

from tensorflow.python.framework import dtypes

from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np

import semeval_eval
import crim_evaluator


# we'll model the embeddings layer as a separate model which we can resuse against the feature extraction
# elements downstream 
def get_embeddings_model(embeddings_matrix, synonym_sample_n):
    hypo_input  = Input(shape=(1,), name='Hyponym')
    neg_input = Input(shape=(synonym_sample_n,), name='Negative')
    hyper_input = Input(shape=(1,), name='Hypernym')


    embeddings_layer_1 = Embedding(embeddings_matrix.shape[0], 
                                   embeddings_matrix.shape[1],
                                   input_length=1, name='TermEmbedding', 
                                   embeddings_constraint = UnitNorm(axis=1))

    embeddings_layer_2 = Embedding(embeddings_matrix.shape[0], 
                                   embeddings_matrix.shape[1], 
                                   input_length=synonym_sample_n, name='NegEmbedding', 
                                   embeddings_constraint = UnitNorm(axis=1))


    hypo_embedding  = embeddings_layer_1(hypo_input)    
    neg_embedding   = embeddings_layer_2(neg_input)
    hyper_embedding = embeddings_layer_1(hyper_input)                    

    embedding_model = Model(inputs=[hypo_input, neg_input, hyper_input], 
                            outputs=[hypo_embedding, neg_embedding, hyper_embedding])

    # inject pre-trained embeddings into this mini, resusable model/layer
    embedding_model.get_layer(name='TermEmbedding').set_weights([embeddings_matrix])
    embedding_model.get_layer(name='TermEmbedding').trainable = False

    embedding_model.get_layer(name='NegEmbedding').set_weights([embeddings_matrix])
    embedding_model.get_layer(name='NegEmbedding').trainable = False

    return embedding_model

# projection initialiser
class RandomIdentity(Initializer):
    def __init__(self, dtype=dtypes.float32, std=0.01):
        self.dtype = dtypes.as_dtype(dtype)
        self.std = std

    
    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        
        rnorm = K.random_normal((shape[-1],shape[-1]), mean=0., stddev=self.std)        
        #identity = K.eye(shape[-1], dtype='float32')        
        rident = tf.eye(shape[-1]) * rnorm
        return rident
    
    def get_config(self):
        return {"dtype": self.dtype.name}

class MultiProjModel:                
    
    def __init__(self, args):
        # contains important bits such as the tokeniser, batch augmentation logic, etc.
        self.data = args['data']
        
        # model parameters
        self.embeddings_layer  = args['embeddings_layer']
        self.batch_size        = args['batch_size']
        self.phi_k             = args['phi_k']
        self.lambda_c          = args['lambda_c']
        self.epochs            = args['epochs']        
        self.negative_sample_n = args['negative_sample_n']
        self.synonym_sample_n  = args['synonym_sample_n']     
        self.lr                = args['lr']
        self.beta1             = args['beta1']
        self.beta2             = args['beta2']
        self.clip_value        = args['clip_value']
        
        # set  patience > epochs to avoid early stop
        self.patience          = args['patience']
        self.save_path         = args['save_path']
        self.eval_after_epoch  = args['eval_after_epoch']
                
        # build the embeddings sub-model
        #self.embeddings_layer = self.get_term_embeddings_model()        
        
        # build and compile mode
        self.model = self.build_model()
        # this object generates the predictions from a model's learned parameters
        self.evaluator = crim_evaluator.CrimEvaluator(self.data, self.model)        
        
        # maintain history object which contains training metrics
        self.history = {metric:[] for metric in ['epoch', 'loss', 'test_loss', 'MAP', 'MRR']}
                               
                                        
    
    # custom loss function which adds a regularisation term to the loss value
    def custom_loss(self, neg_phi_tensor, lambda_c):    
        def combined_loss(y_true, y_pred):                
            simil = lambda_c * K.mean(neg_phi_tensor ** 2)        
            return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) + simil                        

        return combined_loss
    
    
    def reset_model(self, args):                
        # reset model parameters
        self.embeddings_layer  = args['embeddings_layer']
        self.batch_size        = args['batch_size']
        self.phi_k             = args['phi_k']
        self.lambda_c          = args['lambda_c']
        self.epochs            = args['epochs']        
        self.negative_sample_n = args['negative_sample_n']
        self.synonym_sample_n  = args['synonym_sample_n'] 
        self.patience          = args['patience']
        self.save_path         = args['save_path']
        self.eval_after_epoch  = args['eval_after_epoch']
        self.lr                = args['lr']
        self.beta1             = args['beta1']
        self.beta2             = args['beta2']
        self.clip_value        = args['clip_value']
        
        self.model = self.build_model()
        self.evaluator.set_model(self.model)        
        
        # reset history object which contains training metrics
        self.history = {metric:[] for metric in ['epoch', 'loss', 'test_loss', 'test_map']}    
    
    
    # modification that computes similarity of each synonymy vector against every projection
    # and then calculates average similarity and regularises that
    def build_model(self):
        hypo_input  = Input(shape=(1,), name='Hyponym')        
        neg_input = Input(shape=(self.synonym_sample_n,), name='Negative')
        hyper_input = Input(shape=(1,), name='Hypernym')
                        
        hypo_embedding, neg_embedding, hyper_embedding = self.embeddings_layer([hypo_input, neg_input, hyper_input])       

        hypo_embedding  = Dropout(rate=0.3, name='Dropout_Hypo')(hypo_embedding)
        hyper_embedding = Dropout(rate=0.3, name='Dropout_Hyper')(hyper_embedding)
        neg_embedding   = Dropout(rate=0.3, name='Dropout_Neg')(neg_embedding)

        phi_layer = []
        for i in range(self.phi_k):
            phi_layer.append(Dense(self.data.embeddings_matrix.shape[1], 
                                   activation=None, use_bias=False,                                
                                   kernel_initializer = RandomIdentity(),                               
                                   name='Phi%d' % (i)) (hypo_embedding))

        # either merge phi layers in 1 or flatten single phi projection
        if self.phi_k == 1:
            # flatten tensors
            phi = Flatten(name='Flatten_Phi')(phi_layer[0])
            #hyper_embedding = Flatten(name='Flatten_Hyper')(hyper_embedding)    
        else:
            phi = Concatenate(axis=1)(phi_layer)

        phi = Dropout(rate=0.3, name='Dropout_Phi')(phi)

        # compute mean phi projection
        #phi_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(phi)    

        # compute synonymy similarity to each projection
        phi_negative = Dot(axes=-1, normalize=True, name='SimNeg')([phi, neg_embedding])        

        # compute hypernym similarity to each projection
        phi_hyper    = Dot(axes=-1, normalize=True, name='SimHyper')([phi, hyper_embedding])

        if self.phi_k > 1:
            phi_hyper = Flatten(name='Flatten_PhiHyper')(phi_hyper)        
            # in the case of multiple phi, calculate the mean similarity between each projection
            # and negative case
            phi_negative = Lambda(lambda x: K.mean(x, axis=1), name='MeanPhiNeg')(phi_negative)
            
        
        zero_neg = Lambda(lambda x: K.mean(x * 0., axis=-1), name='ZeroPhiNeg') (phi_negative)    
        phi_hyper = Subtract(name='DummySub')([phi_hyper, zero_neg])    

        prediction = Dense(1, 
                           activation="sigmoid", name='Prediction',
                           use_bias=True,                        
                           kernel_initializer='random_normal', bias_initializer=Zeros(),                                                              
                           ) (phi_hyper)


        model = Model(inputs=[hypo_input, neg_input, hyper_input], outputs=prediction)    
        
        regul_loss = self.custom_loss(phi_negative, self.lambda_c)
        adam = Adam(lr = self.lr, beta_1 = self.beta1, beta_2 = self.beta2, clipnorm=self.clip_value)
        #adam = Adadelta()
        model.compile(optimizer=adam, loss=regul_loss, metrics=['accuracy'])
        return model
    
    
    def fit(self, train_data, test_data):
        
        # maintain internal scorer to validate test_data
        test_tuples = self.data.token_to_words(test_data)
        scorer = semeval_eval.HypernymEvaluation(test_tuples)
                
        print ("Fitting model with following parameters:\n batch_size=%d;\n phi_k=%d;\n lambda_c=%0.2f;\n epochs=%d;\n negative_count=%d;\n synonym_count=%d" %\
               (self.batch_size, self.phi_k, self.lambda_c, self.epochs, self.negative_sample_n, self.synonym_sample_n))        
        print ("Optimizer parameters:\n lr=%0.5f;\n beta1=%0.3f;\n beta2=%0.3f;\n clip=%0.2f" % (self.lr, self.beta1, self.beta2, self.clip_value))
        print ("-"*20)
                
        samples = np.arange(len(train_data))    
        validation_samples = np.arange(len(test_data))
        
        # initialise history object
        self.history = {metric:[] for metric in ['epoch', 'loss', 'test_loss', 'MAP', 'MRR']}
                                                 
        self.best_MAP = 0.
        no_gain_n = 0
        
        for epoch in range(self.epochs):
            # reset loss and update counts
            loss = 0.       
            test_loss = 0.

            train_update_count = 0
            test_update_count = 0

            # shuffle training samples in place
            np.random.shuffle(samples)    
            shuffled_pairs = train_data[samples]        

            for b in range(0, len(samples), self.batch_size):
                # product mini-batch, consisting of 32 +ve samples
                batch_32_pairs = shuffled_pairs[b:b + self.batch_size] 

                # complement +ve samples with negatives
                query_input, hyper_input, synonym_input, y_input =\
                    self.data.get_augmented_batch(batch_32_pairs, self.negative_sample_n, self.synonym_sample_n )
                
                loss += self.model.train_on_batch([query_input, synonym_input, hyper_input], y_input)[0]
                train_update_count += 1

            # compute validation loss after first epoch of training is complete
            for b in range(0, len(validation_samples), self.batch_size):
                # product mini-batch, consisting of 32 +ve samples
                batch_32_pairs = test_data[b:b + self.batch_size] 

                # complement +ve samples with negatives to compare loss with training phase
                query_input, hyper_input, synonym_input, y_input =\
                    self.data.get_augmented_batch(batch_32_pairs, self.negative_sample_n, self.synonym_sample_n )

                #loss += keras_model.train_on_batch([query_input, synonym_input, hyper_input], y_input)[0]
                test_loss += self.model.test_on_batch([query_input, synonym_input, hyper_input], y_input)[0]
                test_update_count += 1

            
            if self.eval_after_epoch:
                # compute MAP on validation set
                predictions = self.evaluator.predict(test_data)
                score_names, all_scores = scorer.get_evaluation_scores(predictions)

                scores = {s:0.0 for s in score_names }
                for k in range(len(score_names)):    
                    scores[score_names[k]] = float('%.5f' %\
                                             (sum([score_list[k] for score_list in all_scores]) / len(all_scores)))    

                epoch_test_map = scores['MAP']
                epoch_test_mrr = scores['MRR']
                
            else:
                epoch_test_map = 0.
                epoch_test_mrr = 0.                
                                                                         
            self.history['epoch'].append(epoch)
            self.history['loss'].append(round(loss/train_update_count, 5))
            self.history['test_loss'].append(round(test_loss/test_update_count, 5))            
            self.history['MAP'].append(epoch_test_map)
            self.history['MRR'].append(epoch_test_mrr)                        
                                    
            print ("Epoch: %d; Training Loss: %0.5f; Test Loss: %0.5f; Test MAP: %0.5f; Test MRR: %0.5f" %\
                   (epoch+1, round(loss/train_update_count, 5), 
                             round(test_loss/test_update_count, 5), epoch_test_map, epoch_test_mrr))
                                
            # check whether to stop early
            if (epoch_test_map > self.best_MAP):
                self.best_MAP = epoch_test_map
                self.save_model()
                no_gain_n = 0
            # execute at least 3 epochs
            elif (epoch >= 3):
                no_gain_n += 1
                        
            if (no_gain_n >= self.patience):
                print ("Early Stop invoked at epoch %d" % (epoch+1))
                # load last best model
                if self.best_MAP > 0.:
                    self.load_model()
                break
                                                                                                         
        print ("Done!")

    def save_model(self):
        # save all weights except embeddings which do not change and so don't need to be persistd
        weights = self.model.get_weights()[2:]        
        np.savez_compressed(self.save_path, weights=weights)
    
    def load_model(self):
        weights = np.load(self.save_path)
        self.model.set_weights([self.data.embeddings_matrix]*2 + weights['weights'].tolist())                        
                                 
        # this object generates the predictions from a model's learned parameters
        self.evaluator.set_model(self.model)        
  