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
import crim_dual_evaluator
import multiprojection_model


class MultiProjModelDual:                
    
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
        
        # build and compile mode
        self.feature_extractor = self.build_feature_extractor()
        self.concept_model, self.entity_model = self.build_model(), self.build_model()
      
        # this object generates the predictions from a model's learned parameters                
        self.evaluator = crim_dual_evaluator.CrimDualEvaluator(self.data, self.feature_extractor, self.concept_model, self.entity_model)        
        
        # maintain history object which contains training metrics
        self.history = {metric:[] for metric in ['epoch', 'loss', 'test_loss', 'MAP', 'MRR']}                                                                           
    
    
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
        
        self.feature_extractor = self.build_feature_extractor()
        self.concept_model, self.entity_model = self.build_model(), self.build_model()
        self.evaluator.set_model(self.feature_extractor, self.concept_model, self.entity_model)        
        
        # reset history object which contains training metrics
        self.history = {metric:[] for metric in ['epoch', 'loss', 'test_loss', 'test_map']}    
    
    
    def build_feature_extractor(self):
        hypo_input  = Input(shape=(1,), name='Hyponym')        
        neg_input = Input(shape=(self.synonym_sample_n,), name='Negative')
        hyper_input = Input(shape=(1,), name='Hypernym')
                        
        hypo_embedding, neg_embedding, hyper_embedding = self.embeddings_layer([hypo_input, neg_input, hyper_input])       

        hypo_embedding  = Dropout(rate=0.3, name='Dropout_Hypo')(hypo_embedding)
        hyper_embedding = Dropout(rate=0.3, name='Dropout_Hyper')(hyper_embedding)        

        phi_layer = []
        for i in range(self.phi_k):
            phi_layer.append(Dense(self.data.embeddings_matrix.shape[1], 
                                   activation=None, use_bias=False,                                
                                   kernel_initializer = multiprojection_model.RandomIdentity(),                               
                                   name='Phi%d' % (i)) (hypo_embedding))

        # either merge phi layers in 1 or flatten single phi projection
        if self.phi_k == 1:
            # flatten tensors
            phi = Flatten(name='Flatten_Phi')(phi_layer[0])
            #hyper_embedding = Flatten(name='Flatten_Hyper')(hyper_embedding)    
        else:
            phi = Concatenate(axis=1)(phi_layer)

        phi = Dropout(rate=0.3, name='Dropout_Phi')(phi)        

        # compute hypernym similarity to each projection
        phi_hyper    = Dot(axes=-1, normalize=True, name='SimHyper')([phi, hyper_embedding])

        if self.phi_k > 1:
            phi_hyper = Flatten(name='Flatten_PhiHyper')(phi_hyper)                                                        

        model = Model(inputs=[hypo_input, neg_input, hyper_input], outputs=phi_hyper)
        return model
    
    
    # build head model
    def build_model(self):
        hypo_input  = Input(shape=(1,), name='Hyponym')        
        neg_input = Input(shape=(self.synonym_sample_n,), name='Negative')
        hyper_input = Input(shape=(1,), name='Hypernym')
                        
        phi_hyper = self.feature_extractor([hypo_input, neg_input, hyper_input])               
        
        prediction = Dense(1, 
                           activation="sigmoid", name='Prediction',
                           use_bias=True,
                           kernel_initializer='random_normal', bias_initializer=Zeros(),                                                              
                           ) (phi_hyper)
                
        model = Model(inputs=[hypo_input, neg_input, hyper_input], outputs=prediction)
        adam = Adam(lr = self.lr, beta_1 = self.beta1, beta_2 = self.beta2, clipnorm=self.clip_value)        
        #adam = Adadelta()
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
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
            
            for b in range(0, len(samples), self.batch_size):
                # get mini batch composed of INDICES
                mini_batch = samples[b:b + self.batch_size]
                
                # ids of training data ids, split by concepts and entities
                mini_concept_ids = np.where(self.data.is_concept[mini_batch]==1)[0]
                mini_entity_ids  = np.where(self.data.is_concept[mini_batch]==0)[0]
                                
                # we need to find the minit batch ids which points to the training data samples
                concept_batch = train_data[mini_batch[mini_concept_ids]]
                entity_batch = train_data[mini_batch[mini_entity_ids]]                
                # train concepts examples
                # complement single +ve sample with negatives
                if concept_batch.shape[0] > 0:
                    query_input, hyper_input, neg_input, y_input =\
                        self.data.get_augmented_batch(concept_batch, self.negative_sample_n, 1)                    
                                                            
                    loss += self.concept_model.train_on_batch([query_input, neg_input, hyper_input], y_input)[0]
                    train_update_count += 1
                
                if entity_batch.shape[0] > 0:
                    query_input, hyper_input, neg_input, y_input =\
                        self.data.get_augmented_batch(entity_batch, self.negative_sample_n, 1)                    
                    
                    loss += self.entity_model.train_on_batch([query_input, neg_input, hyper_input], y_input)[0]
                    train_update_count += 1                                

            
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
            #self.history['test_loss'].append(round(test_loss/test_update_count, 5))            
            self.history['test_loss'].append(0)       
            self.history['MAP'].append(epoch_test_map)
            self.history['MRR'].append(epoch_test_mrr)                        
                                    
            print ("Epoch: %d; Training Loss: %0.5f; Test Loss: %0.5f; Test MAP: %0.5f; Test MRR: %0.5f" %\
                   (epoch+1, round(loss/train_update_count, 5), 0., epoch_test_map, epoch_test_mrr))
                                
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
                #self.load_model()
                break
                                                                                                         
        # loading best model
        print ("Load best model")
        self.load_model()
        print ("Done!")

    def save_model(self):
        # save all weights except embeddings which do not change and so don't need to be persistd
        feature_weights = self.feature_extractor.get_weights()[2:]        
        concept_weights = self.concept_model.get_layer('Prediction').get_weights()
        entity_weights = self.entity_model.get_layer('Prediction').get_weights()
        np.savez_compressed(self.save_path, features=feature_weights, 
                                            concept=concept_weights, 
                                            entity=entity_weights)
    
    def load_model(self):
        weights = np.load(self.save_path)
        self.feature_extractor.set_weights([self.data.embeddings_matrix]*2 + weights['features'].tolist())                        
        # if concept_model, entity_model only use a single projections, persisting and loading the 
        # weight array changes the dimensionality of the lr weights
        
        
        if weights['concept'][0].ndim == 1:            
            self.concept_model.get_layer('Prediction').set_weights(
                [weights['concept'][0][np.newaxis, :], weights['concept'][1]])
            
            self.entity_model.get_layer('Prediction').set_weights(
                [weights['entity'][0][np.newaxis, :], weights['entity'][1]])
        else:
            self.concept_model.get_layer('Prediction').set_weights(weights['concept'])
            self.entity_model.get_layer('Prediction').set_weights(weights['entity'])
                                 
        # this object generates the predictions from a model's learned parameters
        self.evaluator.set_model(self.feature_extractor, self.concept_model, self.entity_model)        
  