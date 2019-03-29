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
from scipy.stats import hmean

import semeval_eval
import crim_evaluator_max

import multiprojection_model



class MultiProjMax(multiprojection_model.MultiProjModel):                
    
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
        self.minimum_patience  = args['minimum_patience']
        self.save_path         = args['save_path']
        self.eval_after_epoch  = args['eval_after_epoch']
                        
        
        # build and compile mode
        self.model = self.build_model()
        # this object generates the predictions from a model's learned parameters
        self.evaluator = crim_evaluator_max.CrimEvaluatorMax(self.data, self.model)        
        
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
        self.minimum_patience  = args['minimum_patience']
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

        # compute mean phi projection
        #phi_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(phi)    

        # compute synonymy similarity to each projection
        phi_negative = Dot(axes=-1, normalize=True, name='SimNeg')([phi, neg_embedding])        

        # compute hypernym similarity to each projection
        phi_hyper    = Dot(axes=-1, normalize=True, name='SimHyper')([phi, hyper_embedding])

        if self.phi_k > 1:
            # find projection which yields highest projection            
            phi_hyper = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(phi_hyper)
            phi_hyper = Flatten(name='Flatten_PhiHyper')(phi_hyper)                    
            phi_negative = Lambda(lambda x: K.max(x, axis=1), name='MeanPhiNeg')(phi_negative)#
            
        
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
    
    
  