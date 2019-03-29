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
import crim_evaluator_max
import multiprojection_model


class TransferModelMax(multiprojection_model.MultiProjModel):                
    
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
                
        # this model requires a load path to know where to load the weights from
        self.save_path         = args['save_path']
        self.load_path         = args['load_path']
        self.eval_after_epoch  = args['eval_after_epoch']
        self.dropout_rate      = args['dropout_rate']
                        
        # build new model + compile
        # the new model will have frozen projections and trainable embeddings
        self.model = self.build_model()        
        
        # this object generates the predictions from a model's learned parameters
        self.evaluator = crim_evaluator_max.CrimEvaluatorMax(self.data, self.model) 
        
        # now load the weights of some other model
        self.load_base_model()
        
        # maintain history object which contains training metrics
        self.history = {metric:[] for metric in ['epoch', 'loss', 'test_loss', 'MAP', 'MRR']}
                                                                       
    
    def reset_model(self, args):                
        # reset model parameters
        self.save_path         = args['save_path']
    
    
    # modification that computes similarity of each synonymy vector against every projection
    # and then calculates average similarity and regularises that
    def build_model(self):        
        hypo_input  = Input(shape=(1,), name='Hyponym')        
        neg_input = Input(shape=(self.synonym_sample_n,), name='Negative')
        hyper_input = Input(shape=(1,), name='Hypernym')
                        
        hypo_embedding, neg_embedding, hyper_embedding = self.embeddings_layer([hypo_input, neg_input, hyper_input])                       
        
        hypo_embedding  = Dropout(rate=self.dropout_rate, name='Dropout_Hypo')(hypo_embedding)
        hyper_embedding = Dropout(rate=self.dropout_rate, name='Dropout_Hyper')(hyper_embedding)
        neg_embedding   = Dropout(rate=0.3, name='Dropout_Neg')(neg_embedding)

        phi_layer = []
        for i in range(self.phi_k):
            phi_layer.append(Dense(self.data.embeddings_matrix.shape[1], 
                                   activation=None, use_bias=False,                                                                   
                                   name='Phi%d' % (i)) (hypo_embedding))

        # either merge phi layers in 1 or flatten single phi projection
        if self.phi_k == 1:
            # flatten tensors
            phi = Flatten(name='Flatten_Phi')(phi_layer[0])
            #hyper_embedding = Flatten(name='Flatten_Hyper')(hyper_embedding)    
        else:
            phi = Concatenate(axis=1)(phi_layer)

        phi = Dropout(rate=self.dropout_rate, name='Dropout_Phi')(phi)

        # compute mean phi projection
        #phi_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(phi)    

        # compute synonymy similarity to each projection
        phi_negative = Dot(axes=-1, normalize=True, name='SimNeg')([phi, neg_embedding])        

        # compute hypernym similarity to each projection
        phi_hyper    = Dot(axes=-1, normalize=True, name='SimHyper')([phi, hyper_embedding])

        if self.phi_k > 1:
            phi_hyper = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(phi_hyper)#
            phi_hyper = Flatten(name='Flatten_PhiHyper')(phi_hyper)        
            # in the case of multiple phi, calculate the max similarity between each projection
            # and negative case            
            phi_negative = Lambda(lambda x: K.max(x, axis=1), name='MeanPhiNeg')(phi_negative)#
            
        
        zero_neg = Lambda(lambda x: K.mean(x * 0., axis=-1), name='ZeroPhiNeg') (phi_negative)    
        phi_hyper = Subtract(name='DummySub')([phi_hyper, zero_neg])    

        prediction = Dense(1, 
                           activation="sigmoid", name='Prediction',
                           use_bias=True,                        
                           kernel_initializer='random_normal', bias_initializer=Zeros(),                                                              
                           ) (phi_hyper)


        model = Model(inputs=[hypo_input, neg_input, hyper_input], outputs=prediction)    
        
                
        # freeze projection layer/s
        for phi_projection in [l for l in model.layers if l.name.startswith('Phi')]:            
            phi_projection.trainable = False
                
        regul_loss = self.custom_loss(phi_negative, self.lambda_c)
        adam = Adam(lr = self.lr, beta_1 = self.beta1, beta_2 = self.beta2, clipnorm=self.clip_value)
        #adam = Adadelta()
        model.compile(optimizer=adam, loss=regul_loss, metrics=['accuracy'])
        return model
       

    def save_model(self):
        # load all weights including embeddings which would have been modified
        print ("Saving model to %s now..." % (self.save_path))
        weights = self.model.get_weights()        
        np.savez_compressed(self.save_path, weights=weights)
        print ("Saving model to %s complete." % (self.save_path))
    
    def load_model(self):
        print ("Loading saved model from %s now..." % (self.save_path))
        weights = np.load(self.save_path)
        self.model.set_weights(weights['weights'].tolist())
                                 
        # this object generates the predictions from a model's learned parameters
        self.evaluator.set_model(self.model)        
        
    def load_base_model(self):                
        # projection weights + logisitic regression weights only feature
        weights = np.load(self.load_path)
        weights = weights['weights'].tolist()
        
        # embeddings layers are already loaded with the pre-trained embeddings
        phi_projections = [l for l in self.model.layers if l.name.startswith('Phi')]            
        for idx, phi_projection in enumerate(phi_projections):
            phi_projection.set_weights([weights[idx]])
            
        self.model.get_layer(name='Prediction').set_weights(weights[self.phi_k:])
                                                             
        # this object generates the predictions from a model's learned parameters
        self.evaluator.set_model(self.model) 
  