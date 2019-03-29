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
import multiprojection_model


# we'll model the embeddings layer as a separate model which we can resuse against the feature extraction
# elements downstream 
def get_trainable_embeddings_model(embeddings_matrix, synonym_sample_n):
    hypo_input  = Input(shape=(1,), name='Hyponym')    
    # include this input for backward compatability only
    neg_input = Input(shape=(1,), name='Negative')
    hyper_input = Input(shape=(1,), name='Hypernym')


    embeddings_layer_1 = Embedding(embeddings_matrix.shape[0], 
                                   embeddings_matrix.shape[1],
                                   input_length=1, name='TermEmbedding', 
                                   embeddings_constraint = UnitNorm(axis=1))

    hypo_embedding  = embeddings_layer_1(hypo_input)        
    hyper_embedding = embeddings_layer_1(hyper_input)                    

    # negative input will actually be ignored
    embedding_model = Model(inputs=[hypo_input, neg_input, hyper_input], 
                            outputs=[hypo_embedding, hyper_embedding])

    # inject pre-trained embeddings into this mini, resusable model/layer
    embedding_model.get_layer(name='TermEmbedding').set_weights([embeddings_matrix])
    embedding_model.get_layer(name='TermEmbedding').trainable = True
    
    return embedding_model


class TransferModel(multiprojection_model.MultiProjModel):                
    
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
        self.evaluator = crim_evaluator.CrimEvaluator(self.data, self.model) 
        
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
                        
        hypo_embedding, hyper_embedding = self.embeddings_layer([hypo_input, neg_input, hyper_input])       

        hypo_embedding  = Dropout(rate=self.dropout_rate, name='Dropout_Hypo')(hypo_embedding)
        hyper_embedding = Dropout(rate=self.dropout_rate, name='Dropout_Hyper')(hyper_embedding)        

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

        phi = Dropout(rate=self.dropout_rate, name='Dropout_Phi')(phi)
        
        # compute hypernym similarity to each projection
        phi_hyper = Dot(axes=-1, normalize=True, name='SimHyper')([phi, hyper_embedding])

        if self.phi_k > 1:
            phi_hyper = Flatten(name='Flatten_PhiHyper')(phi_hyper)                                                

        prediction = Dense(1, 
                           activation="sigmoid", name='Prediction',
                           use_bias=True,                        
                           kernel_initializer='random_normal', bias_initializer=Zeros(),                                                              
                           ) (phi_hyper)


        model = Model(inputs=[hypo_input, neg_input, hyper_input], outputs=prediction)    
                
        # freeze projection layer      
        for phi_projection in [l for l in model.layers if l.name.startswith('Phi')]:            
            phi_projection.trainable = False
        
        
        adam = Adam(lr = self.lr, beta_1 = self.beta1, beta_2 = self.beta2, clipnorm=self.clip_value)
        #adam = Adadelta()
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
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
        weights = np.load(self.load_path)
        self.model.set_weights([self.data.embeddings_matrix]*1 + weights['weights'].tolist())                        
                                 
        # this object generates the predictions from a model's learned parameters
        self.evaluator.set_model(self.model) 
  