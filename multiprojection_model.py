#!/usr/bin/env python

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Dot, Flatten, Concatenate, Dropout, Lambda, Subtract

from tensorflow.keras.initializers import Initializer, RandomNormal, Zeros, Ones
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.constraints import UnitNorm, MinMaxNorm

from tensorflow.keras.optimizers import Adam

from tensorflow.python.framework import dtypes

from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np

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
    
    # custom loss function which adds a regularisation term to the loss value
    def custom_loss(self, neg_phi_tensor, lambda_c):    
        def combined_loss(y_true, y_pred):                
            simil = lambda_c * K.mean(neg_phi_tensor ** 2)        
            return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) + simil                        

        return combined_loss
    
    # modification that computes similarity of each synonymy vector against every projection
    # and then calculates average similarity and regularises that
    def build_model(self):
        hypo_input  = Input(shape=(1,), name='Hyponym')
        neg_input = Input(shape=(self.synonym_sample_n,), name='Negative')
        hyper_input = Input(shape=(1,), name='Hypernym')
        

        term_embeddings = Embedding(self.data.embeddings_matrix.shape[0], 
                                    self.data.embeddings_matrix.shape[1], 
                                    input_length=1, name='TermEmbedding')        

        neg_embeddings  = Embedding(self.data.embeddings_matrix.shape[0], 
                                    self.data.embeddings_matrix.shape[1], 
                                    input_length=self.synonym_sample_n, name='NegEmbedding')            

        hypo_embedding  = term_embeddings(hypo_input)    
        neg_embedding   = neg_embeddings(neg_input)
        hyper_embedding = term_embeddings(hyper_input)    

        hypo_embedding  = Dropout(0.3, name='Dropout_Hypo')(hypo_embedding)
        hyper_embedding = Dropout(0,3, name='Dropout_Hyper')(hyper_embedding)
        neg_embedding   = Dropout(0,3, name='Dropout_Neg')(neg_embedding)

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

        phi = Dropout(0.3, name='Dropout_Phi')(phi)

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

        model.get_layer(name='TermEmbedding').set_weights([self.data.embeddings_matrix])    
        model.get_layer(name='NegEmbedding').set_weights([self.data.embeddings_matrix])    
        model.get_layer(name='TermEmbedding').trainable = False
        model.get_layer(name='NegEmbedding').trainable = False

        regul_loss = self.custom_loss(phi_negative, self.lambda_c)
        adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.9, clipnorm=1.)
        model.compile(optimizer=adam, loss=regul_loss, metrics=['accuracy'])
        return model
                
    
    def __init__(self, args):
        # contains important bits such as the tokeniser, batch augmentation logic, etc.
        self.data = args['data']
        
        # model parameters
        self.batch_size        = args['batch_size']
        self.phi_k             = args['phi_k']
        self.lambda_c          = args['lambda_c']
        self.epochs            = args['epochs']        
        self.negative_sample_n = args['negative_sample_n']
        self.synonym_sample_n  = args['synonym_sample_n']
        
        # this object generates the predictions from a model's learned parameters
        self.evaluator = args['evaluator']
        # this object scores the predictions according to MRR, MAP, P@k (k in {1,5,10,15})
        self.scorer = args['scorer']
                                
        # build and compile model
        self.model = self.build_model()
        
        # maintain history object which contains training metrics
        self.history = {metric:[] for metric in ['epoch', 'loss', 'test_loss']}
               
    def fit(self, train_data, test_data):
        
        print ("Fitting model with following parameters: batch_size=%d; phi_k=%d; lambda_c=%0.2f; epochs=%d; negative_count=%d; synonym_count=%d" % (self.batch_size, self.phi_k, self.lambda_c, self.epochs, self.negative_sample_n, self.synonym_sample_n))
                
        samples = np.arange(len(train_data))    
        validation_samples = np.arange(len(test_data))
        
        # initialise history object
        self.history = {metric:[] for metric in ['epoch', 'loss', 'test_loss']}
                
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

            
            self.history['epoch'].append(epoch)
            self.history['loss'].append(round(loss/train_update_count, 5))
            self.history['test_loss'].append(round(test_loss/test_update_count, 5))
            # TODO: write early stopping; computation of MAP
                                    
            print ("Epoch: %d; Training Loss: %0.5f; Test Loss: %0.5f" %  (epoch+1, round(loss/train_update_count, 5), round(test_loss/test_update_count, 5)))

  