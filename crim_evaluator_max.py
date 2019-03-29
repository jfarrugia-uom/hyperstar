#!/usr/bin/env python

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


class CrimEvaluatorMax:
    def __init__(self, data, model):        
        # will need to reference some components in data
        self.data  = data
        # will need model of course but will be create
        self.model = model  
    
    def set_model(self, model):
        self.model = model
            
    
    def algol_max(self, word, emb_matrix, dense, cluster_weight, bias, topN):
        word_id = self.data.tokenizer.word_index[word]
        
        # get projected query vector
        y_hat = np.dot(emb_matrix[word_id], dense)
        # unit-norm
        y_hat /= np.linalg.norm(y_hat, axis=1).reshape(-1,1)
                    
        sim_matrix = np.dot(emb_matrix[1:], y_hat.T)    
        max_sim = np.max(sim_matrix, 1).reshape(1,-1)
    
        sim_matrix = np.dot(cluster_weight.T, max_sim) + bias
        
        sorted_sim = np.argsort(sim_matrix.flatten())[::-1][:topN]
        
        # reverse indices to words and also returns values
        return list(map(lambda x: self.data.tokenizer.index_word[x+1], sorted_sim)), sim_matrix.flatten()[sorted_sim]
    
        
      
    def predict_word(self, word, topN=15):
        # get embeddings
        #emb_matrix = self.model.get_layer(name='TermEmbedding').get_weights()[0]
        
        emb_matrix = [l for l in self.model.layers if type(l) == Model][0].get_layer(name='TermEmbedding').get_weights()[0]
                
        
        # get transformation matrices
        dense = [l.get_weights()[0] for l in self.model.layers if type(l) == Dense and l.name.startswith('Phi') ]
        dense = np.asarray(dense)

        # extract affine transform layer weights
        cluster_weight = self.model.get_layer(name='Prediction').get_weights()[0]
        bias = self.model.get_layer(name='Prediction').get_weights()[1]
        
        return self.algol_max(word, emb_matrix, dense, cluster_weight, bias, topN)[0]
        
    
    def predict(self, dataset, topN=15):
        # given a test dataset, this method will reverse tokens back to words
        # and generate the hypernyms by applying the learned weights on the query vector
        
        queries = list(zip(*self.data.token_to_words(dataset)))[0]
        ordered_queries = sorted(list(set(queries)))
        results = {}
        
        # get embeddings
        #emb_matrix = self.model.get_layer(name='TermEmbedding').get_weights()[0]
        
        emb_matrix = [l for l in self.model.layers if type(l) == Model][0].get_layer(name='TermEmbedding').get_weights()[0]                
        
        # get transformation matrices
        dense = [l.get_weights()[0] for l in self.model.layers if type(l) == Dense and l.name.startswith('Phi') ]
        dense = np.asarray(dense)

        # extract affine transform layer weights
        cluster_weight = self.model.get_layer(name='Prediction').get_weights()[0]
        bias = self.model.get_layer(name='Prediction').get_weights()[1]
        
        for idx, word in enumerate(ordered_queries):        
            if (idx + 1) % 100 == 0:
                print ("Done", idx + 1)
                    
            predicted_hypers = self.algol_max(word, emb_matrix, dense, cluster_weight, bias, topN)[0]
            results[word] = predicted_hypers
        
        return results        
        
          