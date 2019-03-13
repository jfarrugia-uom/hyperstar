#!/usr/bin/env python

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


class YamaneEvaluator:
    def __init__(self, data, ensemble):        
        # will need to reference some components in data
        self.data  = data
        # will need model of course but will be create
        self.ensemble = ensemble  
    
    def set_ensemble(self, ensemble):
        self.ensemble = ensemble
        
    def algol(self, word, emb_matrix, dense, cluster_weight, bias, topN):
        word_id = self.data.tokenizer.word_index[word]
        
        # get projected query vector
        y_hat = np.dot(emb_matrix[word_id], dense)
        # unit-norm
        y_hat /= np.linalg.norm(y_hat, axis=1).reshape(-1,1)
        
        # combine projection with cluster weights and add bias terms
        s = np.dot(emb_matrix[1:], y_hat.T)
        sim_matrix = (s.T * cluster_weight) + bias
        # sort in descending order of linear combination of similarity
        best_projection = np.max(sim_matrix, axis=0)
        top_words = np.argsort(best_projection)[::-1][:15]        
                
        
        # reverse indices to words and also returns values
        return list(map(lambda x: self.data.tokenizer.index_word[x+1], top_words)), best_projection[top_words]
        
      
    def predict_word(self, word, topN=15):
        # get embeddings
        #emb_matrix = self.model.get_layer(name='TermEmbedding').get_weights()[0]
        
        emb_matrix = [l for l in self.ensemble[0]().layers if type(l) == Model][0].get_layer(name='TermEmbedding').get_weights()[0]
    
        # extract the Phi matrices out of trained model
        dense = np.zeros((len(self.ensemble), emb_matrix.shape[1], emb_matrix.shape[1]))
        cluster_weight = np.zeros((len(self.ensemble), 1))
        bias = np.zeros((len(self.ensemble), 1))

        for idx, cluster in enumerate(self.ensemble):
            dense[idx] = cluster().get_layer(name='Phi1').get_weights()[0]
            cluster_weight[idx] = cluster().get_layer(name='Prediction').get_weights()[0]
            bias[idx] = cluster().get_layer(name='Prediction').get_weights()[1]    
        
        return self.algol(word, emb_matrix, dense, cluster_weight, bias, topN)[0]
        
    
    def predict(self, dataset, topN=15):
        # given a test dataset, this method will reverse tokens back to words
        # and generate the hypernyms by applying the learned weights on the query vector
        
        queries = list(zip(*self.data.token_to_words(dataset)))[0]
        ordered_queries = sorted(list(set(queries)))
        results = {}
        
        emb_matrix = [l for l in self.ensemble[0]().layers if type(l) == Model][0].get_layer(name='TermEmbedding').get_weights()[0]
    
        # extract the Phi matrices out of trained model
        dense = np.zeros((len(self.ensemble), emb_matrix.shape[1], emb_matrix.shape[1]))
        cluster_weight = np.zeros((len(self.ensemble), 1))
        bias = np.zeros((len(self.ensemble), 1))

        for idx, cluster in enumerate(self.ensemble):
            dense[idx] = cluster().get_layer(name='Phi1').get_weights()[0]
            cluster_weight[idx] = cluster().get_layer(name='Prediction').get_weights()[0]
            bias[idx] = cluster().get_layer(name='Prediction').get_weights()[1]    
        
        for idx, word in enumerate(ordered_queries):        
            if (idx + 1) % 100 == 0:
                print ("Done", idx + 1)
                    
            predicted_hypers = self.algol(word, emb_matrix, dense, cluster_weight, bias, topN)[0]
            results[word] = predicted_hypers
        
        return results
    
    
    # Using clusters learning during training phase to extract vector diff features
    # and find the average diff vector per cluster and the average hypernym vector per cluster
    def get_cluster_features(self, train_dataset, test_dataset):
        # split train and test datasets in hypo vectos and hyper vectors
        X_test = np.zeros((len(test_dataset), 300))
        Y_test = np.zeros((len(test_dataset), 300))

        for idx, sample in enumerate(test_dataset):            
            X_test[idx] = self.data.embeddings_matrix[sample[0]]
            Y_test[idx] = self.data.embeddings_matrix[sample[1]]

        X_train = np.zeros((len(train_dataset), 300))
        Y_train = np.zeros((len(train_dataset), 300))

        for idx, sample in enumerate(train_dataset):    
            X_train[idx] = self.data.embeddings_matrix[sample[0]]
            Y_train[idx] = self.data.embeddings_matrix[sample[1]]
    
        cluster_n = len(Counter(self.ensemble.sample_clusters).keys())
        avg_diff_cluster = np.zeros((cluster_n, 300))
        avg_hyper_cluster = np.zeros((cluster_n, 300))
        for c in range(cluster_n):
            diff_cluster = (Y_train - X_train)[np.where(self.ensemble.sample_clusters == c)]
            hyper_cluster = Y_train[np.where(self.ensemble.sample_clusters == c)]
            avg_diff_cluster[c] = np.mean(diff_cluster, axis=0)
            avg_hyper_cluster[c] = np.mean(hyper_cluster, axis=0)
                                 
        return avg_diff_cluster, avg_hyper_cluster
        
    
    # attempts to assign unseen word-pair to cluster prior to prediction.
    # failed attempt - does not really work
    def cluster_and_predict(self, train_dataset, test_dataset, topN=15):
        queries = list(zip(*self.data.token_to_words(test_dataset)))[0]
        ordered_queries = sorted(list(set(queries)))
        results = {}
        
        emb_matrix = [l for l in self.ensemble[0]().layers if type(l) == Model][0].get_layer(name='TermEmbedding').get_weights()[0]
    
        # extract the Phi matrices out of trained model
        dense = np.zeros((len(self.ensemble), emb_matrix.shape[1], emb_matrix.shape[1]))
        cluster_weight = np.zeros((len(self.ensemble), 1))
        bias = np.zeros((len(self.ensemble), 1))

        for idx, cluster in enumerate(self.ensemble):
            dense[idx] = cluster().get_layer(name='Phi1').get_weights()[0]
            cluster_weight[idx] = cluster().get_layer(name='Prediction').get_weights()[0]
            bias[idx] = cluster().get_layer(name='Prediction').get_weights()[1]          
    
        avg_diff_cluster, avg_hyper_cluster = self.get_cluster_features(train_dataset, test_dataset)
            
        for idx, word in enumerate(ordered_queries):        
            if (idx + 1) % 100 == 0:
                print ("Done", idx + 1)
                    
            # assign word to clusters first
            term_vector = self.data.embeddings_matrix[self.data.tokenizer.word_index[word]]
            cluster_idx = np.argsort(np.diag(cosine_similarity((term_vector + avg_diff_cluster), avg_hyper_cluster)))[::-1][:3]
            predicted_hypers = self.algol(word, emb_matrix, 
                                          dense[cluster_idx], 
                                          cluster_weight[cluster_idx], 
                                          bias[cluster_idx], topN)[0]
            results[word] = predicted_hypers
        
        return results        
          