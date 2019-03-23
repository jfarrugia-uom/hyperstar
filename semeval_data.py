#!/usr/bin/env python

import numpy as np

from itertools import cycle
from copy import deepcopy

import random

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from collections import defaultdict

import crim_data


class SemevalData(crim_data.CrimData):
    def __init__(self, args):
        
        print("Initialising SemevalData...")
        # embeddings model
        self.w2v = args['w2v']
        # boolean array (1 = True) indicating whether word is concept (1) or entity 
        self.is_concept = np.asarray(args['is_concept'])
        # dictionary of training tuples: keys in {Concept, Entity, Both}
        train = args['train']
        # dictionary of test tuples
        test = args['test']
        # dictionary of validation tuples
        validation = args['validation']                
        vocab = args['vocabulary']
                
        # create emtpy synonym dictionary not to break dependency
        self.synonyms = {}
        
                
        print ("Creating tokenizer")        
        # collect words for exercise                                        
        hyponyms  = list(set([x for x,y in train['Both'] + test['Both'] + validation['Both']]))
        hypernyms = list(set([y for x,y in train['Both'] + test['Both'] + validation['Both']]))
        
        # create dictionary indicating whether hyponym is concept (1) or entity (0)
        self.concept_dictionary = {}
        concepts = set([x for x,y in train['Concept'] + test['Concept'] + validation['Concept']])
        for h in hyponyms:
            self.concept_dictionary[h] = 1 if h in concepts else 0

        # dataset set vocab
        vocab = list(set(hyponyms + hypernyms + vocab))
        vocab_len = len(vocab)
        print ("Dataset vocabulary size is %d" % (vocab_len))                
        
        # create tokenizer from embeddings model
        self.tokenizer = Tokenizer(filters='', lower=True)
        # fit on vocab
        self.tokenizer.fit_on_texts(vocab)
        print ("Vocab size is %d words" % (len(self.tokenizer.index_word)))
        # initialise negative word sampler
        print ("Initialising negative sampler")
        self.negative_sampler = crim_data.make_sampler(list(self.tokenizer.word_index.values()))
        print ("Tokenising all dataset tuples")
        # tokenize dataset -> convert to numbers which will serve as embeddings lookup keys
        
        self.valid_data_token = {}
        self.test_data_token = {}
        self.train_data_token = {}
        for w in ['Concept','Entity','Both']:
            self.valid_data_token[w] = self.tokenizer.texts_to_sequences([[x,y] for x, y  in validation[w] ])
            self.test_data_token[w] = self.tokenizer.texts_to_sequences([[x,y] for x, y  in test[w] ])
            self.train_data_token[w] = self.tokenizer.texts_to_sequences([[x,y] for x, y  in train[w] ])
        
        # convert lists to array
        for w in ['Concept','Entity','Both']:
            self.valid_data_token[w] = np.asarray(self.valid_data_token[w])
            self.test_data_token[w] = np.asarray(self.test_data_token[w])
            self.train_data_token[w] = np.asarray(self.train_data_token[w])
        
        # create hypernym dictionary lookup
        self.hypernym_id_lookup = defaultdict(list)
        for x, y in np.concatenate(( self.valid_data_token['Both'], 
                                     self.test_data_token['Both'], 
                                     self.train_data_token['Both'] )):
            self.hypernym_id_lookup[x].append(y)
        # disable default factory
        self.hypernym_id_lookup.default_factory = None
        
        print ("Creating embeddings matrix")
        # create embeddings matrix
        self.embeddings_matrix = np.zeros( (len(self.tokenizer.index_word) + 1, 300) )
        for k, v in self.tokenizer.index_word.items():
            self.embeddings_matrix[k] = self.w2v[v]
            #vectors should already by nornalised
            #self.embeddings_matrix[k] /= np.linalg.norm(emb_matrix[k])
        print ("Done!")
            
    
    
