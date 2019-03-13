#!/usr/bin/env python
import codecs
import argparse
import csv
import random
#from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from collections import defaultdict
import numpy as np

from copy import deepcopy

""" 
Arguments will be passed via a dictionary arg
"""

#parser = argparse.ArgumentParser(description='Preparation.')
#parser.add_argument('--w2v',  default='all.norm-sz100-w10-cb0-it1-min100.w2v', nargs='?', help='Path to the word2vec model.')
#parser.add_argument('--seed', default=228, type=int, nargs='?', help='Random seed.')
#args = vars(parser.parse_args())


def read_subsumptions(filename):
    subsumptions = []

    with codecs.open(filename, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in reader:
            subsumptions.append((row[0], row[1]))

    return subsumptions

def read_synonyms(filename, hypernym_word_dict):
    synonyms = defaultdict(lambda: list())

    with codecs.open(filename,encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in reader:
            for word in row[1].split(','):
                if (row[0] in hypernym_word_dict and word not in hypernym_word_dict[row[0]]):
                    synonyms[row[0]].append(word)

    return synonyms

def compute_XZ(subsumptions, synonyms, w2v):
    X_index, Z_all = [], []

    for hyponym, hypernym in subsumptions:
        offset        = len(Z_all)
        word_synonyms = [hyponym] + synonyms[hyponym]

        X_index.append([offset, len(word_synonyms)])

        for synonym in word_synonyms:
            Z_all.append(w2v[synonym])

    return (np.array(X_index, dtype='int32'), np.array(Z_all))

def get_synonymys_having_vectors(orig_synonyms, w2v):    
    synonyms = deepcopy(orig_synonyms) 
    # eliminate OOV from synonym list    
    for k, v in list(synonyms.items()):
        if k not in w2v:
            synonyms.pop(k)
        else:
            for word in v:
                if word not in w2v:
                    v.remove(word)
    return synonyms

"""
This class is responsible for serialising the embeddings for eventual use in model training and evaluation
"""
class Prepare:
    def __init__(self, args):
        self.args = args
        self.RANDOM_SEED = self.args['seed']
        self.hyper_dict = self.args['hyper_dict']
        
        random.seed(self.RANDOM_SEED)
        
        # load vectors
        #self.w2v = KeyedVectors.load_word2vec_format(self.args['w2v'], binary=self.is_binary)
        #w2v = Word2Vec.load_word2vec_format(args['w2v'], binary=True, unicode_errors='ignore')
        #self.w2v.init_sims(replace=True)
        self.w2v = self.args['w2v']
        print('Using %d embeddings dimensions.' % (self.w2v.vectors.shape[1]))

    def get_terms_having_vectors(self, dataset):
        return [(q,h) for q, h in dataset if q in self.w2v and h in self.w2v]                    
    
    def __call__(self, train, test):
        
        #subsumptions_train      = read_subsumptions('subsumptions-train.txt')
        #subsumptions_validation = read_subsumptions('subsumptions-validation.txt')
        #subsumptions_test       = read_subsumptions('subsumptions-test.txt')
        
        # eliminate words which don't have corresponding vectors
        self.subsumptions_train = self.get_terms_having_vectors(train)
        self.subsumptions_test = self.get_terms_having_vectors(test)
                
        self.synonyms = read_synonyms('synonyms.txt', self.hyper_dict)        
        # eliminate OOV from synonym list
        for k, v in list(self.synonyms.items()):
            if k not in self.w2v:
                self.synonyms.pop(k)
            else:
                for word in v:
                    if word not in self.w2v:
                        v.remove(word)
    
        X_index_train, Z_all_train = compute_XZ(self.subsumptions_train, self.synonyms, self.w2v)
        #X_index_validation, Z_all_validation = compute_XZ(subsumptions_validation)
        X_index_test,  Z_all_test  = compute_XZ(self.subsumptions_test, self.synonyms, self.w2v)

        Y_all_train  = np.array([self.w2v[w] for _, w in self.subsumptions_train])
        #Y_all_validation = np.array([w2v[w] for _, w in subsumptions_validation])
        Y_all_test   = np.array([self.w2v[w] for _, w in self.subsumptions_test])

        np.savez_compressed('train.npz', X_index=X_index_train,
                                         Y_all=Y_all_train,
                                         Z_all=Z_all_train)

        #np.savez_compressed('validation.npz', X_index=X_index_validation,
        #                                      Y_all=Y_all_validation,
        #                                      Z_all=Z_all_validation)

        np.savez_compressed('test.npz', X_index=X_index_test,
                                        Y_all=Y_all_test,
                                        Z_all=Z_all_test)

        print('I have %d train, %d test examples.' % (
            Y_all_train.shape[0],
            #Y_all_validation.shape[0],
            Y_all_test.shape[0]))
