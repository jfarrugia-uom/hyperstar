#!/usr/bin/env python

import numpy as np

from itertools import cycle
from copy import deepcopy

import random

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from collections import defaultdict

# borrowed from https://github.com/gbcolborne/hypernym_discovery/blob/master/train.py
def make_sampler(things):
    """ Make generator that samples randomly from a list of things. """

    nb_things = len(things)
    shuffled_things = deepcopy(things)
    for i in cycle(range(nb_things)):
        if i == 0:
            random.shuffle(shuffled_things)
        yield shuffled_things[i]

class CrimData:
    def __init__(self, args):
        # embeddings model
        self.w2v = args['w2v']
        
        # training tuples
        train = args['train']
        # test tuples
        test = args['test']
        # validation tuples
        validation = args['validation']
        # synonyms
        self.synonyms = args['synonyms']
        
        # if set to -1 then we will use full vector space vocab
        # otherwise use indicated size
        self.limited_vocab_n = args['limited_vocab_n']
        if self.limited_vocab_n > -1:
            print ("Creating limited vocabulary of %d" % (self.limited_vocab_n))
            # collect words for exercise
            flat_synonym = [word for v in self.synonyms.values() for word in v]
            hyponyms  = list(set([x for x,y in train + test + validation] ))
            hypernyms = list(set([y for x,y in train + test + validation] ))
            
            # dataset set vocab
            vocab = list(set(hyponyms + hypernyms + flat_synonym))
            vocab_len = len(vocab)
            print ("Dataset vocabulary size is %d" % (vocab_len))
            model_words = list(self.w2v.vocab.keys())
            # sample words from vector space; sample more words than requested to handle collisions with dataset words            
            random_words = np.random.choice(model_words, (self.limited_vocab_n+10000), replace=False)
            vocab = vocab + [w for w in random_words.tolist() if w not in vocab][:self.limited_vocab_n - vocab_len]
            print ("Truncated vocab length is %d" % (len(vocab)))
        else:
            # choose all words in vector space
            vocab = list(self.w2v.vocab.keys())
        
        # create tokenizer from embeddings model
        self.tokenizer = Tokenizer(filters='', lower=False)
        # fit on vocab
        self.tokenizer.fit_on_texts(vocab)
        print ("Vocab size is %d words" % (len(self.tokenizer.index_word)))
        # initialise negative word sampler
        print ("Initialising negative sampler")
        self.negative_sampler = make_sampler(list(self.tokenizer.word_index.values()))
        print ("Tokenising all dataset tuples")
        # tokenize dataset -> convert to numbers which will serve as embeddings lookup keys
        self.all_data_token = self.tokenizer.texts_to_sequences([[x,y] for x, y  in train + test + validation])
        
        # create hypernym dictionary lookup
        self.hypernym_id_lookup = defaultdict(list)
        for x, y in self.all_data_token:
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
            
    # get list of padded synonyms
    def sample_synonyms(self, word_id, sample_length):
        # convert word_id to word to look for in synyony dictionary
        word = self.tokenizer.index_word[word_id]

        if word in self.synonyms:
            _syn = self.synonyms[word]
        else:
            _syn = []

        # convert list to embeddings index array
        syn_list = np.asarray(self.tokenizer.texts_to_sequences([_syn])[0])
        result = np.asarray([])
        # if we have enough synonyms, we can randomly sample length-1 from list and add the hyponym itself to 
        # the list
        if (sample_length > 1 and len(syn_list) >= (sample_length-1)):        
            result = np.random.choice(syn_list, sample_length-1, replace=False)
            result = np.append(result, word_id)
        # otherwise, we pick all synyonyms and pad the sequences to match model fixed-input
        else:
            result = np.append(syn_list, word_id)
            result = pad_sequences([result], sample_length, padding='post', value=word_id)        

        # we're expecting 1-D vector 
        return result.flatten()

    def get_negative_random(self, word_id, neg_count):
        neg_samples = []
        while len(neg_samples) < neg_count:
            tmp_neg = next(self.negative_sampler)
            if tmp_neg not in self.hypernym_id_lookup[word_id]:
                neg_samples.append(tmp_neg)

        return neg_samples                
    
    def get_augmented_batch(self, query_batch, neg_count, syn_count):

        # create synonym equivalent in ids, prepending the hyponym to the list of synonyms
        query_input   = np.zeros((len(query_batch) * (neg_count+1), 1), dtype='int32')
        hyper_input   = np.zeros((len(query_batch) * (neg_count+1), 1), dtype='int32')
        synonym_input = np.zeros((len(query_batch) * (neg_count+1), syn_count), dtype='int32')
        y_input = np.zeros(len(query_batch) * (neg_count+1))


        for idx, (query, hyper) in enumerate(query_batch):
            query_input[idx * (neg_count+1)] = np.asarray(query)
            hyper_input[idx * (neg_count+1)] = np.asarray(hyper)
            synonym_input[idx * (neg_count+1)] = self.sample_synonyms(query, syn_count)
            y_input[idx * (neg_count+1)] = 1

            if neg_count > 0:
                negatives = self.get_negative_random(word_id=query, neg_count=neg_count)
                for m, neg in enumerate(negatives):
                    query_input[(idx * (neg_count+1)) + (m + 1)] = np.asarray(query)
                    hyper_input[(idx * (neg_count+1)) + (m + 1)] = np.asarray(neg)
                    synonym_input[(idx * (neg_count+1)) + (m + 1)] = self.sample_synonyms(query, syn_count)

        return query_input, hyper_input, synonym_input, y_input
    
    def token_to_words(self, dataset):
        _q = self.tokenizer.sequences_to_texts(dataset[:,0].reshape(-1,1))
        _h = self.tokenizer.sequences_to_texts(dataset[:,1].reshape(-1,1))

        return list(zip(_q, _h))
    
    
