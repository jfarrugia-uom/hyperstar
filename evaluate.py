#!/usr/bin/env python
from batch_sim.nn_vec import nn_vec
import argparse
import csv
import glob
import os
import pickle
import re
import sys
#from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from collections import defaultdict
import numpy as np
from projlearn import MODELS
from multiprocessing import cpu_count

#parser = argparse.ArgumentParser(description='Evaluation.')
#parser.add_argument('--w2v',          default='all.norm-sz100-w10-cb0-it1-min100.w2v', nargs='?', help='Path to the #word2vec model.')
#parser.add_argument('--test',         default='test.npz',              nargs='?', help='Path to the test set.')
#parser.add_argument('--subsumptions', default='subsumptions-test.txt', nargs='?', help='Path to the test subsumptions.')
#parser.add_argument('--non_optimized', action='store_true', help='Disable most similar words calculation optimization.')
#parser.add_argument('--threads',       nargs='?', type=int, default=cpu_count(), help='Number of threads.')
#parser.add_argument('path', nargs='*', help='List of the directories with results.')
#args = vars(parser.parse_args())

#if not len(sys.argv) > 1:
#    print('Usage: %s path...' % (sys.argv[0]), file=sys.stderr)
#    sys.exit(1)

def extract(clusters, Y_hat_clusters):
    cluster_indices = {cluster: 0 for cluster in Y_hat_clusters}

    Y_all_hat = []

    for cluster in clusters:
        Y_hat = Y_hat_clusters[cluster][cluster_indices[cluster]]
        cluster_indices[cluster] += 1

        Y_all_hat.append(Y_hat)

    assert sum(cluster_indices.values()) == len(clusters)

    return np.array(Y_all_hat)

def compute_ats(measures):
    return [sum(measures[j].values()) / len(subsumptions_test) for j in range(len(measures))]

def compute_auc(ats):
    return sum([ats[j] + ats[j + 1] for j in range(0, len(ats) - 1)]) / 2


class Evaluator:
    def __init__(self, args):
        self.args = args;
        
        # load test vectors
        with np.load(self.args['test']) as npz:
            self.X_index_test  = npz['X_index']
            self.Y_all_test    = npz['Y_all']
            self.Z_all_test    = npz['Z_all']

        self.X_all_test  = self.Z_all_test[self.X_index_test[:, 0],   :]
        # load reference for gold-standard validation data fold
        self.subsumptions_test = self.args['subsumptions']
        # load reference to embeddings
        self.w2v = self.args['w2v']
                
        print (len(self.subsumptions_test), self.X_all_test.shape[0])
        # confirm that validation gold data is the same size as equivalent vector data
        assert len(self.subsumptions_test) == self.X_all_test.shape[0]
        
    def __call__(self, model_name):
        predictions = {}
        
        print('Evaluating "%s" on "%s".' % (model_name, self.args['test']), flush=True)
        
        # load k-means model
        kmeans = pickle.load(open(os.path.join(".", 'kmeans.pickle'), 'rb'))
        print('The number of clusters is %d.' % (kmeans.n_clusters), flush=True)

        # partition test data according to k-means model
        clusters_test  = kmeans.predict(self.Y_all_test - self.X_all_test)
        
        try:
            with np.load(os.path.join(".", '%s.test.npz') % model_name) as npz:
                Y_hat_clusters = {int(cluster): npz[cluster] for cluster in npz.files}
        
        except FileNotFoundError:
            Y_hat_clusters = {}

        if kmeans.n_clusters != len(Y_hat_clusters):
            print('Missing the output for the model "%s"!' % model_name, file=sys.stderr, flush=True)
            return predictions
        
        # get estimated hypernyms for each term in test/validation set
        Y_all_hat = extract(clusters_test, Y_hat_clusters)

        # ensure we have the same number of estimates as we have of test terms
        assert len(self.subsumptions_test) == Y_all_hat.shape[0]
        
        # compute unit-norm of hypernym estimates
        Y_all_hat_norm = Y_all_hat / np.linalg.norm(Y_all_hat,axis=1)[:,np.newaxis]
        # find similar words
        print('nn_vec...')
        similar_indices = nn_vec(Y_all_hat_norm, self.w2v.syn0norm, topn=15, sort=True, return_sims=False, 
                                 nthreads=self.args['threads'], verbose=False)
        print('nn_vec results covert...')
        similar_words = [[self.w2v.index2word[ind] for ind in row] for row in similar_indices]
        print('done')
                
        for i, (hyponym, hypernym) in enumerate(self.subsumptions_test):
            predictions[hyponym] = similar_words[i]
        
        return predictions
        
        

"""
We can comment out loading the embeddings since we're keeping a reference of them in the test harness

WD = os.path.dirname(os.path.realpath(__file__))
w2v = KeyedVectors.load_word2vec_format(os.path.join(WD, args['w2v']),
                                        binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)
"""

"""
We can comment this out because: i) we have already removed the OOV terms in the test harness; ii) we're passing directly a memory reference to a test fold 

with open(args['subsumptions']) as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

    for row in reader:
        subsumptions_test.append((row[0], row[1]))


# remove out-of-vocab entries
def confirmVocab(wordList):
    return [*filter(lambda x: x[0] in w2v.vocab and x[1] in w2v.vocab, wordList)]

subsumptions_test = confirmVocab(subsumptions_test)
"""


        
"""

for path in args['path']:
    print('Doing "%s" on "%s" and "%s".' % (path, args['test'], args['subsumptions']), flush=True)

    kmeans = pickle.load(open(os.path.join(path, 'kmeans.pickle'), 'rb'))
    print('The number of clusters is %d.' % (kmeans.n_clusters), flush=True)

    clusters_test  = kmeans.predict(Y_all_test - X_all_test)

    for model in MODELS:
        try:
            with np.load(os.path.join(path, '%s.test.npz') % model) as npz:
                Y_hat_clusters = {int(cluster): npz[cluster] for cluster in npz.files}
        except FileNotFoundError:
            Y_hat_clusters = {}

        if kmeans.n_clusters != len(Y_hat_clusters):
            print('Missing the output for the model "%s"!' % model, file=sys.stderr, flush=True)
            continue

        Y_all_hat = extract(clusters_test, Y_hat_clusters)

        assert len(subsumptions_test) == Y_all_hat.shape[0]

        measures = [{} for _ in range(10)]

        if not args['non_optimized']:
            # normalize Y_all_hat to make dot product equeal to cosine and monotonically decreasing function of euclidean distance
            Y_all_hat_norm = Y_all_hat / np.linalg.norm(Y_all_hat,axis=1)[:,np.newaxis]
            print('nn_vec...')
            similar_indices = nn_vec(Y_all_hat_norm, w2v.syn0norm, topn=10, sort=True, return_sims=False, nthreads=args['threads'], verbose=False)
            print('nn_vec results covert...')
            similar_words = [[w2v.index2word[ind] for ind in row] for row in similar_indices]
            print('done')

        for i, (hyponym, hypernym) in enumerate(subsumptions_test):
            if args['non_optimized']:
                Y_hat  = Y_all_hat[i].reshape(X_all_test.shape[1],)
                actual = [w for w, _ in w2v.most_similar(positive=[Y_hat], topn=10)]
            else:
                actual = similar_words[i]

            for j in range(0, len(measures)):
                measures[j][(hyponym, hypernym)] = 1. if hypernym in actual[:j + 1] else 0.

            if (i + 1) % 100 == 0:
                ats = compute_ats(measures)
                auc = compute_auc(ats)
                ats_string = ', '.join(['A@%d=%.6f' % (j + 1, ats[j]) for j in range(len(ats))])
                print('%d examples out of %d done for "%s/%s": %s. AUC=%.6f.' % (
                    i + 1,
                    len(subsumptions_test),
                    path,
                    model,
                    ats_string,
                    auc),
                file=sys.stderr, flush=True)

        ats = compute_ats(measures)
        auc = compute_auc(ats)
        ats_string = ', '.join(['A@%d=%.4f' % (j + 1, ats[j]) for j in range(len(ats))])
        print('For "%s/%s": overall %s. AUC=%.6f.' % (
            path,
            model,
            ats_string,
            auc),
        flush=True)
"""