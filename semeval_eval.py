#!/usr/bin/env python
import sys
import numpy as np

class HypernymEvaluation:    
    def __init__(self, dataset):        
        self.dataset = dataset                        

    def convert_hypernyms_to_one_line(self):
        #ordered_queries = sorted(list(set(self.dataset[0])))
        ordered_queries = sorted(list(set([x for (x,y) in self.dataset])))        
        one_line = {}
        for w in ordered_queries:
            word_hypernyms = [h for q, h in self.dataset if q == w]
            one_line[w] = word_hypernyms
        return one_line

    # taken from task_scorer.py provided with shared task resources
    def mean_reciprocal_rank(self, r):
        """Score is reciprocal of the rank of the first relevant item
        First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
        Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Mean reciprocal rank
        """
        r = np.asarray(r).nonzero()[0]
        return 1. / (r[0] + 1) if r.size else 0.

    def precision_at_k(self, r, k, n):
        """Score is precision @ k
        Relevance is binary (nonzero is relevant).
        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Precision @ k
        Raises:
            ValueError: len(r) must be >= k
        """
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return (np.mean(r)*k)/min(k,n)
        # Modified from the first version. Now the gold elements are taken into account

    def average_precision(self, r,n):
        """Score is average precision (area under PR curve)
        Relevance is binary (nonzero is relevant).
        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Average precision
        """
        r = np.asarray(r) != 0
        out = [self.precision_at_k(r, k + 1, n) for k in range(r.size)]
        #Modified from the first version (removed "if r[k]"). All elements (zero and nonzero) are taken into account
        if not out:
            return 0.
        return np.mean(out)

    def mean_average_precision(self, r, n):
        """Score is mean average precision
        Relevance is binary (nonzero is relevant).
        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Mean average precision
        """
        return self.average_precision(r,n)

    # predictions is a dictionary whereby key is query term and value is a list of ranked hypernym predictions
    def get_evaluation_scores(self, predictions):
        all_scores = []    
        scores_names = ['MRR', 'MAP', 'P@1', 'P@5', 'P@10']
        for query, gold_hyps in self.convert_hypernyms_to_one_line().items():

            avg_pat1 = []
            avg_pat2 = []
            avg_pat3 = []

            pred_hyps = predictions[query]
            gold_hyps_n = len(gold_hyps)    
            r = [0 for i in range(15)]

            for j in range(len(pred_hyps)):
                # I believe it's not fair to bias evaluation on how many hypernyms were found in gold set
                # if anything a shorter list (ex. because a hypernym is very particular) will already make 
                # it harder for a match to be found but if system returns correct hypernym in second place
                # why should it be ignored?
                if j < gold_hyps_n:
                    pred_hyp = pred_hyps[j]
                    if pred_hyp in gold_hyps:
                        r[j] = 1

            avg_pat1.append(self.precision_at_k(r,1,gold_hyps_n))
            avg_pat2.append(self.precision_at_k(r,5,gold_hyps_n))
            avg_pat3.append(self.precision_at_k(r,10,gold_hyps_n))    
            
            mrr_score_numb = self.mean_reciprocal_rank(r)
            map_score_numb = self.mean_average_precision(r,gold_hyps_n)            
            avg_pat1_numb = sum(avg_pat1)/len(avg_pat1)
            avg_pat2_numb = sum(avg_pat2)/len(avg_pat2)
            avg_pat3_numb = sum(avg_pat3)/len(avg_pat3)

            score_results = [mrr_score_numb, map_score_numb, avg_pat1_numb, avg_pat2_numb, avg_pat3_numb]
            all_scores.append(score_results)
        return scores_names, all_scores