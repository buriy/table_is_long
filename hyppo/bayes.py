from __future__ import division

import pickle
import typing
import heapq
from collections import defaultdict
from math import log, exp


class NaiveBayes:
    def __init__(self):
        self.classes = {}
        self.freqs = {}
        self.opts = {
            'smoothing': 1e-10,
            'clcap': None,
            'laplace': 'none'
        }
        self.log_smoothing = log(self.opts['smoothing'])

    def train(self, samples: typing.Iterable, opts):
        """
        samples: Iterable[feats, cl].
        """
        self.opts = opts
        self.log_smoothing = log(self.opts['smoothing'])
        clcap = self.opts['clcap']
        laplace = self.opts['laplace']
        assert laplace in ['none', 'cls', 'all']

        classes, freq = defaultdict(int), defaultdict(lambda: defaultdict(int))
        # orig = defaultdict(lambda: defaultdict(int))
        for feats, cl in samples:
            classes[cl] += 1
            for feat in feats:
                freq[feat][cl] += 1  # count features frequencies
                # orig[cl][feat] += 1  # count features frequencies

        freq = {feat: probs for feat, probs in freq.items() if len(freq[feat]) <= clcap}

        # recalculate cl
        # for cl, feats in orig.items():
        #     for feat,val in feats.items():
        #         if feat in freq:
        #             classes[cl] += val
        #             break

        feats = set()
        for feat in freq.keys():
            for cl in freq[feat]:
                # classes[cl] += 1  # count classes frequencies
                feats.add(feat)

        count = 0
        for cl, val in classes.items():
            count += val
        nfeats = len(feats)
        nclasses = len(classes)

        #print("COUNT={} NFEATS={}".format(count, nfeats))

        for feat in freq.keys():  # normalize features frequencies, converting to logprob
            for cl in freq[feat]:
                if laplace == 'none':
                    freq[feat][cl] = -log(freq[feat][cl]) + log(classes[cl])
                elif laplace == 'cls':
                    freq[feat][cl] = -log(freq[feat][cl]) + log(classes[cl]+nclasses)
                elif laplace == 'all':
                    freq[feat][cl] = -log(freq[feat][cl]+1) + log(classes[cl]+nclasses)

        for cl in classes.keys():  # normalize classes frequencies
            if laplace == 'none':
                classes[cl] = -log(classes[cl]) + log(nfeats)
            else:
                classes[cl] = -log(classes[cl]+nclasses) + log(nfeats)

        # return P(C) and P(O|C)
        self.classes = classes
        self.freqs = freq
        self.meta = {
            'count':  count,
            'nclasses': nclasses,
            'nfeats': nfeats
        }

    def predict_proba_fast(self, feats: typing.Iterable):
        scores = self.classes.copy()

        for feat in feats:
            freqs = self.freqs.get(feat)
            if freqs:
                for cl, prob in self.freqs[feat].items():
                    scores[cl] += prob + self.log_smoothing

        return scores

    def predict_proba(self, feats: typing.List, n=None):
        scores = self.predict_proba_fast(feats)
        lf = (1 + len(feats) / 2.)
        m = min(scores.values())
        scores = {k:v-m for k,v in scores.items()}
        if n is None:
            n = len(scores)
        else:
            best = heapq.nsmallest(n, scores.keys(), key=scores.get)
        vals = [(b, exp(-scores.get(b)/lf)) for b in best]
        return vals

    def predict_true(self, feats: typing.List, true: object):
        scores = self.predict_proba_fast(feats)
        lf = (1 + len(feats)) / 2.
        m = min(scores.values())
        bs = scores[true] - m
        return exp(-bs/lf)

    def predict_one(self, feats: typing.Iterable):
        scores = self.predict_proba_fast(feats)
        b = min(scores.keys(), key=scores.get)  # calculate argmin(-log(C|O))
        return b

    def load(self, f):
        self.classes, self.freqs, self.opts = pickle.load(f)
        self.log_smoothing = log(self.opts['smoothing'])

    def save(self, f):
        pickle.dump((self.classes, self.freqs, self.opts), f, protocol=4)
