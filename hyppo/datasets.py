import csv
import os
from gzip import open as gzopen

import sklearn.model_selection

from .tq import tq


class DataSet:
    def __init__(self, name, data, test_size, random_state=1):
        self.name = name
        self.data = data
        self.LAST_SPLIT = {}
        self.test_size = test_size
        self.random_state = random_state

    def make_split(self):
        split_key = 'default'  # len(dataset), tuple(dataset.columns), test_size
        if split_key in self.LAST_SPLIT:
            # print("Reusing existing train/test split")
            dtrain, dtest = self.LAST_SPLIT[split_key]
        else:
            print("Building a new train/test split ")
            dtrain, dtest = sklearn.model_selection.train_test_split(self.data,
                                                                     test_size=self.test_size,
                                                                     random_state=self.random_state)
            # dtrain, dtest = make_split(dataset, test_size, 1)
            self.LAST_SPLIT = {split_key: (dtrain, dtest)}
        return dtrain, dtest

    def set_split(self, dtrain, dtest):
        self.LAST_SPLIT = {'default': (dtrain, dtest)}
    

class PandasColumn:
    def __init__(self, dataset, col):
        self.dataset = dataset
        self.vocab = None
        self.col = col
        self.col_label = col[:-2] if col[:-2] in self.dataset.data.columns else col

    def build_vocab(self):
        if self.vocab: return self.vocab
        ds = self.dataset.data
        iterator = tq(ds.itertuples(), total=len(ds))
        print(f"Building a vocabulary: {self.col} -> {self.col_label}")
        vocab = {}
        for frame in iterator:
            vocab[self.val(frame)] = self.label(frame)
        self.vocab = vocab
        print(f"Found {len(self.vocab)} classes for {self.col}")
        return self.vocab

    def save_vocab(self, fpath):
        if not self.vocab:
            self.build_vocab()
        with gzopen(os.path.join(fpath, f'{self.dataset.name}-{self.col}.vocab.gz'), 'wt', encoding='utf-8') as vocab_file:
            writer = csv.writer(vocab_file, delimiter='\t')
            for k, v in self.vocab.items():
                writer.writerow((k, v))

    def val(self, frame):
        return getattr(frame, self.col)

    def label(self, frame):
        return getattr(frame, self.col_label)
