import csv
import gzip
import os
import pathlib

from .bayes import NaiveBayes
from .experiments import Experiments
from .tester import Tester
from .tq import tq


def tqdm_iter(ds):
    return tq(ds.itertuples(), total=len(ds))


def train_model(col, get_features, opts):
    vocab = col.build_vocab()
    nclasses = len(vocab)
    model = NaiveBayes()
    dtrain, dtest = col.dataset.make_split()
    it = tqdm_iter(dtrain)
    print(f"Training {col.col}, classes={len(vocab)}, opts={opts}")
    train_samples = ((get_features(l), col.val(l)) for l in it)
    model.train(train_samples, opts)
    return model


def save_errors(col, fpath, descr, errors):
    vocab = col.build_vocab()
    with gzip.open(fpath + '.errors.csv.gz', 'wt', encoding='utf-8') as errors_file:
        writer = csv.writer(errors_file, delimiter='\t')
        writer.writerow(['y_true', 'y_pred', 'frame'])
        for f, a, b in errors:
            writer.writerow([vocab[a], vocab[b]] + descr(f))


def load_model(fpath):
    print("Loading {}".format(fpath))
    with gzip.open(fpath, 'rb') as f:
        model = NaiveBayes()
        model.load(f)
    return model


def save_model(model, fpath):
    with gzip.open(fpath, 'wb') as f:
        model.save(f)


def test_model(model, col, get_features, dtest):
    predict = lambda x: model.predict_one(get_features(x))
    test_samples = ((l, col.val(l), predict(l)) for l in tqdm_iter(dtest))
    tester = Tester()
    return tester.test(test_samples)


class TT:
    def __init__(self, fdir):
        self.fdir = pathlib.Path(fdir)
        self.exp = Experiments(fdir)

    def meta(self, col, spec, opts={}):
        ds = col.dataset.name
        name = '{}-{}-{}-{:.3g}'.format(col.col, opts['clcap'], opts['laplace'], opts['smoothing'])
        return {
            'col': col.col,
            'name': name,
            'opts': opts,
            'classes': len(col.build_vocab()),
            'ds': ds,
            'fn': f'{ds}-{name}-{spec}.pickle.gz'
        }

    def train_test_save(self, col, get_features, spec='default', opts={}):  # opts={clcap=None, smoothing=None}
        meta = self.meta(col, spec, opts)
        vocab = col.build_vocab()
        model = train_model(col, get_features, opts)
        save_model(model, self.fdir / meta['fn'])
        return self.test(col, model, get_features, spec, opts=opts)

    def exists(self, fn):
        return os.path.exists(self.fdir / fn)

    def load(self, fn):
        model = load_model(self.fdir / fn)
        return model

    def test(self, col, model, get_features, spec, opts=None, full=False):
        dtrain, dtest = col.dataset.make_split()
        if full:
            dtest = col.dataset.data
        acc, tested, errors = test_model(model, col, get_features, dtest)
        # COLS = ('name', 'accuracy', 'opts', 'train_size', 'test_size', 'voc_size', 'spec', 'filename')
        if opts is not None:
            meta = self.meta(col, spec, opts)
            opts_str = repr(meta['opts'])
            info = (col.dataset.name, meta['col'], acc, opts_str, len(dtrain),
                    len(dtest), meta['classes'], spec, meta['fn'])
            self.exp.add_record(info)
        return model, acc, tested, errors


def multi_test(dataset, models, get_features, combine=lambda x: x):
    def preds(x):
        return [m.predict_one(x) for col, m in models.items()]

    def trues(x):
        return [col.val(x) for col, m in models.items()]

    dtrain, dtest = dataset.make_split()
    test_samples = ((l, combine(trues(l)), combine(preds(get_features(l)))) for l in tqdm_iter(dtest))
    return Tester().test(test_samples)
