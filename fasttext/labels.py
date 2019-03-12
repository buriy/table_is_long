import pickle
from collections import namedtuple

import pandas

from ..log.printer import PRINTER

Label = namedtuple('Label', ['col', 'code', 'value', 'label'])
Label.__str__ = lambda x: x.value


class Labels:
    def __init__(self, fn=None):
        self.by_label = {}
        self.by_value = {}
        self.by_code = {}
        if fn:
            self.load(fn)

    def load(self, fn):
        data = pandas.from_csv(fn)
        for _, label, col, value, code in data.iteritems():
            self.add_label(label, col, value, code)

    def save_pickle(self, fn):
        with open(str(fn), 'wb') as f:
            pickle.dump(self, f)

    def to_csv(self):
        ldf = pandas.DataFrame([(v.col, v.code, v.value, v.label) for v in self.by_label.values()],
                               columns=['col', 'code', 'value', 'label'])
        ldf = ldf.sort_values(['col', 'code', 'value'])
        return ldf

    def save_csv(self, fn):
        ldf = self.to_csv()
        ldf.to_csv(str(fn), index=False)

    def add_label(self, lname, col, value, code=None):
        lbl = Label(col, code, value, lname)
        self.by_label[lname] = lbl
        self.by_value[col, value] = lbl
        self.by_code[col, code] = lbl


def vallabel(frame, col):
    label = getattr(frame, col)
    if hasattr(frame, col + 'Id'):
        lid = str(getattr(frame, col + 'Id'))
        label = lid + '_' + label
    return ('__label__' + col + '_' + label).replace('\n', '_').replace(' ', '_')


def setup_col(labels, df, col):
    for frame in df.drop_duplicates([col, col + 'Id']).itertuples():
        label = vallabel(frame, col)
        labels.add_label(label, col, getattr(frame, col), getattr(frame, col + 'Id'))


def setup_cols(df, cols):
    labels = Labels()
    for col in cols:
        setup_col(labels, df, col)


def save_labels(labels, path):
    labels.save_csv(path / 'labels.csv')
    labels.save_pickle(path / 'labels.pickle')


def load_labels(path):
    with open(str(path / 'labels.pickle'), 'rb') as f:
        return pickle.load(f)


def save_fasttext(fn, ds, cols):
    with open(str(fn), 'w') as f:
        for frame in PRINTER.itertuples(ds):
            f.write(' '.join([getattr(frame, c) for c in cols]) + '\n')


def load_fasttext(fn, labels, cols=None):
    with open(fn, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            words = line.split()
            lbls = [w for w in words if w.startswith('__label__')]
            text = [w for w in words if not w.startswith('__label__')]
            traits = {}
            for lname in lbls:
                lbl = labels.by_label[lname]
                if lbl.col not in cols:
                    continue
                traits[lbl.col] = lbl
            yield ' '.join(text), traits
