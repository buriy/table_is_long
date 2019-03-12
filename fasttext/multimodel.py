import os
import pprint
from pathlib import Path

import tqdm

from ..log.printer import PRINTER
from .ft import FastTextModel
from .labels import Labels, load_labels, save_labels
from ..text.natural_sort import natural_keys


class MultiModel:
    def __init__(self, root):
        self.root = Path(root)
        for p in ['train', 'test', 'results', 'models', 'reports']:
            (self.root / p).mkdir(exist_ok=True, parents=True)
        self.cols = []
        self.labels = Labels()
        self.models = {}

    def find_score(self, fn):
        fn = self.root / 'results' / Path(fn).name.replace('.bin', '.test')
        if not fn.exists():
            return 0
        with fn.open('r') as f:
            s = f.read().split('P@1', 1)[1].strip()
            return float(s.split()[0])

    def find_model(self, col: str) -> Path:
        fn_all = self.root / 'models'
        fn_best = fn_all / (col + '.best.bin')
        score_best = None
        fns = {str(m): self.find_score(m) for m in fn_all.glob(col + '.*.bin')
               if os.path.getsize(str(m)) > 0}
        assert fns, "No model for {}".format(col)
        fns = sorted(fns.items(), key=lambda x: (-int(x[1] * 1000), natural_keys(x[0])))
        if len(fns) > 1:
            pprint.pprint(fns)
        if not fn_best.exists():
            fn_best, score_best = fns[0]
        PRINTER.print("Selected {}, accuracy={} for {}".format(fn_best, score_best or 'unknown', col))
        return fn_best

    def preload_parallel(self, texts, cols):
        return {col: self.models[col].preload_parallel(texts) for col in cols}

    def predict_parallel(self, texts, cols):
        return {col: self.models[col].predict_parallel(texts) for col in cols}

    def predict(self, text, cols):
        return {col: self.models[col].predict(text) for col in cols}

    def predict_values(self, text, cols):
        return {col: self.models[col].predict(text).value for col in cols}

    def predict_codes(self, text, cols):
        return {col: self.models[col].predict(text).code for col in cols}

    def save_cache(self, cols):
        import pickle
        for col in tqdm.tqdm_notebook(cols):
            if self.models[col].cache:
                with open(str(self.root / (col + '.cache')), 'wb') as f:
                    pickle.dump(self.models[col].cache, f)

    def load_cache(self, cols):
        import pickle
        for col in tqdm.tqdm_notebook(cols):
            with open(str(self.root / (col + '.cache')), 'rb') as f:
                self.models[col].cache = pickle.load(f)

    def load(self, cols):
        self.cols = cols
        self.labels = load_labels(self.root)
        for col in sorted(cols):
            model_name = self.find_model(col)
            model = FastTextModel(model_name, self.labels)
            self.models[col] = model

    def save_labels(self):
        save_labels(self.labels, self.root)
