import fastText

from ..mp import mp
from ..log.printer import PRINTER


class FastTextModel:
    def __init__(self, fn, labels):
        fn = str(fn)
        self.fn = fn
        self.model = fastText.load_model(fn)
        self.cache = {}  # text -> predict
        self.labels = labels

    def _predict(self, text):
        r = self.model.predict(text, k=5)
        # print("Calling me.", text, "->", repr(r))
        return r

    def predict(self, text):
        if text not in self.cache:
            self.cache[text] = self._predict(text)
        s0, s1 = self.cache[text]
        return self.labels.by_label[s0[0]]

    def predict_all(self, texts):
        return {text: self.predict(text) for text in texts}

    def preload_parallel(self, texts, n_processes=6, k=100):  # 6, 100
        to_fetch = set()
        for t in texts:
            if t not in self.cache:
                to_fetch.add(t)

        printer.PRINTER.print("Calculating {} new parallel predicts...".format(len(to_fetch)))
        if len(to_fetch) >= k:
            ft = mp.Runner(clazz=self.__class__, args=(self.fn, None), n=n_processes, k=k)
            try:
                updates = ft.calc(('_predict', text) for text in PRINTER.iterate(to_fetch))
                for k, v in updates.items():
                    self.cache[k[1]] = v
            finally:
                ft.close()
        else:
            self.predict_all(to_fetch)

    def predict_parallel(self, texts, n_processes=6, k=100):
        self.preload_parallel(texts, n_processes=n_processes, k=k)
        return self.predict_all(PRINTER.iterate(texts))

    def predict_proba(self, text):
        if text not in self.cache:
            self.cache[text] = self.model.predict(text, k=5)
        s0, s1 = self.cache[text]
        return s1[0] + s1[1] / 2 + s1[2] / 4 + s1[3] / 8 + s1[4] / 16

    def predict_p(self, text):
        if text not in self.cache:
            self.cache[text] = self.model.predict(text, k=5)
        s0, s1 = self.cache[text]
        return s1[0]
