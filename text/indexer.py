import gzip
import heapq
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer


class IndexFTS:
    def __init__(self, tokenizer, items={}, fn=None):  # id -> text
        self.tokenizer = tokenizer
        if items:
            self.index(items)
        elif fn:
            self.load(fn)

    def index(self, items):
        self.ids = list(items.keys())
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=self.tokenizer)
        self.vecs = self.vectorizer.fit_transform(items.values())

    def search(self, query, k=3):
        feats = self.vectorizer.transform([query])
        scores = (self.vecs @ feats.T).toarray().flatten()
        results = heapq.nlargest(k, zip(self.ids, scores), key=lambda x: x[1])
        return results

    def save(self, fn):
        with gzip.open(fn, 'wb', compresslevel=1) as f:
            pickle.dump({
                'ids': self.ids,
                'vecs': self.vecs,
                'vectorizer': self.vectorizer
            }, f)

    def load(self, fn):
        with gzip.open(fn, 'rb') as f:
            data = pickle.load(f)
            self.ids = data['ids']
            self.vecs = data['vecs']
            self.vectorizer = data['vectorizer']


def test():
    import pandas
    TEST_DS = {6: 'Это первое предложение. Это оно снова...',
               8: 'Это пример второго предложения, ха-ха-ха',
               9: 'А это третье предложение'}
    TEST_QUERY = 'предложение для примера'
    TESTS = '/tmp'

    from text.tokenize_mystem import tokenize
    idx = IndexFTS(tokenize, TEST_DS)
    scores = idx.search(TEST_QUERY)
    print(pandas.DataFrame(list(zip(*idx.vecs.toarray(), idx.vectorizer.get_feature_names()))))
    idx.save(TESTS / 'index_fts.pkl.gz')
    idx = IndexFTS(fn=TESTS / 'index_fts.pkl.gz')
    scores2 = idx.search(TEST_QUERY)
    assert scores2 == scores
    print(scores)


if __name__ == "__main__":
    test()
