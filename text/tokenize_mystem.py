import razdel
from pymystem3 import Mystem

_MYSTEM = None


def tag(word):
    global _MYSTEM
    if _MYSTEM is None:
        _MYSTEM = Mystem()
    processed = _MYSTEM.analyze(word)[0]
    if not processed.get('analysis'):
        return word.strip().lower()
    lemma = processed["analysis"][0]["lex"].lower().strip()
    return lemma


def tokenize(doc):
    return [tag(t.text) for t in razdel.tokenize(doc)]


if __name__ == '__main__':
    print(tokenize('выплаты на второго ребёнка'))
