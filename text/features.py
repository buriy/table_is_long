import re


def words(text):
    return [p.strip() for p in re.findall('\w+\.*|\W+', text) if p.strip() and not p.strip() in ',:./()-_']


def biwords(wrds):
    return [a + '_' + b for a, b in zip(wrds[:-1], wrds[1:])]


def add_prefix(prefix, wrds):
    return [prefix + p for p in wrds]


def positional(wrds):
    result = []
    for i in range(3):
        if len(wrds) <= i:
            break
        result.append(wrds[i] + ':{}'.format(i))
        result.append(wrds[-i - 1] + ':-{}'.format(i + 1))
    return result
