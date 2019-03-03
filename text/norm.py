import re


def wnorm(t):
    t = re.sub('(\d),', r'\1.', t).replace('_', '-')
    words = [p.strip() for p in re.findall('\d+\.*|\d+\.\d+|[^\W\d_]+\.*|\s+|[^\w\s]+', t)]
    return ' '.join([p for p in words if p and p not in '-.,:;(){}[]_'])
