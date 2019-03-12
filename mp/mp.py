import itertools
from multiprocessing import Pool


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def groups(n, k, queries):
    works = []
    for chunk in grouper(k, queries):
        works.append(chunk)
        if len(works) >= n:
            yield works
            works = []

    if works:
        yield works


def _mp_initialize(clazz, args):
    global _p
    try:
        if not isinstance(args, (list, tuple)):
            args = [args]
        _p = clazz(*args)
    except Exception:
        import traceback
        traceback.print_exc()


def _mp_process(queries):
    global _p

    ret = {}

    for query in queries:
        cmd = query[0]
        args = query[1]
        if not isinstance(args, (list, tuple)):
            args = [args]
        ret[query] = getattr(_p, cmd)(*args)

    return ret


_pools = []
PROCESSES = 6
CHUNK = 1000


class Runner:
    def __init__(self, clazz, args, n=PROCESSES, chunk=CHUNK):
        global _pools
        self.pool = Pool(n, initializer=_mp_initialize,
                         initargs=(clazz, args))
        _pools.append(self.pool)
        self.processes = n
        self.chunk = chunk

    def calc(self, queries):
        results = {}
        for works in groups(self.processes, self.chunk, queries):
            results_ = self.pool.map_async(_mp_process, works)
            for r in results_.get():
                results.update(r)

        return results

    def close(self):
        self.pool.close()


def runners_close():
    global _pools
    for p in _pools:
        p.close()
    _pools = []
