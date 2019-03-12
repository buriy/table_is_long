class Printer:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def set_verbose(self, verbose):
        self.verbose = verbose

    def iterate(self, x):
        if self.verbose:
            from .tq import tq
            return tq(x)
        else:
            return x

    def print(self, *x):
        if self.verbose:
            print(*x)
        else:
            print(*x)

    def itertuples(self, ds):
        if self.verbose:
            from .tq import tq
            return tq(ds.itertuples(), total=len(ds))
        else:
            return ds.itertuples()


PRINTER = Printer()
