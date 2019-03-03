class NullPrinter:
    @staticmethod
    def iterate(x):
        return x

    @staticmethod
    def print(*x):
        pass

    @staticmethod
    def itertuples(ds):
        return ds.itertuples()


class ConsolePrinter:
    @staticmethod
    def iterate(x):
        import tqdm
        return tqdm.tqdm(x)

    @staticmethod
    def print(*x):
        print(*x)

    @staticmethod
    def itertuples(ds):
        import tqdm
        return tqdm.tqdm(ds.itertuples(), total=len(ds))


class NotebookPrinter:
    @staticmethod
    def iterate(x):
        import tqdm
        return tqdm.tqdm_notebook(x)

    @staticmethod
    def print(*x):
        print(*x)

    @staticmethod
    def itertuples(ds):
        import tqdm
        return tqdm.tqdm_notebook(ds.itertuples(), total=len(ds))


PRINTER = NullPrinter()
