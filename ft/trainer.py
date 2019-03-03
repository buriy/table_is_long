import pandas
from sklearn.model_selection import train_test_split

from hydra import printer
from hydra.labels import save_fasttext, vallabel, Labels


def prepare_df(df, get_text):
    df = df.copy()
    df.index = range(len(df))
    if 'TextFT' not in df.columns:
        df['TextFT'] = [get_text(frame) for frame in printer.PRINTER.itertuples(df)]
    return df


class HydraTrain:
    def __init__(self, hydra, get_text):
        self.hydra = hydra
        self.root = self.hydra.root
        self.get_text = get_text

    def setup_labels(self, df, cols_all):
        labels = Labels()
        for col in cols_all:
            for frame in df.drop_duplicates([col]).itertuples():
                label = vallabel(frame, col)
                labels.add_label(label, col, getattr(frame, col), None)
        self.hydra.labels = labels

    def get_subsets(self, df, test_size):
        idx_train, idx_test = train_test_split(df.index, test_size=test_size, random_state=0)  # 50K here!
        subsets = [
            ('train/{}.full', df.iloc[idx_train]),
            ('test/{}.full', df.iloc[idx_test]),
        ]
        if len(idx_train) >= 100000:
            subsets.append(('train/{}.small', df.iloc[idx_train[:50000]]))
        if len(idx_test) >= 100000:
            subsets.append(('test/{}.small', df.iloc[idx_test[:50000]]))

        return subsets

    def save_subsets(self, df, cols_all, test_size=50000, overwrite=False):
        self.hydra.save_labels()
        df = prepare_df(df, self.get_text)
        ft = []
        for frame in printer.PRINTER.itertuples(df):
            ft.append([self.hydra.labels.by_value[col, getattr(frame, col)].label for col in cols_all])
        df_ft = pandas.DataFrame(ft, columns=cols_all)
        df_ft['TextFT'] = df['TextFT']
        # display(df_ft[:2])

        colsets = [[c] for c in cols_all]

        for sname, sdf in self.get_subsets(df_ft, test_size=test_size):
            printer.PRINTER.print("Preparing subset", sname.format('Col'))
            for colset in colsets:
                fn = self.root / sname.format('_'.join(colset))
                if not fn.exists() or overwrite:
                    save_fasttext(str(fn), sdf, colset + ['TextFT'])

    def run_train(self, df):
        raise Exception(f"Run manually using ./train-all.sh \"{self.root}\" fg")


class HydraPredict:
    def __init__(self, hydra, get_text):
        self.hydra = hydra
        self.get_text = get_text

    def predict_cols(self, texts, cols):
        new = {}
        for c in cols:
            new[c + 'Pred'] = []
            new[c + 'PredId'] = []
            new[c + 'PredProb'] = []

        for x in printer.PRINTER.iterate(texts):
            for c in cols:
                pred = self.hydra.models[c].predict(x)
                prob = self.hydra.models[c].predict_p(x)
                new[c + 'Pred'].append(pred.value)
                new[c + 'PredId'].append(pred.code)
                new[c + 'PredProb'].append(prob)
        return new

    def predict_df(self, df, cols):
        df = prepare_df(df, self.get_text)
        self.hydra.preload_parallel(df.TextFT, cols)
        df_new = pandas.DataFrame.from_records(self.predict_cols(df.TextFT, cols), index=df.index)
        df_pred = pandas.concat([df, df_new], axis=1)
        df_pred['Score'] = df_pred[[c + 'PredProb' for c in cols]].replace('', 0.9, regex=False).mean(axis=1)
        return df_pred
