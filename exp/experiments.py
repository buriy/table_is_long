import csv
import os

import pandas


class Experiments:
    COLS = ('ds', 'name', 'accuracy', 'model_opts', 'train_size', 'test_size', 'voc_size', 'spec', 'filename')

    def __init__(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.fpath = os.path.join(folder, 'experiments.log')

    def get_records(self):
        with open(self.fpath, 'rt', encoding='utf-8') as log:
            reader = csv.reader(log, delimiter='\t')
            rows = list(reader)
        return pandas.DataFrame(rows, columns=self.COLS)

    def add_record(self, info):
        with open(self.fpath, 'at', encoding='utf-8') as log:
            writer = csv.writer(log, delimiter='\t')
            writer.writerow(info)
