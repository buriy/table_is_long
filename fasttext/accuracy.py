from collections import namedtuple

Result = namedtuple('Result', ['acc', 'total', 'errors'])


class Accuracy:
    def __init__(self, verbose=1):
        self.verbose = verbose
        self.ok, self.total, self.accel = 0, 0, 20000
        self.errors = []

    def show_progress(self):
        if self.verbose:
            acc = float(self.ok) * 100. / self.total
            print("Accuracy for {} test samples = {:.2f}%".format(self.total, acc))
            self.accel = int(self.accel * 1.6)

    def add(self, frame, y_true, y_pred):
        if y_true == y_pred:
            self.ok += 1
        else:
            self.errors.append((frame, y_true, y_pred))
        self.total += 1
        if self.total >= self.accel:
            self.show_progress()
            self.accel = int(self.accel * 1.6)

    def result(self):
        if int(self.total * 1.6) != self.accel:
            self.show_progress()
        acc = float(self.ok) * 100. / self.total
        return Result(acc, self.total, self.errors)


def get_accuracy(ds, predict):
    metric = Accuracy()
    metric.accel = 1000
    for frame, y_true in ds:
        y_pred = predict(frame)
        metric.add(frame, y_true, y_pred)
    return metric.result()


def get_accuracy3(ds, true, predict, verbose=1):
    metric = Accuracy(verbose=verbose)
    metric.accel = 1000
    for frame in ds:
        y_true = true(frame)
        y_pred = predict(frame)
        metric.add(frame, y_true, y_pred)
    return metric.result()
