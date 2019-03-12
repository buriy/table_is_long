import typing


class Tester:
    def __init__(self, verbose=1):
        self.verbose = verbose

    def show_progress(self, ok, total):
        if self.verbose:
            acc = float(ok) * 100. / total
            print("Accuracy for {} test samples = {:.2f}%".format(total, acc))

    def test(self, samples: typing.Iterable) -> (float, int, typing.List):
        """
        samples: Iterable[frame, y_true, y_pred]
        """
        ok, total, accel = 0, 0, 20000
        errors = []
        for frame, y_true, y_pred in samples:
            if y_true == y_pred:
                ok += 1
            else:
                errors.append((frame, y_true, y_pred))
            total += 1
            if total >= accel:
                self.show_progress(ok, total)
                accel = int(accel * 1.6)
        if int(total * 1.6) != accel:
            self.show_progress(ok, total)
        acc = float(ok) * 100. / total
        return acc, total, errors
