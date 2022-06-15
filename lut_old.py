import numpy as np
import copy
from sklearn.metrics import accuracy_score

def count_recursive(cnt, x):
    """
    Given an array that stores counts of shape ([2] * (bits + 1)) (the last two dimensions
    for y0 and y1, respectively), increase the count by 1 for given pattern (all but last
    entries of x) and whether it's y0 or y1 (the last entry of x). This function is recursive.

    Parameters
    ==========
    cnt: np.ndarray
        Array that stores counts for a given dataset.
    x: list
        Single training example as list. The last entry is for y.
    """
    if len(x) > 1:
        idx = x.pop(0)
        count_recursive(cnt[int(idx)], x)
    else:
        cnt[int(x[0])] += 1


class Lut:
    def __init__(self, bits):
        self.bits = bits
        self.lut = None
        self.rnd = None
        self.cnt = None
        self.cols = None

    def train(self, training_set, cols=None):
        """
        Train the lut given a training set.

        Parameters
        ==========
        training_set: numpy.ndarray
            Numpy array of shape (N, self.bits + 1) and dtype bool. The last column is for y.
        cols: list, optional
            List of indices to select columns. The lut will then be trained on a training set
            with less columns.
        """
        assert self.lut is None, "Lut is already trained!"
        if cols is not None:
            assert len(cols) == self.bits, f"Number of selected columns has to match bit size"
            self.cols = cols
            cols_ = copy.deepcopy(cols)
            cols_.append(-1)
            training_set_ = training_set[:, cols_]
        else:
            self.cols = list(range(training_set.shape[1] - 1))
            training_set_ = training_set

        cnt = np.zeros([2] * (self.bits + 1), dtype=np.uint32)
        for x in training_set_:
            count_recursive(cnt, list(x))

        cnt_ = cnt.reshape((-1, 2))

        lut = np.zeros(([2] * (training_set_.shape[1] - 1)), dtype=bool)
        rnd = np.zeros_like(lut)

        for i in range(2 ** self.bits):
            if cnt_[i, 0] > cnt_[i, 1]:
                lut.ravel()[i] = 0
                rnd.ravel()[i] = 0
            elif cnt_[i, 0] < cnt_[i, 1]:
                lut.ravel()[i] = 1
                rnd.ravel()[i] = 0
            else:
                lut.ravel()[i] = np.random.choice([0, 1])
                rnd.ravel()[i] = 1

        self.lut = lut
        self.cnt = cnt
        self.rnd = rnd

    def predict(self, data_set):
        assert self.lut is not None, f"Lut not trained yet!"
        preds = np.zeros((data_set.shape[0],), dtype=bool)
        for idx, x in enumerate(data_set):
            cols = copy.deepcopy(self.cols)
            preds[idx] = classify_single_training_example(self, cols, x)
        return preds

    def __repr__(self):
        if self.lut is not None:
            return f"Trained lut, {self.bits} bits"
        else:
            return f"Empty lut, {self.bits} bits"

    def get_lut_table(self):
        if self.lut is not None:
            string = ""
            for i in range(2 ** self.bits):
                bit_string = "0" * (self.bits - len(f"{i:b}")) + f"{i:b}"
                is_rnd = "*" if self.rnd.ravel()[i] else ""
                string += bit_string + "  " + f"{self.lut.ravel()[i]:d}" + is_rnd + "\n"
            return string
        else:
            return f"Lut not trained yet, no table to return"

    def __getitem__(self, idx):
        return self.lut[idx]

    def get_cnt_table(self):
        """
        Returns a string of the count table for the lut.
        """
        assert self.cnt is not None, "Lut not trained yet, no counts to return"
        cnt_ = self.cnt.reshape((-1, 2))
        string = ""
        for i in range(2 ** self.bits):
            bit_string = "0" * (self.bits - len(f"{i:b}")) + f"{i:b}"
            string += bit_string + "  " + f"{cnt_[i, 0]} " + f"{cnt_[i, 1]}" + "\n"
        return string


def classify_single_training_example(lut, cols, training_example):
    """
    Given a single training example and list of column indices that the lut operates on,
    classify that bit pattern.

    Parameters
    lut: Lut
        Lookup table object.
    cols: list
        List of columns that the lut operates on.
    training_example: list
        Single row from a training set.
    """
    if len(cols) > 1:
        idx = cols.pop(0)
        return classify_single_training_example(lut[int(training_example[idx])], cols, training_example)
    else:
        return lut[int(training_example[cols[0]])]


def training_set_from_luts(luts, orig_training_set):
    """
    Given a list of luts trained on a subset of the original dataset, obtain a new training set
    where the y labels are the same and the features come according to what the luts classify.

    Parameters
    ==========
    luts: array-like
        List of luts.
    orig_training_set: np.ndarray
        Original training_set where the luts were trained on.
    """
    training_set = np.zeros((orig_training_set.shape[0], len(luts) + 1), dtype=bool)
    training_set[:, -1] = orig_training_set[:, -1]
    for i in range(len(orig_training_set)):
        for j in range(len(luts)):
            cols = copy.deepcopy(luts[j].cols)
            x = list(orig_training_set[i])
            training_set[i, j] = classify_single_training_example(luts[j], cols, x)

    return training_set


def get_lut_performance(lut, X_train, X_test):
    preds = lut.predict(X_train)
    print(f"Accuracy on training set: {accuracy_score(preds, X_train[:, -1]):.2f}%")
    preds = lut.predict(X_test)
    print(f"Accuracy on test set: {accuracy_score(preds, X_test[:, -1]):.2f}%")
    print(f"{lut.rnd.sum() / len(lut.rnd.ravel()) * 100:.2f}% of lut entries are random")

