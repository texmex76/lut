import numpy as np
import copy
import multiprocessing
import os
import pdb
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def get_idxs(X, bit_pattern_tiled, N, bits):
    """
    Get indexes of bit pattern. Each row of X corresponds to one bit pattern, e.g.
    `[0, 0, 1]`. The first entry of bit_pattern is `[0, 0, 0]`. So this function
    would return index 1 for `[0, 0, 1]`.

    Parameters
    ==========
    X: np.ndarray
        Dataset of shape (N, bits) and dtype bool.
    bit_pattern_tiled: np.ndarray
        Tiled bit pattern: `np.tile(bit_pattern, (N, 1))`.
    N: int
        Number of examples, i.e. `X.shape[0]`.
    bits:
        Number of bits, i.e. `X.shape[1]`.
    """
    return np.where(
        np.all(bit_pattern_tiled == np.repeat(X, 2 ** bits, axis=0), axis=1,).reshape(
            (N, 2 ** bits)
        )
        == True
    )[1]

def get_lut(indexes, labels, bits):
    """
    Get lookup table of length `2 ** bits` given an array that specifies which bit
    pattern a training example belongs to. For example, if `x=[0, 0, 0]`, then the
    respective entry in `indexes` is 0, if `x=[0, 1, 0]`, then it should be 2.

    Parameters
    ==========
    indexes: array_like, int
        Each entry corresponds to a row in the training set and specifies which
        bit pattern it belongs to.
    labels: array_like, int
        Labels for the training set. Class 0 should be an int < 0, class 1 should be
        an int > 0.
    bits: int
        Number of bits of the lut.
    """
    lut = np.bincount(indexes, weights=labels, minlength=2 ** bits)
    where_rnd = (lut == 0)
    np.put(lut, np.where(where_rnd)[0], np.random.choice([0, 1], size=(where_rnd).sum()))
    np.put(lut, np.where(lut < 0)[0], 0)
    np.put(lut, np.where(lut > 0)[0], 1)
    return np.hstack((lut.astype(bool), where_rnd))

def get_bit_pattern(bits):
    """
    See the example for explanation.

    Parameters
    ==========
    bits: int

    Returns
    =======
    np.ndarray

    Examples
    ========
    >>> get_bit_pattern(3)
    array([[False, False, False],
           [False, False,  True],
           [False,  True, False],
           [False,  True,  True],
           [ True, False, False],
           [ True, False,  True],
           [ True,  True, False],
           [ True,  True,  True]])
    """
    bit_pattern = np.empty((2 ** bits, bits), dtype=bool)
    for i in range(2 ** bits):
        bit_string = np.binary_repr(i, width=bits)
        for j, bit in enumerate(bit_string):
            bit_pattern[i, j] = int(bit)
    return bit_pattern

def get_cols(inp_len, bits):
    """
    Get random assortment of columns for a lut to operate on.

    Parameters
    ==========
    inp_len: int
        Number of luts of previous layer.
    bits: int
        Bit size of the luts.

    Returns
    =======
    np.ndarray
    """
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    return np.random.choice(range(inp_len), size=bits)


class Lut:
    """
    Create a single lut or a network of luts. If an empty list is passed for the
    `hidden_layers` argument, then a single lut will be created.

    Parameters
    ==========
    bits: list of ints
        Number of bits of each individual lut in each layer. When creating a single
        lut, if the number of bits is the same as the number of features, then
        all columns of the features will be used, otherwise a random assortment
        of columns.
    hidden_layers: list of ints
        Hidden layer sizes. The depth of the lut network is determined by how many
        ints are in this list. If empty, a single lut will be created.

    Attributes
    ==========
    cols_arr_: list containing np.ndarrays
        Each array corresponds to a layer and will have shape `(num_luts, bits)`
        and dtype int. The entries specify which columns the luts operate on. For
        example, `cols_arr_[0][1] = array([13, 5, 9])` meaning in the first hidden
        layer, the 2nd lut takes columns 13, 5 and 9 from the training set.

    lut_arr_: list containing np.ndarrays
        Each array corresponds to a layer and will have shape `(num_luts, 2**bits)`.
        A single lut is filled with boolean values. For example, if we have 3 bits
        and `lut_arr_[1][2][0] = True`, that means that in the second hidden layer the
        third lut classifies `[False, False, False]` with `True`.
    rnd_arr_: list containing np.ndarrays
        Each array corresponds to a layer and will have shape `(num_luts, 2**bits)`.
        For each entry in `lut_arr_`, there is a corresponding entry in `rnd_arr_`.
        If the corresponding entry in `rnd_arr_` is `True`, then the lut entry was
        randomly selected during tie-breaking.
    """
    def __init__(self, bits, hidden_layers=[], discard_bad_luts=False, des_acc=None, discard_num=10, patience=10, make_aig=False, discard_randoms=False, verbose=False):
        self.bits = bits
        self.hidden_layers = hidden_layers
        assert len(bits) == len(hidden_layers) + 1

        self.cols_arr_ = []
        self.lut_arr_ = []
        self.rnd_arr_ = []

        self.discard_bad_luts = discard_bad_luts
        self.des_acc = des_acc
        self.discard_num = discard_num
        self.patience = patience
        self.make_aig = make_aig
        self.discard_randoms = discard_randoms
        self.verbose = verbose

    def train(self, X, y, majority_vote=False):
        """
        Train the network of luts or a single lut. The prediction for the training set
        is returned because during training, propagation is already happening and this
        way we don't have to call `predict()` and do it twice.

        Parameters
        ==========
        X: np.ndarray
            Training examples of shape `(N, bits)` and dtype bool.
        y: np.ndarray
            Labels of shape `(N,)` and dtype bool.

        Returns
        =======
        preds_train: np.ndarray
            Predictions for the training set.
        """
        assert X.dtype == bool, f"Dtype of X has to be bool, got {X.dtype}"
        assert y.dtype == bool, f"Dtype of y has to be bool, got {y.dtype}"
        N = X.shape[0]
        y_ = y.copy().astype(int)
        y_[y_ == 0] = -1
        y_[y_ == 1] = 1

        pool = multiprocessing.Pool()

        if len(self.hidden_layers) > 0:
            with tqdm(self.hidden_layers, disable=not self.verbose) as t:
                for j, num_luts in enumerate(t):
                # for j, num_luts in enumerate(tqdm(self.hidden_layers)):
                    bit_pattern = get_bit_pattern(self.bits[j])
                    bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
                    cols = np.random.choice(X.shape[1] if j == 0 else self.hidden_layers[j - 1], size=num_luts*self.bits[j]).reshape((num_luts, self.bits[j]))
                    self.cols_arr_.append(cols)
                    idxs = np.array(
                        pool.starmap(
                            get_idxs,
                            [
                                [
                                    X[:, self.cols_arr_[-1][i]]
                                    if j == 0
                                    else X_[:, self.cols_arr_[-1][i]],
                                    bit_pattern_tiled,
                                    N,
                                    self.bits[j],
                                ]
                                for i in range(num_luts)
                            ],
                        )
                    )
                    tmp = np.array(
                            pool.starmap(
                                get_lut,
                                [[idxs[i], y_, self.bits[j]] for i in range(num_luts)],
                            )
                        )
                    self.lut_arr_.append(tmp[:, :2**self.bits[j]].copy())
                    self.rnd_arr_.append(tmp[:, 2**self.bits[j]:].copy())
                    X_ = np.array([self.lut_arr_[-1][i][idxs[i]] for i in range(num_luts)]).T

                    #####################################################################
                    ##### Discarding bad Luts ###########################################
                    #####################################################################
                    if self.discard_bad_luts:
                        acc_layer = []
                        for i in range(X_.shape[1]):
                            acc_layer.append(accuracy_score(X_[:, i], y))
                        best = 0
                        no_change = 0
                        while(np.mean(acc_layer) < self.des_acc[j] and not no_change > self.patience):
                            t.set_description(f"Layer {j} Acc {np.mean(acc_layer):.4f}")
                            # Delete discard_num worst luts
                            del_idxs = [x[1] for x in sorted(list(zip(acc_layer, range(len(acc_layer)))), key=lambda x: x[0])[:self.discard_num]]
                            self.lut_arr_[-1] = np.delete(self.lut_arr_[-1], del_idxs, axis=0)
                            self.cols_arr_[-1] = np.delete(self.cols_arr_[-1], del_idxs, axis=0)
                            self.rnd_arr_[-1] = np.delete(self.rnd_arr_[-1], del_idxs, axis=0)
                            cols = np.random.choice(X.shape[1] if j == 0 else self.hidden_layers[j - 1], size=self.discard_num*self.bits[j]).reshape((self.discard_num, self.bits[j]))
                            idxs = np.array(
                                pool.starmap(
                                    get_idxs,
                                    [
                                        [
                                            X[:, cols[i]]
                                            if j == 0
                                            else X_[:, cols[i]],
                                            bit_pattern_tiled,
                                            N,
                                            self.bits[j],
                                        ]
                                        for i in range(self.discard_num)
                                    ],
                                )
                            )
                            tmp = np.array(
                                    pool.starmap(
                                        get_lut,
                                        [[idxs[i], y_, self.bits[j]] for i in range(self.discard_num)],
                                    )
                                )
                            self.lut_arr_[-1] = np.vstack((self.lut_arr_[-1], tmp[:, :2**self.bits[j]].copy()))
                            self.cols_arr_[-1] = np.vstack((self.cols_arr_[-1], cols))
                            self.rnd_arr_[-1] = np.vstack((self.rnd_arr_[-1], tmp[:, 2**self.bits[j]:].copy()))
                            idxs = np.array(
                                pool.starmap(
                                    get_idxs,
                                    [
                                        [
                                            X[:, self.cols_arr_[-1][i]]
                                            if j == 0
                                            else X_[:, self.cols_arr_[-1][i]],
                                            bit_pattern_tiled,
                                            N,
                                            self.bits[j],
                                        ]
                                        for i in range(num_luts)
                                    ],
                                )
                            )
                            X_ = np.array([self.lut_arr_[-1][i][idxs[i]] for i in range(num_luts)]).T
                            acc_layer = []
                            for i in range(X_.shape[1]):
                                acc_layer.append(accuracy_score(X_[:, i], y))

                            # To avoid getting stuck
                            current = np.mean(acc_layer)
                            if current > best:
                                best = current
                                no_change = 0
                            else:
                                no_change += 1
                    #####################################################################
                    ##### End Discarding bad Luts #######################################
                    #####################################################################

                    #####################################################################
                    ##### Discarding Randoms ############################################
                    #####################################################################
                    if self.discard_randoms:
                        no_change = 0
                        randoms_old = np.any(self.rnd_arr_[-1], axis=1).sum()
                        randoms = np.any(self.rnd_arr_[-1], axis=1).sum()
                        while(self.rnd_arr_[-1].sum() != 0 and not no_change > self.patience):
                            t.set_description(f"Layer {j} Randoms: {randoms} No change: {no_change}")
                            # Delete luts that have random entries
                            del_idxs = np.any(self.rnd_arr_[-1], axis=1)
                            self.lut_arr_[-1] = np.delete(self.lut_arr_[-1], del_idxs, axis=0)
                            self.cols_arr_[-1] = np.delete(self.cols_arr_[-1], del_idxs, axis=0)
                            self.rnd_arr_[-1] = np.delete(self.rnd_arr_[-1], del_idxs, axis=0)
                            cols = np.random.choice(X.shape[1] if j == 0 else self.hidden_layers[j - 1], size=randoms*self.bits[j]).reshape((randoms, self.bits[j]))
                            idxs = np.array(
                                pool.starmap(
                                    get_idxs,
                                    [
                                        [
                                            X[:, cols[i]]
                                            if j == 0
                                            else X_[:, cols[i]],
                                            bit_pattern_tiled,
                                            N,
                                            self.bits[j],
                                        ]
                                        for i in range(randoms)
                                    ],
                                )
                            )
                            tmp = np.array(
                                    pool.starmap(
                                        get_lut,
                                        [[idxs[i], y_, self.bits[j]] for i in range(randoms)],
                                    )
                                )
                            self.lut_arr_[-1] = np.vstack((self.lut_arr_[-1], tmp[:, :2**self.bits[j]].copy()))
                            self.cols_arr_[-1] = np.vstack((self.cols_arr_[-1], cols))
                            self.rnd_arr_[-1] = np.vstack((self.rnd_arr_[-1], tmp[:, 2**self.bits[j]:].copy()))
                            idxs = np.array(
                                pool.starmap(
                                    get_idxs,
                                    [
                                        [
                                            X[:, self.cols_arr_[-1][i]]
                                            if j == 0
                                            else X_[:, self.cols_arr_[-1][i]],
                                            bit_pattern_tiled,
                                            N,
                                            self.bits[j],
                                        ]
                                        for i in range(num_luts)
                                    ],
                                )
                            )
                            X_ = np.array([self.lut_arr_[-1][i][idxs[i]] for i in range(num_luts)]).T

                            randoms = np.any(self.rnd_arr_[-1], axis=1).sum()
                            if randoms >= randoms_old:
                                no_change += 1
                            randoms_old = randoms
                    #####################################################################
                    ##### End Discarding Randoms ########################################
                    #####################################################################

                    #####################################################################
                    ##### Make AIG ######################################################
                    #####################################################################
                    if self.make_aig:
                        iteration = 0
                        while(np.sum(self.lut_arr_[-1].sum(1) == 1) < num_luts):
                            # Delete luts that are not AIGs
                            del_idxs = np.where(self.lut_arr_[-1].sum(1) != 1)[0]
                            num_discarded = len(del_idxs)
                            frac_non_aig = num_discarded / num_luts
                            t.set_description(f"Layer {j} Iter {iteration} Frac non AIG {frac_non_aig:.3f}")
                            self.lut_arr_[-1] = np.delete(self.lut_arr_[-1], del_idxs, axis=0)
                            self.cols_arr_[-1] = np.delete(self.cols_arr_[-1], del_idxs, axis=0)
                            self.rnd_arr_[-1] = np.delete(self.rnd_arr_[-1], del_idxs, axis=0)
                            cols = np.random.choice(X.shape[1] if j == 0 else self.hidden_layers[j - 1], size=num_discarded*self.bits[j]).reshape((num_discarded, self.bits[j]))
                            idxs = np.array(
                                pool.starmap(
                                    get_idxs,
                                    [
                                        [
                                            X[:, cols[i]]
                                            if j == 0
                                            else X_[:, cols[i]],
                                            bit_pattern_tiled,
                                            N,
                                            self.bits[j],
                                        ]
                                        for i in range(num_discarded)
                                    ],
                                )
                            )
                            tmp = np.array(
                                    pool.starmap(
                                        get_lut,
                                        [[idxs[i], y_, self.bits[j]] for i in range(num_discarded)],
                                    )
                                )
                            self.lut_arr_[-1] = np.vstack((self.lut_arr_[-1], tmp[:, :2**self.bits[j]].copy()))
                            self.cols_arr_[-1] = np.vstack((self.cols_arr_[-1], cols))
                            self.rnd_arr_[-1] = np.vstack((self.rnd_arr_[-1], tmp[:, 2**self.bits[j]:].copy()))
                            idxs = np.array(
                                pool.starmap(
                                    get_idxs,
                                    [
                                        [
                                            X[:, self.cols_arr_[-1][i]]
                                            if j == 0
                                            else X_[:, self.cols_arr_[-1][i]],
                                            bit_pattern_tiled,
                                            N,
                                            self.bits[j],
                                        ]
                                        for i in range(num_luts)
                                    ],
                                )
                            )
                            X_ = np.array([self.lut_arr_[-1][i][idxs[i]] for i in range(num_luts)]).T
                            iteration += 1
                    #####################################################################
                    ##### End Make AIG ##################################################
                    #####################################################################

            cols = np.random.choice(self.hidden_layers[-1], size=self.bits[-1])
            self.cols_arr_.append(cols)
            bit_pattern = get_bit_pattern(self.bits[-1])
            bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
            idxs = get_idxs(
                X_[:, self.cols_arr_[-1]], bit_pattern_tiled, N, self.bits[-1]
            )
            tmp = get_lut(idxs, y_, self.bits[-1])
            self.lut_arr_.append(tmp[:2**self.bits[-1]].copy())
            self.rnd_arr_.append(tmp[2**self.bits[-1]:].copy())
            preds_train = self.lut_arr_[-1][idxs]

            #####################################################################
            ##### Majority Vote #################################################
            #####################################################################
            if majority_vote:
                Xa = X_.copy().astype(int)
                Xa[Xa == 0] = -1
                Xa[Xa == 1] = 1
                Xa = Xa.sum(1)
                Xa[Xa > 0] = 1
                Xa[Xa < 0] = 0
                Xa[Xa == 0] = 0 # for now, this'll do
                Xa = Xa.astype(bool)
                preds_train = Xa
            #####################################################################
            ##### End Majority Vote #############################################
            #####################################################################

            return preds_train
        else:
            if X.shape[1] == self.bits[0]:
                self.cols_arr_.append(np.arange(X.shape[1]))
            else:
                self.cols_arr_.append(
                    np.random.choice(X.shape[1], size=self.bits[0])
                )
            bit_pattern = get_bit_pattern(self.bits[0])
            bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
            idxs = get_idxs(
                X[:, self.cols_arr_[-1]], bit_pattern_tiled, N, self.bits[0]
            )
            tmp = get_lut(idxs, y_, self.bits[0])
            self.lut_arr_.append(tmp[:2**self.bits[0]].copy())
            self.rnd_arr_.append(tmp[2**self.bits[0]:].copy())
            preds_train = self.lut_arr_[-1][idxs]
            return preds_train


    def predict(self, X, majority_vote=False):
        """
        Predict using the lut classifier.

        Parameters
        ==========
        X: np.ndarray
            The input data of shape `(N, bits)` and dtype bool.

        Returns
        =======
        preds: np.ndarray
        """
        assert X.dtype == bool, f"Dtype of X has to be bool, got {X.dtype}"
        N = X.shape[0]

        if len(self.hidden_layers) == 0:
            X_ = X

        pool = multiprocessing.Pool()
        for j, num_luts in enumerate(self.hidden_layers):
            bit_pattern = get_bit_pattern(self.bits[j])
            bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
            idxs = np.array(
                pool.starmap(
                    get_idxs,
                    [
                        [
                            X[:, self.cols_arr_[0][i]]
                            if j == 0
                            else X_[:, self.cols_arr_[j][i]],
                            bit_pattern_tiled,
                            N,
                            self.bits[j],
                        ]
                        for i in range(num_luts)
                    ],
                )
            )
            X_ = np.array([self.lut_arr_[j][i][idxs[i]] for i in range(num_luts)]).T

        bit_pattern = get_bit_pattern(self.bits[-1])
        bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
        idxs = get_idxs(X_[:, self.cols_arr_[-1]], bit_pattern_tiled, N, self.bits[-1])
        preds = self.lut_arr_[-1][idxs]

        #####################################################################
        ##### Majority Vote #################################################
        #####################################################################
        if majority_vote:
            Xa = X_.copy().astype(int)
            Xa[Xa == 0] = -1
            Xa[Xa == 1] = 1
            Xa = Xa.sum(1)
            Xa[Xa > 0] = 1
            Xa[Xa < 0] = 0
            Xa[Xa == 0] = 0 # for now, this'll do
            Xa = Xa.astype(bool)
            preds = Xa
        #####################################################################
        ##### End Majority Vote #############################################
        #####################################################################

        return preds


    def get_accuracies_per_layer(self, X, y):
        """
        Get accuracies per layer.

        Parameters
        ==========
        X: np.ndarray
            The input data of shape `(N, bits)` and dtype bool.

        Returns
        =======
        acc: list
        """
        assert X.dtype == bool, f"Dtype of X has to be bool, got {X.dtype}"
        N = X.shape[0]

        if len(self.hidden_layers) == 0:
            X_ = X

        pool = multiprocessing.Pool()
        acc = []
        for j, num_luts in enumerate(tqdm(self.hidden_layers)):
            bit_pattern = get_bit_pattern(self.bits[j])
            bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
            idxs = np.array(
                pool.starmap(
                    get_idxs,
                    [
                        [
                            X[:, self.cols_arr_[0][i]]
                            if j == 0
                            else X_[:, self.cols_arr_[j][i]],
                            bit_pattern_tiled,
                            N,
                            self.bits[j],
                        ]
                        for i in range(num_luts)
                    ],
                )
            )
            X_ = np.array([self.lut_arr_[j][i][idxs[i]] for i in range(num_luts)]).T
            acc_layer = []
            for i in range(X_.shape[1]):
                acc_layer.append(accuracy_score(X_[:, i], y))
            acc.append(acc_layer)

        bit_pattern = get_bit_pattern(self.bits[-1])
        bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
        idxs = get_idxs(X_[:, self.cols_arr_[-1]], bit_pattern_tiled, N, self.bits[-1])
        preds = self.lut_arr_[-1][idxs]
        acc.append([accuracy_score(preds, y)])
        return acc
