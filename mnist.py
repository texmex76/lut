import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import struct
from array import array

def load_binary(img_path, lbl_path):
    with open(lbl_path, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Magic number should be 2049, got {magic}"
        labels = array("B", f.read())

    with open(img_path, "rb") as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Magic number should be 2051, got {magic}"
        image_data = array("B", f.read())

    images = []
    for i in range(size):
        images.append([0] * rows * cols)

    for i in range(size):
        images[i][:] = image_data[i * rows * cols : (i + 1) * rows * cols]

    return images, labels

def load_mnist_binary(pca=False, n_components=8, thresh=0.5):
    train_img_path = r"data/mnist/train-images.idx3-ubyte"
    train_lbl_path = r"data/mnist/train-labels.idx1-ubyte"

    train_img, train_lbl = load_binary(train_img_path, train_lbl_path)

    test_img_path = r"data/mnist/t10k-images.idx3-ubyte"
    test_lbl_path = r"data/mnist/t10k-labels.idx1-ubyte"

    test_img, test_lbl = load_binary(test_img_path, test_lbl_path)

    X_ = np.vstack((train_img, test_img))
    y_ = np.hstack((train_lbl, test_lbl))

    if pca:
        pca_ = PCA(n_components=n_components)
        X_ = pca_.fit_transform(X_)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_tf = scaler.fit_transform(X_)

    X = (X_tf > thresh).astype(bool)
    y = (y_ == 5) | (y_ == 6) | (y_ == 7) | (y_ == 8) | (y_ == 9)

    X_train = X[:60_000]
    X_test = X[60_000:]
    y_train = y[:60_000]
    y_test = y[60_000:]

    return X_train, X_test, y_train, y_test

def load_mnist():
    train_img_path = r"data/mnist/train-images.idx3-ubyte"
    train_lbl_path = r"data/mnist/train-labels.idx1-ubyte"

    train_img, train_lbl = load_binary(train_img_path, train_lbl_path)

    test_img_path = r"data/mnist/t10k-images.idx3-ubyte"
    test_lbl_path = r"data/mnist/t10k-labels.idx1-ubyte"

    test_img, test_lbl = load_binary(test_img_path, test_lbl_path)

    X_train = np.array(train_img)
    X_test = np.array(test_img)
    y_train = np.array(train_lbl)
    y_test = np.array(test_lbl)

    return X_train, X_test, y_train, y_test
