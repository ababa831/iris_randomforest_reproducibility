# import matplotlib.pyplot as plt
import random
import sklearn.datasets
import sklearn.model_selection
import sklearn.ensemble
from sklearn.preprocessing import OneHotEncoder
import numpy as np

iris = sklearn.datasets.load_iris()


def evaluate_randomstate_reproducibility(X, y, test_size=0.3):
    train_x, test_x, train_y, test_y = \
        sklearn.model_selection.train_test_split(X,
                                                 y,
                                                 test_size=test_size)

    rf = sklearn.ensemble.RandomForestClassifier()
    rf.fit(train_x, train_y)

    n_trial = 499
    y_pred_1st = rf.predict(test_x)
    errmsg = '同一モデル，入力データで推論したら値が違っちゃったぜ'
    for i in range(n_trial):
        assert (y_pred_1st == rf.predict(test_x)).all(), errmsg


def test_reproducibility_1():
    """
    答えをわざと説明変数に入れてしまう
    """
    features = iris.data
    target = iris.target.reshape(iris.target.shape[0], 1)
    features = np.concatenate([features, target], axis=1)
    evaluate_randomstate_reproducibility(features, iris.target)


def test_reproducibility_2():
    """
    答え（一部誤答あり）をわざと説明変数に入れてしまう
    """
    features = iris.data
    target = iris.target.copy()
    max_target = iris.target.max()
    len_target = target.shape[0]

    # 一部のサンプルに対して誤答を作り出して，説明変数に入れる
    for _ in range(20):
        row_rand = random.randint(0, len_target - 1)
        target[row_rand] = random.randint(0, max_target)
    errmsg = '誤答が含まれてないぞ'
    assert not (target == iris.target).all(), errmsg
    target = target.reshape(len_target, 1)
    features = np.concatenate([features, target], axis=1)

    evaluate_randomstate_reproducibility(features, iris.target)


def test_reproducibility_3():
    """
    答え（一部誤答あり，one-hot形式）
    をわざと説明変数に入れてしまう
    """
    features = iris.data
    target = iris.target.copy()
    max_target = iris.target.max()
    len_target = target.shape[0]

    # 一部のサンプルに対して誤答を作り出す
    for _ in range(20):
        row_rand = random.randint(0, len_target - 1)
        target[row_rand] = random.randint(0, max_target)
    errmsg = '誤答が含まれてないぞ'
    assert not (target == iris.target).all(), errmsg

    # 各特徴に対して0/1値をもたせる
    target = target.reshape(len_target, 1)
    enc = OneHotEncoder(categories="auto", sparse=False, dtype=int)
    onehot_target = enc.fit_transform(target)
    expected_shape = (len_target, max_target + 1)
    errmsg = '想定した特徴量のshapeになっとらんぞ'
    assert onehot_target.shape == expected_shape, errmsg

    # 説明変数に追加する
    features = np.concatenate([features, onehot_target], axis=1)

    evaluate_randomstate_reproducibility(features, iris.target)
