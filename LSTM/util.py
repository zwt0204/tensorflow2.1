# -*- encoding: utf-8 -*-
"""
@File    : util.py
@Time    : 2019/11/18 15:42
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from tensorflow import keras


def get_data(total_words):
    # 加载IMDB数据集，此处的数据采用数字编码，一个数字代表一个单词
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
    return x_train, y_train, x_test, y_test


def word_index_word():
    """
    返回字-数和数-字的对应
    :return:
    """
    word_index = keras.datasets.imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return word_index, reverse_word_index