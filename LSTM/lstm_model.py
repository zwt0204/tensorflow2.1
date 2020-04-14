# -*- encoding: utf-8 -*-
"""
@File    : lstm_model.py
@Time    : 2019/11/19 10:14
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from tensorflow import keras
import json


class RNN(keras.Model):

    def __init__(self, units, vocab_file, trainable=True):
        """
        Cell方式构建多层网络
        :param units:
        """
        self.vocab_file =vocab_file
        self.char_index = {' ': 0}
        self.units = units
        self.load_dict()
        self.vocab_size = len(self.char_index) + 1
        self.embedding_len = 100
        self.sentence_length = 70
        if trainable:
            self.dropout = 0.4
        else:
            self.dropout = 1
        self.classes = 1
        super(RNN, self).__init__()
        # 词向量编码[b, 80 -> [b, 80, 100]
        self.embedding = keras.layers.Embedding(
            self.vocab_size, self.embedding_len,
            input_length=self.sentence_length
        )

        # 构建RNN
        self.rnn = keras.Sequential([
            keras.layers.LSTM(self.units, dropout=self.dropout, return_sequences=True),
            keras.layers.LSTM(self.units, dropout=self.dropout, return_sequences=True),
            keras.layers.LSTM(self.units, dropout=self.dropout),
        ])

        # 构建分类网络 用于将cell的输出特征进行分类，2分类
        # [b, 80, 100] -> [b, 64] -> [b, 1]
        self.outlayer = keras.Sequential([
            keras.layers.Dense(self.units, activation='relu'),
            keras.layers.Dropout(rate=self.dropout),
            keras.layers.ReLU(),
            keras.layers.Dense(self.classes),
        ])

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 3
                i += 1
        self.char_index["<PAD>"] = 0
        self.char_index["<START>"] = 1
        self.char_index["<UNK>"] = 2  # unknown
        self.char_index["<UNUSED>"] = 3

    def call(self, input, training=None):
        # [b, 80]
        x = input
        # embedding [b, 80] -> [b, 80, 100]
        x = self.embedding(x)
        # rnn cell 计算[b, 80, 100] - > [b, 64]
        x = self.rnn(x)
        # 末层最后一个输出作为分类网络的输入: [b, 64] => [b, 1]
        x = self.outlayer(x, training)
        # p(y is pos|x)
        prob = keras.activations.sigmoid(x)
        # prob = tf.nn.softmax(x)
        return prob
