# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2020/4/29 15:17
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from model import bilstm
import numpy as np
import json
import tensorflow as tf
from sklearn.metrics import precision_score


class Train:

    def __init__(self):
        self.model_path = 'D:\mygit\\tensorflow2.1\model\\bilstm\\bilstm.h5'
        self.embedding_size = 256
        self.unit = 512
        self.batch_size = 256
        self.dropout = 0.3
        self.learning_rate = 0.001
        self.vocab_file = 'D:\mygit\\tf1.0\data\肯定否定\dictionary.json'
        self.char_index = {' ': 0}
        self.sequence = 70
        self.class_size = 15
        self.model = bilstm(self.embedding_size, self.unit, self.batch_size, self.dropout, self.learning_rate,
                            self.vocab_file, self.sequence, self.class_size)

    def convert_vector(self, input_text, limit):
        char_vector = np.zeros((self.sequence), dtype=np.float32)
        count = len(input_text.strip().lower())
        if count > limit:
            count = limit
        for i in range(count):
            if input_text[i] in self.model.char_index.keys():
                char_vector[i] = self.model.char_index[input_text[i]]
        return char_vector

    def train(self, n_epochs=1):
        x = []
        y = []
        x_test = []
        y_test = []
        with open('..\model\data\分类\\train.json', 'r', encoding='utf8') as f:
            for line in f.readlines():
                data = json.loads(line)
                temp = self.convert_vector(data['sentence'].strip(), self.sequence)
                y.append(int(data['label']))
                x.append(temp)

        with open('..\model\data\分类\\dev.json', 'r', encoding='utf8') as f:
            for line in f.readlines():
                data = json.loads(line)
                temp = self.convert_vector(data['sentence'].strip(), self.sequence)
                y_test.append(int(data['label']))
                x_test.append(temp)

        x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=self.sequence)
        self.model.train(n_epochs, x, y)
        self.model.save(self.model_path)

    def predict(self, x):
        x_test = []
        y_test = []
        with open('..\model\data\分类\\dev.json', 'r', encoding='utf8') as f:
            for line in f.readlines():
                data = json.loads(line)
                temp = self.convert_vector(data['sentence'].strip(), self.sequence)
                y_test.append(int(data['label']))
                x_test.append(temp)

        x = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=self.sequence)
        res = self.model.predict(x, self.model_path)
        print(precision_score(y_test, [np.argmax(i) for i in res], average='micro'))


if __name__ == '__main__':
    Train().train(30)
    # Train().predict('如何重塑老城？嘉兴市委书记为何去了这四个地方调研')