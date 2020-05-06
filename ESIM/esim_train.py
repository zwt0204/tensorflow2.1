# -*- encoding: utf-8 -*-
"""
@File    : esim_train.py
@Time    : 2020/5/6 14:26
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from model_esim import ESIM
import tensorflow as tf
import json
import numpy as np


class Train:

    def __init__(self):
        self.model_path = 'D:\mygit\\tensorflow2.1\model\esim\\esim.h5'
        self.embedding_size = 256
        self.unit = 512
        self.batch_size = 256
        self.dropout = 0.1
        self.learning_rate = 0.001
        self.vocab_file = 'D:\mygit\\tf1.0\data\肯定否定\dictionary.json'
        self.char_index = {' ': 0}
        self.sequence = 70
        self.class_size = 2
        self.dense_layer = [128, 256]
        self.model = ESIM(self.embedding_size, self.unit, self.batch_size, self.dropout, self.learning_rate,
                            self.vocab_file, self.sequence, self.class_size, self.dense_layer)

    def convert_vector(self, input_text):
        char_vector = np.zeros((len(input_text)), dtype=np.float32)
        count = len(input_text.strip().lower())
        for i in range(count):
            if input_text[i] in self.model.char_index.keys():
                char_vector[i] = self.model.char_index[input_text[i]]
        return char_vector

    def train(self, n_epochs=1):
        left = []
        right = []
        label = []
        with open('D:\mygit\\tf1.0\data\比赛\\train.txt', 'r', encoding='utf8') as f:
            for line in f.readlines():
                data = json.loads(line)
                temp = self.convert_vector(data['question'].strip())
                temp_1 = self.convert_vector(data['similar'].strip())
                label.append(int(data['label']))
                left.append(temp)
                right.append(temp_1)
        x_left = tf.keras.preprocessing.sequence.pad_sequences(left, maxlen=self.sequence)
        x_right = tf.keras.preprocessing.sequence.pad_sequences(right, maxlen=self.sequence)
        self.model.train(n_epochs, x_left, x_right, label, self.model_path)
        # self.model.save(self.model_path)

    def predict(self, input_text1, input_text2):
        input_text1 = self.convert_vector(input_text1.strip())
        input_text2 = self.convert_vector(input_text2.strip())
        x_test1 = tf.keras.preprocessing.sequence.pad_sequences([input_text1], maxlen=self.sequence)
        x_test2 = tf.keras.preprocessing.sequence.pad_sequences([input_text2], maxlen=self.sequence)
        res = self.model.predict(x_test1, x_test2, self.model_path)
        print(res)


if __name__ == '__main__':
    Train().train(2)