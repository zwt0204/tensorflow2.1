# -*- encoding: utf-8 -*-
"""
@File    : dssm_train.py
@Time    : 2020/4/15 9:48
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from DSSM.dssm_model import DssmModel
import numpy as np
import json
import tensorflow as tf
# print(tf.__version__)


class Train:

    def __init__(self):
        self.model = DssmModel()
        self.model_path = '../model/DSSM.h5'

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
        x_left = tf.keras.preprocessing.sequence.pad_sequences(left, maxlen=self.model.max_seq_length)
        x_right = tf.keras.preprocessing.sequence.pad_sequences(right, maxlen=self.model.max_seq_length)
        self.model.train(n_epochs, x_left, x_right, label)
        self.model.save(self.model_path)

    def predict(self, input_text1, input_text2):
        input_text1 = self.convert_vector(input_text1.strip())
        input_text2 = self.convert_vector(input_text2.strip())
        x_test1 = tf.keras.preprocessing.sequence.pad_sequences([input_text1], maxlen=self.model.max_seq_length)
        x_test2 = tf.keras.preprocessing.sequence.pad_sequences([input_text2], maxlen=self.model.max_seq_length)
        res = self.model.predict(x_test1, x_test2, self.model_path)
        print(res)


if __name__ == '__main__':
    Train().train(20)
    # Train().predict('怀孕感冒了怎么办', '怀孕感冒需要做什么')