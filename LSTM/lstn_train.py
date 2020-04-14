# -*- encoding: utf-8 -*-
"""
@File    : lstn_train.py
@Time    : 2019/11/19 10:28
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from LSTM.lstm_model import RNN
from tensorflow import keras
import tensorflow as tf
import os
import numpy as np
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class model_train():

    def __init__(self):
        self.batch_size = 64  # 批量大小
        self.units = 64  # RNN状态向量长度f
        self.epochs = 10  # 训练epochs
        self.lr = 0.0001
        # 词典
        self.vocab_file = "../data/dictionary.json"
        self.model = RNN(self.units, self.vocab_file)

    def read_sample_file(self, datafile):
        row_mapper = {}
        with open(datafile, "r+", encoding="utf-8") as reader:
            for line in reader:
                record = json.loads(line.strip().lower())
                classid = int(record['label'])
                raw_text = record['question']
                row_mapper[raw_text] = classid
        return row_mapper

    def shuffle(self, *arrs):
        """
        打乱list的顺序，可以同时传入多个
        p, h, label = self.shuffle(p, h, label)
        :param arrs:
        :return:
        """
        arrs = list(arrs)
        for i, arr in enumerate(arrs):
            assert len(arrs[0]) == len(arrs[i])
            arrs[i] = np.array(arr)
        p = np.random.permutation(len(arrs[0]))
        return tuple(arr[p] for arr in arrs)

    def load_samples_test(self, datafiles):
        xrows = []
        yrows = []
        text_mapper = self.read_sample_file(datafiles)
        for k, v in text_mapper.items():
            temp = self.convert_vector(k, self.model.sentence_length)
            xrows.append(temp)
            yrows.append(v)
        xrows, yrows = self.shuffle(xrows, yrows)
        return xrows, yrows

    def convert_vector(self, input_text, limit):
        char_vector = np.zeros((self.model.sentence_length), dtype=np.float32)
        count = len(input_text.strip().lower())
        if count > limit:
            count = limit
        for i in range(count):
            if input_text[i] in self.model.char_index.keys():
                char_vector[i] = self.model.char_index[input_text[i]]

        return char_vector

    def data_process(self):
        # x_train, y_train, x_test, y_test = get_data(self.model.vocab_size)
        # 加载数据集
        x_train, y_train = self.load_samples_test(datafiles='data.txt')
        # word_index, reverse_word_index = word_index_word()
        # 截断和填充句子，使得等长，此处长句子保留句子后面的部分，短句子在前面填充
        x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=self.model.sentence_length)
        # x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=self.model.sentence_length)
        # 构建数据集，打散，批量，并丢掉最后一个不够batch_size的batch
        db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        db_train = db_train.shuffle(1000).batch(self.batch_size, drop_remainder=True)
        # db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        # db_test = db_test.batch(self.batch_size, drop_remainder=True)
        # return db_train, db_test
        return db_train

    def train(self):
        # db_train, db_test = self.data_process()
        db_train = self.data_process()
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        # self.model.fit(db_train, epochs=self.epochs, validation_data=db_test)
        self.model.fit(db_train, epochs=self.epochs, validation_data=None)
        self.model.save_weights('lstm.ckpt')
        print('save weights')

    def predict(self, input_text):
        # Make a prediction on all the batch
        inputs = self.convert_vector(input_text, self.model.sentence_length)
        print(inputs.shape)
        inputs = keras.layers.Reshape(inputs, [-1, 70])
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        self.model.load_weights('lstm.ckpt')
        predictions = self.model(inputs)
        # a = tf.argmax(predictions, axis=1)
        print(predictions)
        return predictions


if __name__ == '__main__':
    model = model_train()
    # model.train()
    model.predict('abc')
