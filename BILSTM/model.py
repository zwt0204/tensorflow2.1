# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2020/4/29 14:45
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf
import json
import numpy as np


class bilstm():

    def __init__(self, embedding_size, unit, batch_size, dropout, learning_rate, vocab_file, sequence, class_size):
        self.embedding_size = embedding_size
        self.unit = unit
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.vocab_file = vocab_file
        self.char_index = {' ': 0}
        self.sequence = sequence
        self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.vocab_size = len(self.char_index) + 1
        self.class_size = class_size
        self.create_input()
        self.create_model()

    def load_dict(self):
        i = 0
        with open(self.vocab_file, 'r', encoding='utf8') as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue] = i + 1
                i += 1

    def create_input(self):
        self.input = tf.keras.layers.Input(shape=(self.sequence,), dtype='int32')

    def create_model(self):

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.sequence))
        self.model.add(tf.keras.layers.Dropout(self.dropout))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.unit)))
        self.model.add(tf.keras.layers.Dropout(self.dropout))
        self.model.add(tf.keras.layers.Dense(self.class_size, activation='softmax'))

        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        # 如果loss在100个epoch后没有提升，学习率减半。
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=100)
        # 当loss在200个epoch后没有提升，则提前终止训练。
        stop_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=200)
        self.callbacks_list = [lr_callback, stop_callback]

    def train(self, n_epoch, x, y):
        self.model.fit(np.array(x), np.array(y), batch_size=self.batch_size, epochs=n_epoch,
                       callbacks=self.callbacks_list)

    def predict(self, x, model_path):
        self.model.load_weights(model_path)
        return self.model.predict(x)

    def save(self, path):
        self.model.save_weights(path)
        self.model.save('D:\mygit\\tensorflow2.1\model\\tf_model', save_format='tf')
