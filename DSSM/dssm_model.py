# -*- encoding: utf-8 -*-
"""
@File    : dssm_model.py
@Time    : 2020/4/15 9:34
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from tensorflow import keras
import json
import numpy as np


class CosineLayer():

    def __call__(self, x1, x2):
        def _cosine(x):
            # 和x.dot(y.T)一样
            dot1 = keras.backend.batch_dot(x[0], x[1], axes=1)
            dot2 = keras.backend.batch_dot(x[0], x[0], axes=1)
            dot3 = keras.backend.batch_dot(x[1], x[1], axes=1)
            # keras.backend.epsilon() == 1e-07
            max_ = keras.backend.maximum(keras.backend.sqrt(dot2 * dot3), keras.backend.epsilon())
            return dot1 / max_

        output_shape = (1,)
        value = keras.layers.Lambda(_cosine, output_shape=output_shape)([x1, x2])
        return value


class DssmModel:

    def __init__(self):
        self.char_index = {' ': 0}
        self.embedding_dim = 300
        self.max_seq_length = 50
        self.gpus = 1
        self.batch_size = 128 * self.gpus
        self.n_hidden = 128
        self.vocab_file = "D:\mygit\\tf1.0\data\肯定否定\dictionary.json"
        self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.vocab_size = len(self.char_index) + 1
        self.create_model()

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 1
                i += 1

    def create_model(self):
        left_input = keras.layers.Input(shape=(self.max_seq_length,), dtype='int32')
        right_input = keras.layers.Input(shape=(self.max_seq_length,), dtype='int32')
        shared_model = keras.models.Sequential()
        shared_model.add(keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_shape=(self.max_seq_length,),
                                                trainable=True))
        shared_model.add(keras.layers.Bidirectional(keras.layers.LSTM(self.n_hidden)))
        shared_model.add(keras.layers.Dense(2))
        a = CosineLayer()
        malstm_distance = a(shared_model(left_input), shared_model(right_input))
        self.model = keras.models.Model(inputs=[left_input, right_input], outputs=[malstm_distance])

        def scheduler(epoch):
            if epoch % 2 == 0 and epoch != 0:
                lr = keras.backend.get_value(self.model.optimizer.lr)
                keras.backend.set_value(self.model.optimizer.lr, lr * 0.9)
                print("lr changed to {}".format(lr * 0.9))
            return keras.backend.get_value(self.model.optimizer.lr)

        self.reduce_lr = keras.callbacks.LearningRateScheduler(scheduler)

        if self.gpus >= 2:
            self.model = keras.utils.multi_gpu_model(self.model, gpus=self.gpus)
        self.model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        self.model.summary()
        shared_model.summary()

    def train(self, n_epoch, x_left, x_right, label):
        self.model.fit([np.array(x_left), np.array(x_right)], np.array(label), batch_size=self.batch_size,
                       epochs=n_epoch)

    def predict(self, input_text1, input_text2, model_path):
        self.model.load_weights(model_path)
        return self.model.predict([input_text1, input_text2])

    def save(self, path):
        self.model.save_weights(path)
