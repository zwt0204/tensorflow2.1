# -*- encoding: utf-8 -*-
"""
@File    : dssm_lstm.py
@Time    : 2020/5/6 17:12
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf
import numpy as np
import json


class CosineLayer():

    def __call__(self, x1, x2):
        def _cosine(x):
            # 和x.dot(y.T)一样
            dot1 = tf.keras.backend.batch_dot(x[0], x[1], axes=1)
            dot2 = tf.keras.backend.batch_dot(x[0], x[0], axes=1)
            dot3 = tf.keras.backend.batch_dot(x[1], x[1], axes=1)
            max_ = tf.keras.backend.maximum(tf.keras.backend.sqrt(dot2 * dot3), tf.keras.backend.epsilon())
            return dot1 / max_

        output_shape = (1,)
        value = tf.keras.layers.Lambda(_cosine, output_shape=output_shape)([x1, x2])
        return value


class Evaluate(tf.keras.callbacks.Callback):
    def __init__(self):
        self.num_passed_batchs = 0
        self.warmup_epochs = 10

    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        if self.params['steps'] is None:
            self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
        else:
            self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            # 前10个epoch中，学习率线性地从零增加到0.001
            tf.keras.backend.set_value(self.model.optimizer.lr,
                        0.0001 * (self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
            self.num_passed_batchs += 1


class DSSMLSTMMODEL:

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

    # 孪生网络用对比损失
    def contrastive_loss(self, Ew, y):
        # tf.square对其中的每个元素求平方
        # 1 - Ew 余弦距离
        l_1 = y * 0.25 * tf.square(1 - Ew)
        # 相当于margin为1
        l_0 = (1 - y) * tf.square(tf.maximum(Ew, 0))
        loss = tf.reduce_sum(l_1 + l_0)
        return loss

    def create_model(self):
        left_input = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype='int32')
        right_input = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype='int32')
        shared_model = tf.keras.models.Sequential()
        shared_model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_shape=(self.max_seq_length,),
                                                trainable=True))

        peephole_lstm_cells = [tf.keras.experimental.PeepholeLSTMCell(size) for size in [self.n_hidden, self.n_hidden]]

        shared_model.add(tf.keras.layers.RNN(peephole_lstm_cells))
        shared_model.add(tf.keras.layers.Dense(2))
        a = CosineLayer()
        malstm_distance = a(shared_model(left_input), shared_model(right_input))
        self.model = tf.keras.models.Model(inputs=[left_input, right_input], outputs=[malstm_distance])

        def scheduler(epoch):
            if epoch % 2 == 0 and epoch != 0:
                lr = tf.keras.backend.get_value(self.model.optimizer.lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, lr * 0.9)
                print("lr changed to {}".format(lr * 0.9))
            return tf.keras.backend.get_value(self.model.optimizer.lr)

        # 学习率衰减
        self.reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

        if self.gpus >= 2:
            # 数据并行
            self.model = tf.keras.utils.multi_gpu_model(self.model, gpus=self.gpus)
        # dssm
        self.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
        # 孪生网络
        # self.model.compile(loss=self.contrastive_loss, optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
        from utils.radam import RAdam
        from utils.lookhead import Lookahead
        # from utils.lamb import LAMB
        # self.model.compile(loss=self.contrastive_loss, optimizer=LAMB(lr=0.0001), metrics=['accuracy'])
        self.model.summary()
        shared_model.summary()

    def train(self, n_epoch, x_left, x_right, label):
        # callbacks=[Evaluate()]
        self.model.fit([np.array(x_left), np.array(x_right)], np.array(label), batch_size=self.batch_size,
                       epochs=n_epoch)

    def predict(self, input_text1, input_text2, model_path):
        self.model.load_weights(model_path)
        return self.model.predict([input_text1, input_text2])

    def save(self, path):
        self.model.save_weights(path)

    # 有预训练向量
    def get_embedding_weight(self, weight_path, word_index):
        # embedding_weight = np.random.rand('word_num', 'embedding_dim')
        embedding_weight = np.random.uniform(-0.05, 0.05, size=['word_num', 'embedding_dim'])
        cnt = 0
        with open(weight_path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in word_index.keys() and word_index[word] < 'word_num':
                    weight = np.asarray(values[1:], dtype='float32')
                    embedding_weight[word_index[word] + 3] = weight
                    cnt += 1
        print('word num: {}, matched num: {}'.format(len(word_index), cnt))
        return embedding_weight