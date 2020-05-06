# -*- encoding: utf-8 -*-
"""
@File    : model_esim.py
@Time    : 2020/5/6 12:03
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf
import numpy as np
import json


def unchanged_shape(input_shape):
    """Function for Lambda layer"""
    return input_shape


class ESIM(object):

    def __init__(self, embedding_size, unit, batch_size, dropout, learning_rate, vocab_file, sequence, class_size,
                 dense_hidden_sizes):
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
        self.dense_hidden_sizes = dense_hidden_sizes
        self.create_input()
        self.create_model()

    def apply_multipy(self, layers, inputs):
        agg_ = []
        for layer in layers:
            agg_.append(layer(inputs))
        return tf.keras.layers.Concatenate()(agg_)

    def attention(self, input_1, input_2):
        # 计算点积注意力
        # batch_size, time_step, hidden_size * 2
        attention = tf.keras.layers.Dot(axes=-1)([input_1, input_2])
        # axis=1按照第一个维度进行计算
        w_att_1 = tf.keras.layers.Lambda(lambda x: tf.keras.activations.softmax(x, axis=1),
                                         output_shape=unchanged_shape)(attention)
        w_att_2 = tf.keras.layers.Permute((2, 1))(
            tf.keras.layers.Lambda(lambda x: tf.keras.activations.softmax(x, axis=2),
                                   output_shape=unchanged_shape)(attention)
        )
        in1_aligned = tf.keras.layers.Dot(axes=1)([w_att_1, input_1])
        in2_aligned = tf.keras.layers.Dot(axes=1)([w_att_2, input_2])
        return in1_aligned, in2_aligned

    def sub_mult(self, input_1, input_2):
        # Layer that multiplies (element-wise) a list of inputs
        mult = tf.keras.layers.Multiply()([input_1, input_2])
        # returns a single tensor, (inputs[0] - inputs[1])
        sub = tf.keras.layers.Subtract()([input_1, input_2])
        # Layer that concatenates a list of inputs
        return tf.keras.layers.Concatenate()([sub, mult])

    def load_dict(self):
        i = 0
        with open(self.vocab_file, 'r', encoding='utf8') as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue] = i + 1
                i += 1

    def create_input(self):
        self.sentence_1 = tf.keras.layers.Input(shape=(self.sequence,), dtype='int32')
        self.sentence_2 = tf.keras.layers.Input(shape=(self.sequence,), dtype='int32')

    def create_model(self):
        # embedding layer
        embedding_layer = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size,
                                                    input_length=self.sequence)
        # batch normalization layer
        # bn_layer = tf.keras.layers.BatchNormalization()

        # embedding + batch normalization, share
        # sent1_embed = bn_layer(embedding_layer(self.sentence_1))
        # sent2_embed = bn_layer(embedding_layer(self.sentence_2))

        sent1_embed = embedding_layer(self.sentence_1)
        sent2_embed = embedding_layer(self.sentence_2)

        # bi-lstm layer, encoder
        bilstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.unit, return_sequences=True))

        sent1_encoded = bilstm_layer(sent1_embed)
        sent2_encoded = bilstm_layer(sent2_embed)

        # attention layer
        sent1_aligned, sent2_aligned = self.attention(sent1_encoded, sent2_encoded)

        # Compose op
        sent1_combined = tf.keras.layers.Concatenate()(
            [sent1_encoded, sent2_aligned, self.sub_mult(sent1_encoded, sent2_aligned), tf.multiply(sent1_encoded, sent2_aligned)])
        sent2_combined = tf.keras.layers.Concatenate()(
            [sent2_encoded, sent1_aligned, self.sub_mult(sent2_encoded, sent1_aligned), tf.multiply(sent1_encoded, sent2_aligned)])

        # bi-lstm layer
        bilstm_layer_ = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.unit, return_sequences=True))
        sent1_composed = bilstm_layer_(sent1_combined)
        sent2_composed = bilstm_layer_(sent2_combined)

        # Aggregate
        sent1_aggregated = self.apply_multipy([tf.keras.layers.GlobalAvgPool1D(), tf.keras.layers.GlobalMaxPool1D()],
            sent1_composed)
        sent2_aggregated = self.apply_multipy([tf.keras.layers.GlobalAvgPool1D(), tf.keras.layers.GlobalMaxPool1D()],
            sent2_composed)

        # Classifier
        merged = tf.keras.layers.Concatenate()([sent1_aggregated, sent2_aggregated])
        # x = tf.keras.layers.BatchNormalization()(merged)
        x = tf.keras.layers.Dense(self.dense_hidden_sizes[0], activation=tf.keras.activations.relu)(merged)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Dense(self.dense_hidden_sizes[1], activation=tf.keras.activations.relu)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        output_ = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(x)

        self.model = tf.keras.Model(inputs=[self.sentence_1, self.sentence_2], outputs=output_)
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def train(self, n_epoch, x_left, x_right, label, path):
        tensorboad = tf.keras.callbacks.TensorBoard(log_dir='log')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=path, monitor='accuracy', mode='max',
                                                        save_best_only=True)

        return self.model.fit([np.array(x_left), np.array(x_right)], np.array(label), batch_size=self.batch_size,
                              epochs=n_epoch, callbacks=[checkpoint, tensorboad])

    def predict(self, input_text1, input_text2, model_path):
        self.model.load_weights(model_path)
        return self.model.predict([input_text1, input_text2])

    def save(self, path):
        self.model.save_weights(path)
