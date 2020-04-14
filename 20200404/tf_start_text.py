from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re, string

train_data_path = '../data/imdb_datasets/xx_train_imdb'
test_data_path = '../data/imdb_datasets/xx_test_imdb'

MAX_WORDS = 10000
MAX_LEN = 200
BATCH_SIZE = 20


def split_line(line):
    arr = tf.strings.split(line, '\t')
    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]), tf.int32), axis=0)
    text = tf.expand_dims(arr[1], axis=0)
    return (text, label)


ds_train_raw = tf.data.TextLineDataset(filenames=[train_data_path]) \
    .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

ds_test_raw = tf.data.TextLineDataset(filenames=[test_data_path]) \
    .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)


def clean_text(text):
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    cleaned_punctuation = tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')
    return cleaned_punctuation


vectorize_layer = keras.layers.experimental.preprocessing.TextVectorization(
    standardize=clean_text,
    split='whitespace',
    max_tokens=MAX_WORDS - 1,
    output_mode='int',
    output_sequence_length=MAX_LEN
)

ds_text = ds_train_raw.map(lambda text, label: text)
vectorize_layer.adapt(ds_text)
# print(vectorize_layer.get_vocabulary()[0:100])

# 单词编码
ds_train = ds_train_raw.map(lambda text, label: (vectorize_layer(text), label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test_raw.map(lambda text, label: (vectorize_layer(text), label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)

# 使用model基类构建自定义模型
keras.backend.clear_session()


class CnnModel(keras.models.Model):

    def __init__(self):
        super(CnnModel, self).__init__()

    def build(self, input_shape):
        self.embedding = keras.layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN)
        self.conv1 = keras.layers.Conv1D(16, kernel_size=5, name='conv_1', activation='relu')
        self.pool = keras.layers.MaxPool1D()
        self.conv2 = keras.layers.Conv1D(128, kernel_size=2, name='conv_2', activation='relu')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(1, activation='sigmoid')
        super(CnnModel, self).build(input_shape)

    def call(self, x):
        x = self.embedding(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return (x)


model = CnnModel()
model.build(input_shape=(None, MAX_LEN))
model.summary()


# 打印时间分割线
@tf.function
def printbar():
    ts = tf.timestamp()
    today_ts = ts % (24 * 60 * 60)

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return (tf.strings.format("0{}", m))
        else:
            return (tf.strings.format("{}", m))

    timestring = tf.strings.join([timeformat(hour), timeformat(minite),
                                  timeformat(second)], separator=":")
    tf.print("==========" * 8, end="")
    tf.print(timestring)


optimizer = keras.optimizers.Nadam()
loss_func = keras.losses.BinaryCrossentropy()

train_loss = keras.metrics.Mean(name='train_loss')
train_metric = keras.metrics.BinaryAccuracy(name='train_accuracy')

valid_loss = keras.metrics.Mean(name='valid_loss')
valid_metric = keras.metrics.BinaryAccuracy(name='valid_accuracy')


@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)


@tf.function
def valid_step(model, features, labels):
    predictions = model(features, training=False)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)


def train_model(model, ds_train, ds_valid, epochs):
    for epoch in tf.range(1, epochs + 1):

        for features, labels in ds_train:
            train_step(model, features, labels)

        for features, labels in ds_valid:
            valid_step(model, features, labels)

        # 此处logs模板需要根据metric具体情况修改
        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'

        if epoch % 1 == 0:
            printbar()
            tf.print(tf.strings.format(logs,
                                       (epoch, train_loss.result(), train_metric.result(), valid_loss.result(),
                                        valid_metric.result())))
            tf.print("")

        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()


train_model(model, ds_train, ds_test, epochs=6)