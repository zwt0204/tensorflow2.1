# 为什么选择tf：在工业界最重要的是模型落地，tf支持并且已经被众多的公司所应用于线上部署
# torch：在模型的实现或者新模型的尝试方面有优势，目前学术界的主流
# keras在2.3.0后不再更新，嵌入到tf.keras中
# 泰坦尼克号生存者预测问题
"""
数据说明：
survied：是否存活，0，1
pclass：持有票的类型，1，2，3
name：乘客姓名，舍去
sex：乘客性别，转换为one-hot
age：乘客年龄，数值做辅助特征
sibsp：乘客兄弟姐妹或者配偶的个数，数值特征
parch：乘客父母或者孩子的数目
ticket：票号，
fare：票价
cabin：乘客所在船舱
embarked：乘客登船港口
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras


dftrain_raw = pd.read_csv('../data/titanic/train.csv')
dftest_raw = pd.read_csv('../data/titanic/test.csv')

# print(dftrain_raw.head(10))
# ax = dftrain_raw['Survived'].value_counts().plot(kind='bar',
#                                                 figsize=(12, 8), fontsize=15,rot=0)
# ax.set_ylabel('Counts', fontsize=15)
# ax.set_xlabel('Survived', fontsize=15)
# plt.show()

# age_x = dftrain_raw['Age'].value_counts().plot(kind='hist', bins=20, color='purple',
#                                                figsize=(12, 8), fontsize=15)
# age_x.set_ylabel('Frequency', fontsize=15)
# age_x.set_xlabel('Age', fontsize=15)
# plt.show()

# 年龄和标签的相关性
# ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind='density',
#                                                     figsize=(12, 8), fontsize=15)
# dftrain_raw.query('Survived == 1')['Age'].plot(kind='density',
#                                                     figsize=(12, 8), fontsize=15)
# ax.legend(['Survived == 0', 'Survived == 1'], fontsize=12)
# ax.set_ylabel('Density', fontsize=15)
# ax.set_xlabel('Age', fontsize=15)
# plt.show()


def plot_metric(history, metric):
    train_metric = history.history[metric]
    val_metric = history.history['val_' + metric]
    epochs = range(1, len(train_metric)+1)
    plt.plot(epochs, train_metric, 'bo--')
    plt.plot(epochs, val_metric, 'ro--')
    plt.title('Training and validation' + metric)
    plt.xlabel('Epoches')
    plt.ylabel(metric)
    plt.legend(['train_'+metric, 'val_'+metric])
    plt.show()


def process_data(dfdata):

    dfresult = pd.DataFrame()

    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' + str(x) for x in dfPclass.columns]
    # axis=1表示按行拼接
    dfresult = pd.concat([dfresult, dfPclass], axis=1)

    dfsex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfsex], axis=1)

    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return (dfresult)


# [712, 15]
x_train = process_data(dftrain_raw)
y_train = dftrain_raw['Survived'].values

# [179, 15]
x_test = process_data(dftest_raw)
# print(dftest_raw.columns)
# y_test = dftest_raw['Survived'].values

# 定义模型
"""
1. 使用sequential按层顺序构建模型
2. 使用函数式API构建任意结构模型
3. 继承Mode基类构建自定义模型
"""

model = keras.models.Sequential()
model.add(keras.layers.Dense(20, activation='relu', input_shape=(15, )))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# 训练
"""
1. 内置fit
2. 内置train_on_batch
3. 自定义循环
"""
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
history = model.fit(x_train, y_train, batch_size=64,
                    epochs=50,
                    validation_split=0.2)

# plot_metric(history, 'loss')

# plot_metric(history, 'AUC')

# 预测概率
model.predict(x_test)

# 预测类别
model.predict_classes(x_test)

# 保存模型结构和权重
model.save('')

del model  # 删除现有模型

# 加载模型
model = model.load_model()

# 保存模型结构
json_str = model.to_json()

# 恢复模型结构
model_json = model.model_from_json(json_str)

# 保存模型权重
model.save_weights()

# 恢复模型结构
model_json = model.model_from_json(json_str)
model_json.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['AUC']
)
# 加载权重
model_json.load_weights()


# tf原生方式保存
# 保存权重
model.save_weights('', save_format='tf')

# 保存模型结构和模型参数到文件，该方式保存的模型具有跨平台性易于部署
model.save('', save_format='tf')

model = keras.models.load_model()



