from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np


train_dir = '../data/cifar2_datasets/train'
test_dir = '../data/cifar2_datasets/test'

# 对训练数据设置数据增强
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # 图片随机反转的角度
    width_shift_range=0.2,  # 图像水平或者垂直方向上的平移范围
    height_shift_range=0.2,
    shear_range=0.2,  # 随机切换的角度
    zoom_range=0.2,  # 随机缩放的范围
    horizontal_flip=True,  # 随机将图像水平反转
    fill_mode='nearest'  # 用于填充新创建像素的方法，这些新像素可能来自于旋转活宽度高度平移
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

fnames = [os.path.join('/Users/zhangweitao/Downloads/zwt/tf2.0_practice/data/cifar2_datasets/train/0_airplane', fname) for
          fname in os.listdir('/Users/zhangweitao/Downloads/zwt/tf2.0_practice/data/cifar2_datasets/train/0_airplane')]

# 载入第3张图片
img_path = fnames[3]
# print(img_path)
img = keras.preprocessing.image.load_img(img_path, target_size=(32, 32))
x = keras.preprocessing.image.img_to_array(img)
plt.figure(1, figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(keras.preprocessing.image.array_to_img(x))
plt.title('original image')

# 数据增强后的图像
x = x.reshape((1, ) + x.shape)
i = 0
for batch in train_datagen.flow(x, batch_size=1):
    plt.subplot(2, 2, i+2)
    plt.imshow(keras.preprocessing.image.array_to_img(batch[0]))
    plt.title('after augumentation %d' % (i+1))
    i = i + 1
    if i % 3 == 0:
        break
# plt.show()


train_generator = train_datagen.flow_from_directory(
    '/Users/zhangweitao/Downloads/zwt/tf2.0_practice/data/cifar2_datasets/train',
    target_size=(32, 32),
    batch_size=32,
    shuffle=True,
    class_mode='binary'
)

test_generator = train_datagen.flow_from_directory(
    '/Users/zhangweitao/Downloads/zwt/tf2.0_practice/data/cifar2_datasets/test',
    target_size=(32, 32),
    batch_size=32,
    shuffle=True,
    class_mode='binary'
)

# print(train_generator)
# print(test_generator)

keras.backend.clear_session()
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['acc'])

model.summary()


train_eps_per_epoch = np.ceil(10000/32)
test_steps_per_epoch = np.ceil(1000/32)

# 使用内存友好的fit_generator方法进行训练
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_eps_per_epoch,
    epochs=5,
    validation_data=test_generator,
    validation_steps=test_steps_per_epoch,
    workers=1,
    use_multiprocessing=False
)