import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow import keras

import os
from PIL import Image
import numpy as np
from tensorflow.python.keras.backend import reshape

all_files = []
for i in range(0, 10):
    path_dir = './images/training/{0}'.format(i)
    file_list = os.listdir(path_dir)
    file_list.sort()
    all_files.append(file_list)

x_train_datas = []
y_train_datas = []

for num in range(0, 10):
    for numbers in all_files[num]:
        img_path = './images/training/{0}/{1}'.format(num, numbers)
        print("lord: " + img_path)
        img = Image.open(img_path)

        imgarr = np.array(img) / 255.0
        x_train_datas.append(np.reshape(imgarr, newshape=(784, 1)))  # 입력값
        y_tab = np.zeros(shape=(10))
        y_tab[num] = 1
        y_train_datas.append(y_tab)  # 정답

print(len(x_train_datas))
print(len(y_train_datas))

# -===================
eval_files = []
for i in range(0, 10):
    path_dir = './images/testing/{0}'.format(i)
    file_list = os.listdir(path_dir)
    file_list.sort()
    eval_files.append(file_list)

x_test_datas = []
y_test_datas = []

for num in range(0, 10):
    for numbers in eval_files[num]:
        img_path = './images/testing/{0}/{1}'.format(num, numbers)
        print("lord: " + img_path)
        img = Image.open(img_path)
        imgarr = np.array(img) / 255.0
        x_test_datas.append(np.reshape(imgarr, newshape=(784, 1)))  # 입력값
        y_tab = np.zeros(shape=(10))
        y_tab[num] = 1
        y_test_datas.append(y_tab)  # 정답

print(len(x_test_datas))
print(len(y_test_datas))

x_train_datas = np.reshape(x_train_datas, newshape=(-1, 784))
y_train_datas = np.reshape(y_train_datas, newshape=(-1, 10))

input = tf.keras.Input(shape=(784), name="Input")
hidden = tf.keras.layers.Dense(512, activation="relu", name="Hidden1")(input)
output = tf.keras.layers.Dense(10, activation="softmax", name="Output")(hidden)

model = tf.keras.Model(inputs=[input], outputs=[output])
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])
model.summary()
model.fit(x_train_datas, y_train_datas, epochs=5, shuffle=True)

x_test_datas = np.reshape(x_test_datas, newshape=(-1,784))
y_test_datas = np.reshape(y_test_datas, newshape=(-1, 10))
test_loss, test_acc = model.evaluate(x_test_datas, y_test_datas)
print('테스트 정확도:', test_acc)



result = model.predict(x_test_datas)
diff = []
predictnum = []
for i in range(len(result)):
    if np.argmax(result[i]) !=np.argmax(y_test_datas[i]):
        img = np.reshape(x_test_datas[i], newshape=(28,28))*255
        img = Image.fromarray(img)
        diff.append(img)
        predictnum.append(np.argmax(result[i]))
for i in range(5):
    pickidx = random.choice(range(0,len(diff)))
    print(predictnum[pickidx])
    diff[pickidx].show()