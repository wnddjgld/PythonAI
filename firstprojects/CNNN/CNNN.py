import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 픽셀 값을 0~1 사이로 정규화
train_images, test_images = train_images / 255.0, test_images / 255.0

model = keras. Sequential()
model.add(keras.layers.Conv2D(32,(3,3),activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=30, validation_data=(test_images, test_labels))

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('LOSS_GRAPH')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

class_names = ['0','1','2','3','4','5','6','7','8','9']

predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                            100*np.max(predictions_array),
                            class_names[true_label]),
                            color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    this_plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')

# 예측이 실패한 이미지를 필터링
misclassified_indexes = []
for i in range(len(test_labels)):
    if np.argmax(predictions[i]) != test_labels[i]:
        misclassified_indexes.append(i)

# 예측 실패한 이미지를 행렬 형식으로 시각화
num_rows = 5  # 출력할 행 수
num_cols = 3  # 출력할 열 수
num_images = min(num_rows * num_cols, len(misclassified_indexes))  # 표시할 이미지 수 제한

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))  # 그래프 크기 설정
for i in range(num_images):
    mis_idx = misclassified_indexes[i]  # 예측 실패한 이미지의 인덱스
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(mis_idx, predictions[mis_idx], test_labels, test_images)  # 이미지 출력
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(mis_idx, predictions[mis_idx], test_labels)  # 확률 막대 그래프 출력

plt.tight_layout()  # 시각화 간격 조정
plt.show()