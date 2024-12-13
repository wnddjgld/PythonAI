from tabnanny import verbose

import tensorflow as tf
from keras.backend import binary_crossentropy
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

def multi_hot_sequences(sequences, dimension):
    # 0으로 채워진 (len(sequences), dimension) 크기의 행렬을 만듭니다
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0 # results[i]의 특정 인덱스만 1로 설정합니다
    return results

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                    '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])
    plt.show()

NUM_WORDS = 1000
(train_data, train_labels), (test_data, test_labels) =(
    keras.datasets.imdb.load_data(num_words=NUM_WORDS))
train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

baseline_model = keras.Sequential([
    # `.summary` 메서드 때문에 `input_shape`가 필요합니다
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

baseline_model.compile(optimizer='adam',loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])
baseline_model.summary()

checkpoint_path = "baseline_model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

loss = baseline_model.evaluate(test_data, test_labels, verbose=2)
print("훈련되지 않은 모델의 정확도: {:5.2f}%".format(100*loss[1]))

baseline_model.load_weights(checkpoint_path)
loss = baseline_model.evaluate(test_data, test_labels, verbose=2)
print("훈련된 모델의 정확도: {:5.2f}%".format(100*loss[1]))