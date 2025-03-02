import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

NUM_WORDS = 1000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))

    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0
    return results

#멀티핫인코딩
train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

# plt.plot(train_data[0])
# plt.show()

baseline_model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])
baseline_model.summary()

smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
smaller_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])
smaller_model.summary()

bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy','binary_crossentropy'])
bigger_model.summary()

l2_model = keras.models.Sequential([
    keras.layers.Dense(16,activation='relu', input_shape=(NUM_WORDS,), kernel_regularizer=keras.regularizers.l2(0.001)),#억제
    keras.layers.Dense(16,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001),),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')])

l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])
l2_model.summary()

dpt_model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])
dpt_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])
dpt_model.summary()

l3_model = keras.models.Sequential([
    keras.layers.Dense(16,activation='relu', input_shape=(NUM_WORDS,), kernel_regularizer=keras.regularizers.l2(0.001)),#억제
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001),),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')])

l3_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])

l3_model.summary()

baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=200,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)
smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs=200,
                                    batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)
bigger_history = bigger_model.fit(train_data,
                                  train_labels,
                                  epochs=200,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)
l2_history = l2_model.fit(train_data,
                          train_labels,
                          epochs=200,
                          batch_size=512,
                          validation_data=(test_data, test_labels),
                          verbose=2)

dpt_history = dpt_model.fit(train_data,
                          train_labels,
                          epochs=200,
                          batch_size=512,
                          validation_data=(test_data, test_labels),
                          verbose=2)
l3_history = l3_model.fit(train_data,
                                      train_labels,
                                      epochs=200,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)

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


plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history),
             ('l2_model', l2_history),
              ('dpt_model', dpt_history),
              ('l3_model', l3_history)])