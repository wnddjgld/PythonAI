import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
word_index = {k:(v+3)for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reversed_word_index.get(i,'?') for i in text])
print(decode_review(train_data[0]))
print(train_labels[0])

# 256개의 고정된 길이를 갖게끔 전처리
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
value=word_index["<PAD>"], padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
value=word_index["<PAD>"], padding='post', maxlen=256)
print(len(train_data[0]),len(test_data[1]))
# print(train_data[0])

vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])

# 상관 없음
# 학습 데이터
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

# 긍정과 부정
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
#train test 밸런스 되어져 있음

#40번의 에콕별로 히스토리에 저장됨
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=100,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)
# 학습에 대한 결과를 출력 하여 확인
results = model.evaluate(test_data, test_labels, verbose=2)
print(results)

# 출력 결과 분석
# loss: 오차, 네트워크의 출력과 정답과 값의 차이
# accuracy: 모델이 예측한 값 중에서 실제로 맞춘 비율 1에 가까울수록 정확도가 높은거임
# val_loss: validation_data를 넣었을 때의 loss
# val_accuracy: 훈련되지 않은 데이터를 넣었을 때의 모델이 예측한 값 중에서 실제로 맞춘 비율

# 점은 traing된 로스, 선은 벨리데이션 로스
# 차이가 벌어지는건 정답과 멀어지고 있다고 보면됨
# 점은 traing된 accuracy, 선은 벨리데이션 accuracy

history_dict = history.history
print(history_dict.keys())

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

# "bo"는 "파란색 점"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b는 "파란 실선"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf() # 그림을 초기화
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
