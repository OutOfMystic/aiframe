import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, Input, Dropout, Embedding, Reshape, LSTM, Lambda, \
    concatenate, Conv2D, MaxPooling2D, Flatten, Softmax, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from aiframe.data.parse import divide, load_last
from aiframe.learn.learn import Experiment
from examples.rnn.input_data.input_data import load_synthetic_input_words, ru_words, get_pairs, eng_words, \
    format_input_data, human_to_bot, str_to_tokens, str_to_wordstats, str_to_one_hot, differences


class Slice(tf.keras.layers.Layer):
    def __init__(self, begin, size, **kwargs):
        super(Slice, self).__init__(**kwargs)
        self.begin = begin
        self.size = size

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'begin': self.begin,
            'size': self.size,
        })
        return config

    def call(self, inputs):
        return tf.slice(inputs, self.begin, self.size)


"""model = Sequential()
model.add(Embedding(100, 10, input_length=1000, input_shape=(1000,)))
model.add(Reshape((500, 20,)))
model.add(LSTM(128, return_sequences=True, name='LSTM1'))
model.add(LSTM(64, name='LSTM2'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()"""

"""input = Input(shape=(1000,))
x1 = Lambda(lambda x: tf.slice(x, begin=[0, 0], size=[-1, 500]))(input)
x2 = Lambda(lambda x: tf.slice(x, begin=[0, 500], size=[-1, 500]))(input)
x1 = Embedding(100, 10, input_length=500)(x1)
x2 = Embedding(100, 10, input_length=500)(x2)
x1 = Reshape((500, 10, 1))(x1)
x2 = Reshape((500, 10, 1))(x2)
#x1 = LSTM(128, return_sequences=True, name='1LSTM1')(x1)
#x2 = LSTM(128, return_sequences=True, name='2LSTM1')(x2)
#x1 = LSTM(64, name='1LSTM2')(x1)
#x2 = LSTM(64, name='2LSTM2')(x2)
x1 = Conv2D(64, (3, 3), activation='relu')(x1)
x2 = Conv2D(64, (3, 3), activation='relu')(x2)
x1 = MaxPooling2D((2, 2))(x1)
x2 = MaxPooling2D((2, 2))(x2)
x1 = Conv2D(32, (3, 3), activation='relu')(x1)
x2 = Conv2D(32, (3, 3), activation='relu')(x2)
x1 = MaxPooling2D((2, 2))(x1)
x2 = MaxPooling2D((2, 2))(x2)
x1 = Flatten()(x1)
x2 = Flatten()(x2)
x1 = Dense(128, activation='relu')(x1)
x2 = Dense(128, activation='relu')(x2)
x = concatenate([x1, x2])
x = Dense(92, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs=[input], outputs=[x])
model.summary()"""

"""input = Input(shape=(1000,))
x1 = Lambda(lambda x: tf.slice(x, begin=[0, 0], size=[-1, 500]))(input)
x2 = Lambda(lambda x: tf.slice(x, begin=[0, 500], size=[-1, 500]))(input)
x1 = Embedding(100, 10)(x1)
x2 = Embedding(100, 10)(x2)
x1 = Reshape((500, 10, 1))(x1)
x2 = Reshape((500, 10, 1))(x2)
x1 = Conv2D(64, (3, 3), activation='relu')(x1)
x2 = Conv2D(64, (3, 3), activation='relu')(x2)
x1 = MaxPooling2D((2, 2))(x1)
x2 = MaxPooling2D((2, 2))(x2)
x1 = Conv2D(32, (3, 3), activation='relu')(x1)
x2 = Conv2D(32, (3, 3), activation='relu')(x2)
x1 = MaxPooling2D((2, 2))(x1)
x2 = MaxPooling2D((2, 2))(x2)
x1 = Flatten()(x1)
x2 = Flatten()(x2)
x1 = Dense(128, activation='relu')(x1)
x2 = Dense(128, activation='relu')(x2)
x = concatenate([x1, x2])
x = Dense(92, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs=[input], outputs=[x])
model.summary()"""

# 200 - 0.5
"""input = Input(shape=(200,))
x1 = Lambda(lambda x: tf.slice(x, begin=[0, 0], size=[-1, 100]))(input)
x2 = Lambda(lambda x: tf.slice(x, begin=[0, 100], size=[-1, 100]))(input)
x1 = Embedding(100, 10)(x1)
x2 = Embedding(100, 10)(x2)
x1 = LSTM(128, return_sequences=True, name='1LSTM1')(x1)
x2 = LSTM(128, return_sequences=True, name='2LSTM1')(x2)
x1 = LSTM(64, name='1LSTM2')(x1)
x2 = LSTM(64, name='2LSTM2')(x2)
x = concatenate([x1, x2])
x = Dense(92, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs=[input], outputs=[x])
model.summary()"""

# 200 - 0.5
"""input = Input(shape=(200,))
x1 = Lambda(lambda x: tf.slice(x, begin=[0, 0], size=[-1, 100]))(input)
x2 = Lambda(lambda x: tf.slice(x, begin=[0, 100], size=[-1, 100]))(input)
x1 = Reshape((100, 1))(x1)
x2 = Reshape((100, 1))(x2)
x1 = LSTM(128, return_sequences=True, name='1LSTM1')(x1)
x2 = LSTM(128, return_sequences=True, name='2LSTM1')(x2)
x1 = LSTM(64, name='1LSTM2')(x1)
x2 = LSTM(64, name='2LSTM2')(x2)
x = concatenate([x1, x2])
x = Dense(92, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs=[input], outputs=[x])
model.summary()"""

"""input = Input(shape=(1000,))
x = Reshape((500, 2))(input)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(92, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(64, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(48, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(32, 3, activation='relu')(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs=[input], outputs=[x])
model.summary()"""

"""input_ = Input(shape=(550,))
x = Reshape((550, 1))(input_)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(64, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(48, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(32, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(24, 3, activation='relu')(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs=[input_], outputs=[x])
model.summary()"""

input_ = Input(shape=(550,))
#x = Reshape((550, 1))(input_)
x = Dense(240, activation='relu')(input_)
x = Dense(140, activation='relu')(x)
x = Dense(96, activation='relu')(x)
#x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs=[input_], outputs=[x])
experiment = Experiment(model)

# 200 - 0.7181
"""input = Input(shape=(200,))
x = Reshape((100, 2))(input)
x = LSTM(128, return_sequences=True, name='1LSTM1')(x)
x = LSTM(64, name='2LSTM2')(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs=[input], outputs=[x])
model.summary()"""

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0001))
translate_func = str_to_one_hot

rows = load_synthetic_input_words(5000, words=ru_words)
pairs = get_pairs(rows, ru_words)
rows = load_synthetic_input_words(5000, words=eng_words)
pairs += get_pairs(rows, ru_words)
matrixez, results, tokenizer = format_input_data(pairs, differences, translate_func)
input_data = divide(matrixez, results, test_prop=0.05, shuffle=True)
experiment.run_once(input_data)
experiment.save('Semantic')

rows = [
    ["Спартак Динамо", "Спартак Динамо"],
    ["Спартак Динамо", "Динамо Спартак"],
    ["Спартак Динамо", "Чемпионат Спартак Динамо"],
    ["Спартак Динамо", "Факел Москва"],
    ["Спартак Динамо", "Спартак ЦСКА"],
    ["Безумный день", "Безумный день"],
    ["Безумный день", "Война и мир"],
    ["Безумный день", "Щелкунчик"],
    ["Безумный день", "Четыре персонажа в поисках сюжета"]
]
#predict_func = lambda str1, str2: human_to_bot(str1, str2, tokenizer, translate_func)
predict_func = lambda str1, str2: differences(str1, str2, tokenizer)
to_predict = [predict_func(*row) for row in rows]

res = model.predict(np.array(to_predict))
print(res)

while True:
    input1 = input('First:\n')
    input2 = input('Second:\n')
    to_predict = predict_func(input1, input2)
    res = model.predict(np.array([to_predict]))
    print(res[0][0], res[0][0])
