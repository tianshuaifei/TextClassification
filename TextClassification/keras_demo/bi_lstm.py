
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras.models import load_model
from a_keras_demo.classifyModel.dataConfig import config_bi_lstm
from os.path import join, dirname, abspath
WORK_DIR = dirname(abspath(__file__)) # 3
print(WORK_DIR)


class bi_lstm():
    def __init__(self,config):
        self.model_name = config.model_name

        self.max_features = 20000

        # cut texts after this number of words
        # (among top max_features most common words)
        self.lstm_output_size=64
        self.embedding_size = 128
        self.maxlen = 100
        self.batch_size = 32
        self.epochs = 1

    def parse_data(self):
        print('Loading data...')
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(num_words=self.max_features)
        print(len(self.x_train), 'train sequences')
        print(len(self.x_test), 'test sequences')

        print('Pad sequences (samples x time)')
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=self.maxlen)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=self.maxlen)
        print('x_train shape:', self.x_train.shape)
        print('x_test shape:', self.x_test.shape)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

    def build_model(self):

        self.model = Sequential()
        self.model.add(Embedding(self.max_features, self.embedding_size, input_length=self.maxlen))
        self.model.add(Bidirectional(LSTM(self.lstm_output_size)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        self.model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    def train(self):
        self.parse_data()
        self.build_model()
        print('Train...')
        self.model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_data=[self.x_test, self.y_test])

        score, acc = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)
        self.model.save(self.model_name)
    def predict(self):
        self.parse_data()
        model = load_model(self.model_name)
        pre = model.predict(self.x_test, batch_size=self.batch_size)
        print(pre)

if __name__ == '__main__':
    config=config_bi_lstm()
    f_text=bi_lstm(config=config)
    f_text.train()