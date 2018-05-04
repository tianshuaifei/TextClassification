
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb
from keras.models import load_model

from os.path import join, dirname, abspath
WORK_DIR = dirname(abspath(__file__)) # 3
print(WORK_DIR)

class cnn_lstm():
    def __init__(self):
        self.model_name = join(WORK_DIR, "model/cnn_lstm.model")

        self.max_features = 20000
        self.maxlen = 100
        self.embedding_size = 128

        # Convolution
        self.kernel_size = 5
        self.filters = 64
        self.pool_size = 4

        # LSTM
        self.lstm_output_size = 70
        # Training
        self.batch_size = 30
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

    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.max_features, self.embedding_size, input_length=self.maxlen))
        self.model.add(Dropout(0.25))
        self.model.add(Conv1D(self.filters,
                         self.kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        self.model.add(MaxPooling1D(pool_size=self.pool_size))
        self.model.add(LSTM(self.lstm_output_size))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])




    def train(self):
        self.parse_data()
        self.build_model()

        print('Train...')
        self.model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_data=(self.x_test, self.y_test))
        score, acc = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)
        self.model.save(self.model_name)

    def predict(self):
        self.parse_data()
        model = load_model(self.model_name)
        classes = model.predict(self.x_test, batch_size=self.batch_size)
        print(len(classes))
        print(self.y_test[:5])
        print(classes[:5])


if __name__ == '__main__':
    f_text=cnn_lstm()
    f_text.train()
    f_text.predict()