
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.models import load_model

from os.path import join, dirname, abspath
WORK_DIR = dirname(abspath(__file__)) # 3
print(WORK_DIR)

class cnn():
    def __init__(self):
        self.model_name = join(WORK_DIR, "model/cnn.model")
        #self.MAX_NUM_WORDS = 20000
        self.max_features = 20000
        #self.EMBEDDING_DIM = 50
        self.embedding_size = 50
        #self.MAX_SEQUENCE_LENGTH = 1000
        self.maxlen = 1000
        self.batch_size=128
        self.epochs = 3
        self.VALIDATION_SPLIT = 0.2

    def parse_embedding(self):
        BASE_DIR = 'C:/Users/zhangsen/.keras/datasets'
        GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
        print('Indexing word vectors.')
        self.embeddings_index = {}
        with open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'), encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
        # print(self.embeddings_index)
        print('Found %s word vectors.' % len(self.embeddings_index))

        print('Preparing embedding matrix.')
        # prepare embedding matrix
        self.num_words = min(self.max_features, len(self.word_index) + 1)
        self.embedding_matrix = np.zeros((self.num_words, self.embedding_size))
        for word, i in self.word_index.items():
            if i >= self.max_features:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

    def build_model(self):
        embedding_layer = Embedding(self.num_words,
                                    self.embedding_size,
                                    weights=[self.embedding_matrix],
                                    input_length=self.maxlen,
                                    trainable=False)

        print('Training model.')

        # train a 1D convnet with global maxpooling
        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(len(self.labels_index), activation='softmax')(x)

        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
    def train(self):
        self.parse_data()
        self.parse_embedding()
        self.build_model()

        self.model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_data=(self.x_val, self.y_val))
        score, acc = self.model.evaluate(self.x_val, self.y_val, batch_size=self.batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)
        self.model.save(self.model_name)

    def predict(self):
        self.parse_data()
        model = load_model(self.model_name)
        pre = model.predict(self.x_val, batch_size=self.batch_size)
        print(pre)
    def parse_data(self):
        BASE_DIR = 'C:/Users/zhangsen/.keras/datasets'
        GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
        TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')


        # first, build index mapping words in the embeddings set
        # to their embedding vector



        # second, prepare text samples and their labels
        print('Processing text dataset')

        texts = []  # list of text samples
        self.labels_index = {}  # dictionary mapping label name to numeric id
        labels = []  # list of label ids
        for name in sorted(os.listdir(TEXT_DATA_DIR)):
            path = os.path.join(TEXT_DATA_DIR, name)
            if os.path.isdir(path):
                label_id = len(self.labels_index)
                self.labels_index[name] = label_id
                for fname in sorted(os.listdir(path)):
                    if fname.isdigit():
                        fpath = os.path.join(path, fname)
                        args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                        with open(fpath, **args) as f:
                            t = f.read()
                            i = t.find('\n\n')  # skip header
                            if 0 < i:
                                t = t[i:]
                            texts.append(t)
                        labels.append(label_id)

        print('Found %s texts.' % len(texts))

        # finally, vectorize the text samples into a 2D integer tensor
        tokenizer = Tokenizer(num_words=self.max_features)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        self.word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))

        data = pad_sequences(sequences, maxlen= self.maxlen)

        labels = to_categorical(np.asarray(labels))
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)

        # split the data into a training set and a validation set
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        num_validation_samples = int(self.VALIDATION_SPLIT * data.shape[0])

        self.x_train = data[:-num_validation_samples]
        self.y_train = labels[:-num_validation_samples]
        self.x_val = data[-num_validation_samples:]
        self.y_val = labels[-num_validation_samples:]


if __name__ == '__main__':
    f_text = cnn()
    f_text.train()