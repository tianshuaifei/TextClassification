
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dense, Activation
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb
#from a_keras_demo.classifyModel.data_parse.data_load import load_train_data
import keras
from os.path import join, dirname, abspath
WORK_DIR = dirname(abspath(__file__)) # 3
print(WORK_DIR)
def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


class fasttext():
    def __init__(self):
        self.model_name=join(WORK_DIR,"model/fasttext.model")
        self.ngram_range = 1
        self.max_features = 20000
        self.maxlen = 400
        self.batch_size = 32
        self.embedding_dims = 50
        self.epochs = 1
        self.num_classes=2
        self.load_data=imdb.load_data(num_words=self.max_features)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data
        #self.x_train, self.y_train=load_train_data("E:/souhu/News_info_train_filter_word.txt")
        self.x_test=self.x_train
        self.y_test=self.y_train
    def parse_data(self):
        if self.ngram_range>1:
            self.n_gram()
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=self.maxlen)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=self.maxlen)
        print('x_train shape:', self.x_train.shape)
        print('x_test shape:', self.x_test.shape)
        print(self.y_train[:5])
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        print(self.y_train[:5])


    def train(self):

        self.parse_data()

        print('Build model...')
        model = Sequential()
        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(self.max_features,
                            self.embedding_dims,
                            input_length=self.maxlen))

        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        model.add(GlobalAveragePooling1D())

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(2, activation='softmax'))
        #model.add(Activation('softmax'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_data=(self.x_test, self.y_test))

        score, acc = model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)
        model.save(self.model_name)


    def predict(self):
        self.parse_data()
        model = load_model(self.model_name)
        pre=model.predict(self.x_test,batch_size=self.batch_size)
        print(len(pre))
        print(self.y_test[:5])
        print(pre[:5])
        classes=model.predict_classes(self.x_test)
        print(classes[:5])


    def n_gram(self):
        print('Adding {}-gram features'.format(self.ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in self.x_train:
            for i in range(2, self.ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = self.max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        max_features = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        self.x_train = add_ngram(self.x_train, token_indice, self.ngram_range)
        self.x_test = add_ngram(self.x_test, token_indice, self.ngram_range)
        print('Average train sequence length: {}'.format(np.mean(list(map(len, self.x_train)), dtype=int)))
        print('Average test sequence length: {}'.format(np.mean(list(map(len, self.x_test)), dtype=int)))

if __name__ == '__main__':
    f_text=fasttext()
    f_text.train()
    f_text.predict()

