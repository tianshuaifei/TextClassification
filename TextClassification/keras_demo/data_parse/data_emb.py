# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

import numpy as np
from gensim.models.word2vec import LineSentence, Word2Vec
from TextClassification.keras_demo.data_parse.data_config import DataConfig

class DataEmbedding(object):
    def __init__(self):
        self.data_config = DataConfig()
        self.load_embeddings()

    def build_embeddings(self):
        sentences = LineSentence(self.data_config.caption_word_txt)
        dims = [50, 100, 200]
        for dim_size in dims:
            model_file_name = "word2vec_" + str(dim_size) + ".model"
            model_file = os.path.join(self.data_config.embedding_dir, model_file_name)
            print("begin token2vec model {} generation".format(model_file))
            model = Word2Vec(sentences, size=dim_size, window=5, min_count=1, workers=4)
            model.save(model_file)
            print("Generated token2vec model to {}".format(model_file))
        pass

    def load_embeddings(self):
        """
        load char2vec or word2vec model for token embeddings
        :return:
        """
        token2vec = Word2Vec.load(self.data_config.word2vec_model)
        self.vocab = dict()
        for token, item in token2vec.wv.vocab.items():
            self.vocab[token] = {'count': item.count,
                                 'index': item.index}
        self.vocab[self.data_config.token_unknown] = {'count': 0,
                                                      'index': len(token2vec.wv.vocab)}

        self.token2index = dict()
        self.index2token = dict()
        self.token_embedding_matrix = np.zeros(
            [len(self.vocab), self.data_config.embedding_dim_size])

        for idx, token in enumerate(token2vec.wv.index2word):
            token_embedding = token2vec.wv[token]
            self.index2token[idx] = token
            self.token2index[token] = idx
            self.token_embedding_matrix[idx] = token_embedding
        idx += 1
        # for unknown token
        self.token_embedding_matrix[idx] = np.zeros(shape=[self.data_config.embedding_dim_size])
        self.token2index[self.data_config.token_unknown] = idx
        self.index2token[idx] = self.data_config.token_unknown

        self.vocab_size = len(self.vocab)
        self.embedding_size = self.data_config.embedding_dim_size
        pass



if __name__ == '__main__':
    data_embeddings = DataEmbedding()
    data_embeddings.build_embeddings()