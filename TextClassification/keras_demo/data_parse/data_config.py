#!/usr/bin/python
# -*- coding: utf-8 -*-

TOKEN_START = "<S>"
TOKEN_END = "</S>"
TOKEN_UNKNOWN = "<UNKNOWN>"
TOKEN_PAD = '<PAD>'
TOKEN_UNKNOWN_ID = 0

class DataConfig():
    def __init__(self):
        self.embedding_dir = "E:/souhu/word2vec"
        self.embedding_dim_size = 100

        self.caption_char_txt=""
        self.char2vec_model = ""


        self.caption_word_txt = "E:/souhu/News_info_train_filter_word.txt"
        self.word2vec_model = "E:/souhu/word2vec/word2vec_100.model"

        self.token_unknown=TOKEN_UNKNOWN

        self.seq_max_length = 100
        self.token_start = TOKEN_START
        self.token_end = TOKEN_END
        self.token_pad = TOKEN_PAD