#!/usr/bin/python
# -*- coding: utf-8 -*-
from TextClassification.keras_demo.data_parse.data_emb import DataEmbedding
def file_content(filename):
    try:
        d_file = open(filename, "r",encoding="utf-8")
        lines = d_file.readlines()
    except UnicodeDecodeError as e:
        d_file = open(filename, "r", encoding="gbk")
        lines = d_file.readlines()
    return lines
def get_label_dict():
    filename = "E:/souhu/News_pic_label_train.txt"
    data_list = file_content(filename)
    label_dict={}
    for line in data_list:
        tuple=line.split("	")
        label_dict[tuple[0]]=tuple[1]
    return label_dict

def data_p(lines):
    words_list=[]
    labels_list=[]
    for line in lines:
        tuple = line.split("	")
        word_list = tuple[2:]
        words_list.append(word_list)
        labels_list.append(int(tuple[1]))
    return  words_list,labels_list
def read_data(filename):
    with open(filename,"r",encoding="utf-8") as f:
        lines=f.readlines()
    words_list,labels_list=data_p(lines)
    return words_list,labels_list

def load_train_data(filename):
    words_list, labels_list = read_data(filename)
    data_embeddings = DataEmbedding()
    ids_list=[]
    for words in words_list:
        #ids=[data_embeddings.token2index.get(word) for word in words ]
        ids=[]
        for word in words:
            id=data_embeddings.token2index.get(word)
            if id:
                ids.append(id)
            else:
                print(word)
                pass

    ids_list.append(ids)
    return ids_list,labels_list

if __name__ == '__main__':
    TRAIN_FILE = "E:/souhu/News_info_train_filter_word.txt"
    x,y=load_train_data(TRAIN_FILE)
    print(x[:5])
    print(y[:5])