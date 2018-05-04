#!/usr/bin/python
# -*- coding: utf-8 -*-
SEQ="	"
from TextClassification.keras_demo.data_parse.utils import *
from TextClassification.keras_demo.data_parse.sent_seg import sent_seg


def parse_seg(filename,outname):
    data_list=file_content(filename)
    label_dict=get_label_dict()
    data_seq=[]
    for line in data_list:
        tuple=line.split(SEQ)
        word_list=sent_seg.seg(tuple[1])
        if label_dict.get(tuple[0]):
            data_seq.append(tuple[0]+SEQ+label_dict.get(tuple[0])+SEQ+SEQ.join(word_list))
        else:
            data_seq.append(tuple[0] + SEQ + "6" + SEQ + SEQ.join(word_list))
    with open(outname, "w", encoding="utf-8") as f:
        f.writelines("\n".join(data_seq))

def bulid_data(filename1,filename2,outname):
    data_list = file_content(filename1)
    data_list1 = file_content(filename2)
    data_seq = []
    for line in data_list:
        tuple = line.split(SEQ)
        word_list = sent_seg.seg(tuple[1])
        data_seq.append(SEQ.join(word_list))
    for data in data_list1:
        tuple = data.split(SEQ)
        word_list = sent_seg.seg(tuple[1])
        data_seq.append(SEQ.join(word_list))

    with open(outname, "w", encoding="utf-8") as f:
        f.writelines("\n".join(data_seq))


if __name__ == '__main__':
    filename="E:/souhu/News_info_unlabel_filter.txt"
    filename1 = "E:/souhu/News_info_train_filter.txt"
    outname="E:/souhu/News_info_train_filter_word.txt"
    parse_seg(filename1,outname)
    #bulid_data(filename,filename1,"E:/搜狐/train_vec.txt")