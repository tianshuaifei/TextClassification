#!/usr/bin/python
# -*- coding: utf-8 -*-
from os.path import join, dirname, abspath
PROJECT_DIR = dirname(dirname(abspath(__file__))) # 3
def file_content(filename):
    try:
        d_file = open(filename, "r",encoding="utf-8")
        lines = d_file.readlines()
    except UnicodeDecodeError as e:
        d_file = open(filename, "r", encoding="gbk")
        lines = d_file.readlines()
    return lines

def write_dict(list):
    with open(PROJECT_DIR+'/library/feature_dict', 'w',encoding="utf-8") as file:
        file.writelines("\n".join(list))

def readfeature():
    with open(PROJECT_DIR+'/library/feature_dict', 'r',encoding="utf-8") as file:
        lines=file.readlines()
    return lines

def write_result_file(pred,result_filename="result_rf.txt"):
    result_list = []
    seq_resu=[]
    with open("E:/souhu/News_info_validate_filter.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            tuple = line.split("	")
            result_list.append(tuple[0] + "	" + str(pred[i]) + "	" + "NULL" + "	" + "NULL")
            if pred[i]==1:
                t=list(tuple[1].strip())
                seq_resu.append(tuple[0]+"	"+str(" ".join(t)))


    with open("E:/souhu/"+result_filename, "w", encoding="utf-8") as f:
        f.writelines("\n".join(result_list))
    with open("E:/souhu/"+"seg_pre.txt", "w", encoding="utf-8") as f:
        f.writelines("\n".join(seq_resu))


def get_label_dict():
    filename = "E:/souhu/News_pic_label_train.txt"
    data_list = file_content(filename)
    label_dict={}
    for line in data_list:
        tuple=line.split("	")
        label_dict[tuple[0]]=tuple[1]
    return label_dict