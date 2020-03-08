# encoding: utf-8
# @Time    : 2020/3/2 下午7:01
# @Author  : Opsn
# @File    : run.py
import json
import logging
import pandas as pd
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

FILE_MAPPING = {
    "train": "raw/train_20200228.csv",
    "dev": "raw/dev_20200228.csv"
    }


def run_label_v1(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for mode in FILE_MAPPING:
        file = FILE_MAPPING[mode]
        df = pd.read_csv(file)
        data = df.to_dict(orient='list')
        ret = []
        for d in range(len(data["id"])):
            index = data["id"][d]
            text_a = data["query1"][d]
            text_b = data["query2"][d]
            label = data["label"][d]
            category = data["category"][d]
            ret.append({"index": index, "text_a": text_a, "text_b": text_b, "label": label, "category": category})
        file_name = os.path.join(dir_name, "{}.json".format(mode))
        json.dump(ret, open(file_name, "w"))
        logging.info("完成{}数据的转化...".format(mode))


run_label_v1("data_v1")


def run_label_v2(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for mode in FILE_MAPPING:
        file = FILE_MAPPING[mode]
        df = pd.read_csv(file)
        data = df.to_dict(orient='list')
        ret = []
        for d in range(len(data["id"])):
            index = data["id"][d]
            text_a = data["query1"][d]
            text_b = data["query2"][d]
            label = data["label"][d]
            category = data["category"][d]
            text_a = "{}#{}".format(category, text_a)
            ret.append({"index": index, "text_a": text_a, "text_b": text_b, "label": label})
        file_name = os.path.join(dir_name, "{}.json".format(mode))
        json.dump(ret, open(file_name, "w"))
        logging.info("完成{}数据的转化...".format(mode))


run_label_v2("data_v2")

