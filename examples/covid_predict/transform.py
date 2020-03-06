# encoding: utf-8
# @Time    : 2020/3/2 下午10:25
# @Author  : Opsn
# @File    : transform.py
import json
import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def run():
    df = pd.read_csv("/tcdata/test.csv")
    data = df.to_dict(orient='list')
    ret = []
    for d in range(len(data["id"])):
        index = data["id"][d]
        text_a = data["query1"][d]
        text_b = data["query2"][d]
        category = data["category"][d]
        ret.append({"index": index, "text_a": text_a, "text_b": text_b, "label": 0, "category": category})
    json.dump(ret, open("dev.json", "w"))
    logging.info("finish test data transform...")


run()
