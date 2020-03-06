# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Sentence Classification Task Fine-Tuned with BERT """


import logging
import os
import json

from json import JSONDecodeError
import pandas as pd
from random import sample
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
    # multilabel_confusion_matrix


logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
        category: string. Category of the sentence
    """

    def __init__(self, guid, text_a, text_b=None, label=None, category=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.category = category


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None, category=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.category = category


def transform(e):
    return InputExample(guid=e["index"], text_a=e["text_a"], text_b=e.get("text_b", None), label=e["label"],
                        category=e["category"])


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    examples = json.load(open(file_path, "r"))
    examples = list(map(lambda e: transform(e), examples))
    return examples


def form_multi_label(label, label_list, multi_label_loss):
    if not isinstance(label, list):
        raise ValueError("在multi-label模式下，训练数据中的label字段必须为list...")
    if multi_label_loss == "MultiLabelSoftMarginLoss":
        ret = [0] * len(label_list)
        for l in label:
            index = label_list.index(l)
            if index == -1:
                raise ValueError("训练数据中出现了不在规范类别中的类别: {}".format(l))
            ret[index] = 1
        return ret
    elif multi_label_loss == "MultiLabelMarginLoss":
        ret = []
        for l in label:
            index = label_list.index(l)
            if index == -1:
                raise ValueError("训练数据中出现了不在规范类别中的类别: {}".format(l))
            ret.append(index)
        # padding with -1
        ret += [-1] * (len(label_list) - len(label))
        return ret


def convert_examples_to_features(
    examples,
    label_list,
    category_list,
    tokenizer,
    max_length=512,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads examples into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        label_list: List of ``labels``.
        category_list: List of ``categories``.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    label_map = {label: i for i, label in enumerate(label_list)}
    category_map = {category: i for i, category in enumerate(category_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d", ex_index)

        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
            )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
            )

        label = label_map[example.label]
        category = category_map[example.category]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: {} (id = {})".format(example.label, label))
            logger.info("category: {} (id = {})".format(example.category, category))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label,
                category=category
                )
            )
    return features


def is_unique(seq):
    if len(set(seq)) == len(seq):
        return True
    return False


def get_list_from_json(path):
    if not path:
        raise RuntimeError('文件路径为空...')
    try:
        labels = json.load(open(path, 'r'))
        if not is_unique(labels):
            raise RuntimeError('传入的list中item必须是不同的...')
        logging.info('文件: {}\t内容: {}'.format(path, labels))

    except JSONDecodeError:
        raise RuntimeError('label文件只支持json格式...')
    return labels


def model_result_print(preds, annos, label_list, prefix=""):
    # precision
    precision_score_micro = precision_score(annos, preds, average="micro")
    precision_score_macro = precision_score(annos, preds, average="macro")
    precision_score_detail = precision_score(annos, preds, labels=label_list, average=None)
    # recall
    recall_score_micro = recall_score(annos, preds, average="micro")
    recall_score_macro = recall_score(annos, preds, average="macro")
    recall_score_detail = recall_score(annos, preds, labels=label_list, average=None)
    # f1
    f1_score_micro = f1_score(annos, preds, average="micro")
    f1_score_macro = f1_score(annos, preds, average="macro")
    f1_score_detail = f1_score(annos, preds, labels=label_list, average=None)
    # format-output
    logger.info("***** Classification Metrics *****")
    logger.info("  P(Precision)_micro = {}  ".format(precision_score_micro))
    logger.info("  P(Precision)_macro = {}  ".format(precision_score_macro))
    for index, label in enumerate(label_list):
        logger.info("  P_{} = {}  ".format(label, precision_score_detail[index]))
    logger.info("  R(Recall)_micro = {}  ".format(recall_score_micro))
    logger.info("  R(Recall)_macro = {}  ".format(recall_score_macro))
    for index, label in enumerate(label_list):
        logger.info("  R_{} = {}  ".format(label, recall_score_detail[index]))
    logger.info("  F1_micro = {}  ".format(f1_score_micro))
    logger.info("  F1_macro = {}  ".format(f1_score_macro))
    for index, label in enumerate(label_list):
        logger.info("  F1_{} = {}  ".format(label, f1_score_detail[index]))
    return {"{}P_micro".format(prefix): precision_score_micro, "{}P_macro".format(prefix): precision_score_macro,
            "{}R_micro".format(prefix): recall_score_micro, "{}R_macro".format(prefix): recall_score_macro,
            "{}F1_micro".format(prefix): f1_score_micro, "{}F1_macro".format(prefix): f1_score_macro}


def compute_metrics(preds, annos, labels, prefix=""):
    label_list = get_list_from_json(labels)
    label_mapping = {i: d for i, d in enumerate(label_list)}
    preds = list(map(lambda p: label_mapping[p], preds))
    annos = list(map(lambda a: label_mapping[a], annos))
    res = model_result_print(preds, annos, label_list, prefix=prefix)
    return res
