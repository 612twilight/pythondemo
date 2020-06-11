import math
import re
from collections import OrderedDict

import jieba
import numpy as np
import tensorflow as tf


def load_data():
    raw_texts = ["文档一", "文档2", "文档3", "文档4", "文档5", "文档6", "文档7"]
    labels = ["label1", "label2", "label2", "label1", "label3", "label2", "label1"]
    return raw_texts, labels


def handle_data(raw_texts, labels):
    no_chinese = re.compile("[^\u4e00-\u9f5a]")
    handle_texts = []
    handle_labels = []
    for text, label in zip(raw_texts, labels):
        text = no_chinese.sub("", text)
        if not text:
            continue
        split_words = filter(lambda x: x and x != " ", jieba.cut(text))
        handle_texts.append(" ".join(split_words))
        handle_labels.append(label)
    return handle_texts, handle_labels


def generate_tf_records(texts, labels, tf_record_path, token_path, label_path, max_len):
    token_set = set()
    label_set = set()
    for text, label in zip(texts, labels):
        token_set.update(text.split())
        label_set.update(label)
    token_set = list(sorted(token_set))
    label_set = list(sorted(label_set))
    token_set.insert(0, '[PAD]')
    token_set.insert(1, '[UNK]')
    with open(token_path, 'w', encoding='utf8') as writer:
        writer.write("\n".join(token_set))
    with open(label_path, 'w', encoding='utf8') as writer:
        writer.write("\n".join(label_set))
    index_label = dict((int(index), label) for index, label in enumerate(label_set))
    index_token = dict((int(index), token) for index, token in enumerate(token_set))
    token_index = dict((token, index) for index, token in index_token.items())
    label_index = dict((label, index) for index, label in index_label.items())
    file_based_convert_examples_to_features(texts, labels, token_index, label_index, tf_record_path, max_len)


class Feature():
    def __init__(self, input_ids, label_ids):
        self.input_ids = input_ids
        self.label_ids = label_ids


def convert_to_feature(example, token_index, label_index, max_len):
    feature, label = example
    input_ids = [token_index[token] if token in token_index else token_index['[UNK]'] for token in feature]
    index = label_index[label]
    label_ids = np.zeros(len(label_index), dtype=np.int32)
    label_ids[index] = 1
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids.extend([token_index['[PAD]'] * (max_len - len(input_ids))])
    return Feature(input_ids, label_ids)


def file_based_convert_examples_to_features(features, labels, token_index, label_index, tf_record_path, max_len):
    writer = tf.io.TFRecordWriter(tf_record_path)

    def _int64_features(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    sample_len = len(features)
    point = sample_len // 5
    for index, example in enumerate(zip(features, labels)):
        if 0 == (index + 1) % point:
            log_info = "tfrecords生成中，进度 %d %%" % math.ceil((index + 1) * 100 / sample_len)
            print(log_info)
        feature = convert_to_feature(example, token_index, label_index, max_len)
        order_dict = OrderedDict()
        order_dict["input_ids"] = _int64_features(feature.input_ids)
        order_dict['label_ids'] = _int64_features(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=order_dict))
        writer.write(tf_example.SerializeToString())
    writer.close()


def read_tf_records(file_path, max_len, label_len):
    def parse_example(record):
        feature_dict = {
            "input_ids": tf.io.FixedLenFeature([max_len], tf.int64),
            "label_ids": tf.io.FixedLenFeature([label_len], tf.int64)
        }
        example = tf.io.parse_single_example(record, feature_dict)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t
        return {"input_ids": example['input_ids']}, {"score": example['label_ids']}

    data_set = tf.data.TFRecordDataset([file_path])
    data_set = data_set.map(parse_example)
    return data_set
