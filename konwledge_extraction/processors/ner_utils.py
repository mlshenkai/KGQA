# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2022/2/23 2:03 PM
# @Organization: YQN
# @Email: mlshenkai@163.com
import csv
import json
import torch
from transformers import AutoTokenizer


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quote_char=None):
        "读取csv文件，可指定分隔符"
        with open(input_file, "r", encoding="urf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quote_char)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_txt(cls, input_file):
        lines = []
        with open(input_file, "r", encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    @classmethod
    def _read_json(cls, input_file):
        lines = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line.strip())
                text = line["text"]
                label_entities = line.get("label", None)
                words = list(text)
                labels = ["O"] * len(words)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert (
                                    "".join(words[start_index : end_index + 1])
                                    == sub_name
                                )
                                if start_index == end_index:
                                    labels[start_index] = "S-" + key
                                else:
                                    labels[start_index] = "B-" + key
                                    labels[start_index + 1 : end_index + 1] = [
                                        "I-" + key
                                    ] * (len(sub_name) - 1)
                lines.append({"words": words, "labels": labels})
            return lines


def get_entity_bios(seq, id_label):
    """
    获取bios entity
    :param seq:
    :param id_label:
    :return:
    """

    chunks = []
    chunk = [-1, -1, -1]
    for idx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id_label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = idx
            chunk[2] = idx
            chunk[0] = "-".join(tag.split("-")[1:])
            chunks.append(chunk)
            chunk = [-1, -1, -1]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = idx
            chunk[0] = "-".join(tag.split("-")[1:])
        elif tag.startswith("I-") and chunk[1] != -1:
            _type = "-".join(tag.split("-")[1:])
            if _type == chunk[0]:
                chunk[2] = idx
            if idx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id_label):
    """
    获取bio entity
    :param seq:
    :param id_label:
    :return:
    """
    chunks = []
    chunk = [-1, -1, -1]
    for idx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id_label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = idx
            chunk[0] = tag.split("-")[1]
            chunk[2] = idx
            if idx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith("I-") and chunk[1] != -1:
            _type = tag.split("-")[1]
            if _type == chunk[0]:
                chunk[2] = idx

            if idx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, id_label, markup="bios"):
    """
    获取tag
    :param seq:
    :param id_label:
    :param markup:
    :return:
    """
    assert markup in ["bios", "bio"]
    if markup == "bio":
        return get_entity_bio(seq, id_label)
    else:
        return get_entity_bios(seq, id_label)


def bert_extract_item(start_logit, end_logit):
    s = []
    start_pred = torch.argmax(start_logit, dim=-1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logit, dim=-1).cpu().numpy()[0][1:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred):
            if s_l == e_l:
                s.append([s_l, i, i * j])
                break
    return s
