# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2022/2/21 5:37 PM
# @Organization: YQN
# @Email: mlshenkai@163.com
import torch.nn as nn
from torch.utils import data
from transformers import AutoTokenizer
import csv


class DataSet(data.Dataset):
    def __init__(
        self, config: dict, text_list: list, label_list: list = None, with_label=True
    ):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get("pre_model_name", "bert-base-chinese")
        )
        self.with_label = with_label
        self.text_list = text_list
        self.label_list = label_list

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        text = self.text_list[index]
        encode_pair = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=70,
            return_tensors="pt",
        )
        input_ids = encode_pair["input_ids"].squeeze(0)
        attention_mask = encode_pair["attention_mask"].squeeze(0)
        token_type_ids = encode_pair["token_type_ids"].squeeze(0)
        if self.with_label:
            label = self.label_list[index]
            return input_ids, attention_mask, token_type_ids, label
        return input_ids, attention_mask, token_type_ids

def load_data(data_file_path):
    text_list = []
    label_list = []
    label_map = dict()
    with open(data_file_path,"r",encoding="utf-8") as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            text = row[0]
            label = int(row[2])
            if label_map.get(label) is None:
                label_map[label] = row[1]
            label_list.append(label)
            text_list.append(text)
    return text_list, label_list, label_map


if __name__ == "__main__":
    file_path = "../resources/files/train.csv"
    text_list, label_list, label_map = load_data(file_path)
    print(label_map)