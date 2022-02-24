# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2022/2/23 1:59 PM
# @Organization: YQN
# @Email: mlshenkai@163.com
import torch
from collections import Counter
from konwledge_extraction.processors.ner_utils import get_entities


class SeqEntityScore(object):
    """
    char 级别
    """

    def __init__(self, id_label, markup="bios"):
        self.id_label = id_label
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    @staticmethod
    def compute(origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = (
            0.0
            if recall + precision == 0
            else (2 * precision * recall) / (precision + recall)
        )
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {
                "acc": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return (
            {
                "acc": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            },
            class_info,
        )

    def update(self, label_paths, pred_paths):
        """
        :param label_paths:  [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        :param pred_paths: [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        :return:
        """
        for label_path, pred_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path, self.id_label)
            pred_entities = get_entities(pred_path, self.id_label)
            self.origins.extend(label_entities)
            self.founds.extend(pred_entities)
            self.rights.extend(
                [
                    pre_entity
                    for pre_entity in pred_entities
                    if pre_entity in label_entities
                ]
            )


class SpanEntityScore(object):
    def __init__(self, id_label):
        self.id_label = id_label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    @staticmethod
    def compute(origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = (
            0.0
            if recall + precision == 0
            else (2 * precision * recall) / (precision + recall)
        )
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([self.id_label[x[0]] for x in self.origins])
        found_counter = Counter([self.id_label[x[0]] for x in self.founds])
        right_counter = Counter([self.id_label[x[0]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {
                "acc": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {"acc": precision, "recall": recall, "f1": f1}, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend(
            [pre_entity for pre_entity in pred_subject if pre_entity in true_subject]
        )
