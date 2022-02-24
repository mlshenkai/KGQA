# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2022/2/23 5:30 PM
# @Organization: YQN
# @Email: mlshenkai@163.com
import json
import os
import copy
from transformers import AutoTokenizer, PreTrainedTokenizer, BartTokenizerFast
from .ner_utils import DataProcessor
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, text_a, labels):
        """
        构建 InputExample对象
        Args:
            guid:
            text_a:
            labels:
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def __repr__(self):
        return str(self.to_json_string())


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def __repr__(self):
        return str(self.to_json_string())


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(
        torch.stack, zip(*batch)
    )
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens


def convert_example_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """

    Args:
        example: Example
        label_list:
        max_seq_length:
        tokenizer:
        cls_token_at_end:define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        cls_token:
        cls_token_segment_id: define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        sep_token:
        pad_on_left:
        pad_token:
        pad_token_segment_id:
        sequence_a_segment_id:
        mask_padding_with_zero:

    Returns:

    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("writing example %d of %d", ex_index, len(examples))
        if isinstance(example.text_a, list):
            example.text_a = " ".join(example.text_a)
        tokens = tokenizer.tokenize(example.text_a)
        label_ids = [label_map[x] for x in example.labels]
        special_tokens_count = 2  # CLS SEP 两个特殊占位符
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens += [sep_token]
        label_ids += [label_map["O"]]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label_map["O"]]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map["O"]] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_len = len(input_ids)
        input_mask = [1 if mask_padding_with_zero else 0] * input_len
        padding_length = max_seq_length - len(input_mask)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token] * padding_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join(list(map(str, tokens))))
            logger.info("input_ids: %s", " ".join(list(map(str, input_ids))))
            logger.info("input_mask: %s", " ".join(list(map(str, input_mask))))
            logger.info("segment_ids: %s", " ".join(list(map(str, segment_ids))))
            logger.info("label_ids: %s", " ".join(list(map(str, label_ids))))
        feature = InputFeatures(
            input_ids, input_mask, input_len, segment_ids, label_ids
        )
        features.append(feature)
    return features


class CnerProcessor(DataProcessor):
    """
    数据加载
    """

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_txt(os.path.join(data_dir, "train.char.bmes")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_txt(os.path.join(data_dir, "dev.char.bmes")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_txt(os.path.join(data_dir, "test.char.bmes")), "test"
        )

    def get_labels(self):
        return [
            "X",
            "B-CONT",
            "B-EDU",
            "B-LOC",
            "B-NAME",
            "B-ORG",
            "B-PRO",
            "B-RACE",
            "B-TITLE",
            "I-CONT",
            "I-EDU",
            "I-LOC",
            "I-NAME",
            "I-ORG",
            "I-PRO",
            "I-RACE",
            "I-TITLE",
            "O",
            "S-NAME",
            "S-ORG",
            "S-RACE",
            "[START]",
            "[END]",
        ]

    @staticmethod
    def _create_example(lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line["word"]
            labels = []
            for x in line["labels"]:
                if "M-" in x:
                    labels.append(x.replace("M-", "I-"))
                elif "E-" in x:
                    labels.append(x.replace("E-", "I-"))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


class CluenerProcessor:
    pass


if __name__ == "__main__":
    a = AutoTokenizer.from_pretrained("bert-base-chinese")
    print(a)
