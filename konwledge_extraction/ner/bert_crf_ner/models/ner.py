# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2022/2/22 8:42 PM
# @Organization: YQN
# @Email: mlshenkai@163.com
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from .crf import CRF
from transformers import AutoModel
from konwledge_extraction.ner.bert_crf_ner.losses import (
    FocalLoss,
    LabelSmoothingCrossEntry,
)
from torch.nn import CrossEntropyLoss
from .linear import PoolerStartLogits, PoolerEndLogits


class BertSoftMaxForNer(nn.Module):
    def __init__(self, config: dict):
        super(BertSoftMaxForNer, self).__init__()
        pre_model_name = config.get("pre_model_name", "bert-base-chinese")
        self.num_labels = config["num_labels"]
        self.bert = AutoModel.from_pretrained(pre_model_name)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.classifier = nn.Linear(config["hidden_size"], self.num_labels)
        self.loss_type = config["loss_type"]

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            assert self.loss_type in ["lsr", "focal", "ce"]
            if self.loss_type == "focal":
                loss_ft = FocalLoss(ignore_index=0)
            elif self.loss_type == "lsr":
                loss_ft = LabelSmoothingCrossEntry(ignore_index=0)
            else:
                loss_ft = CrossEntropyLoss(ignore_index=0)
            "只计算不被mask部分的损失"
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logit = logits.view(-1, self.num_labels)[active_loss]
                active_label = labels.view(-1)[active_loss]
                loss = loss_ft(active_logit, active_label)
            else:
                loss = loss_ft(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


class BertCrfForNer(nn.Module):
    def __init__(self, config: dict):
        super(BertCrfForNer, self).__init__()
        pre_model_name = config.get("pre_model_name", "bert-base-chinese")
        self.num_labels = config["num_labels"]
        self.bert = AutoModel.from_pretrained(pre_model_name)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.classifier = nn.Linear(config["hidden_size"], self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logit = self.classifier(sequence_output)
        outputs = (logit,)
        if labels is not None:
            loss = self.crf(emissions=logit, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs


class BertSpanForNer(nn.Module):
    def __init__(self, config: dict):
        super(BertSpanForNer, self).__init__()
        self.config = config
        pre_model_name = config.get("pre_model_name", "bert-base-chinese")
        self.num_labels = config["num_labels"]
        self.bert = AutoModel.from_pretrained(pre_model_name)
        self.soft_label = config["soft_label"]
        self.loss_type = config["loss_type"]
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.start_fc = PoolerStartLogits(config["hidden_size"], self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(
                config["hidden_size"] + self.num_labels, self.num_labels
            )
        else:
            self.end_fc = PoolerEndLogits(config["hidden_size"] + 1, self.num_labels)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        start_position=None,
        end_position=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logit = self.start_fc(sequence_output)
        if start_position is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logit = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logit.zero_()
                label_logit = label_logit.to(input_ids.device)
                label_logit.scatter_(2, start_position.unsqueeze(2), 1)
            else:
                label_logit = start_position.unsqueeze(2).float()
        else:
            label_logit = F.softmax(start_logit, -1)
            if not self.soft_label:
                label_logit = torch.argmax(label_logit, -1).unsqueeze(2).float()
        end_logit = self.end_fc(sequence_output, label_logit)
        outputs = (start_logit, end_logit,) + outputs[2:]

        if start_position is not None and end_position is not None:
            assert self.loss_type in ["lsr", "focal", "ce"]
            if self.loss_type == "lsr":
                loss_fct = LabelSmoothingCrossEntry()
            elif self.loss_type == "focal":
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            start_logit = start_logit.view(-1, self.num_labels)
            end_logit = end_logit.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logit = start_logit[active_loss]
            active_end_logit = end_logit[active_loss]

            active_start_labels = start_position.view(-1)[active_loss]
            active_end_labels = end_position.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logit, active_start_labels)
            end_loss = loss_fct(active_end_logit, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs
