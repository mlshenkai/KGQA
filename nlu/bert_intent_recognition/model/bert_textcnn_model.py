# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2022/2/20 7:52 PM
# @Organization: YQN
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig, PretrainedConfig
import json

# config = AutoConfig.from_pretrained("bert-base-chinese")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AutoModel.from_pretrained("bert-base-chinese").to(DEVICE)
#
# print(model)
# inputs = tokenizer("我是上海海事大学信息工程学院的老师", return_tensors="pt")
# outputs = model(**inputs)
# print(outputs)
# pooler_output = outputs["pooler_output"]
# print(pooler_output)


class TextCNN(nn.Module):
    def __init__(self, config: dict):
        super(TextCNN, self).__init__()
        self.config = config
        self.filter_sizes = config["filter_sizes"]
        self.num_filters = config["num_filters"]
        self.hidden_size = config["hidden_size"]
        self.encode_layer = config["encode_layer"]
        self.num_filter_total = self.num_filters * len(self.filter_sizes)
        self.num_class = config["num_class"]
        self.linear = nn.Linear(self.num_filter_total, self.num_class, bias=False)
        self.bias = nn.Parameter(torch.ones([self.num_class]))
        self.filter_list = nn.ModuleList(
            [
                nn.Conv2d(1, self.num_filters, kernel_size=(size, self.hidden_size))
                for size in self.filter_sizes
            ]
        )

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        x = x.unsqueeze(1)  # x: [batch_size, 1, seq_len, hidden_size]
        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x))  # batch_size, 1, seq_len=kernel_size+1, 1
            mp = nn.MaxPool2d(
                kernel_size=(self.encode_layer - self.filter_sizes[i] + 1, 1)
            )
            pooled = mp(h).permute(0, 3, 2, 1)  # batch, 1, 1, 1
            pooled_outputs.append(pooled)
        h_pool = torch.cat(
            pooled_outputs, len(self.filter_sizes)
        )  # batch_size, 1, 1, 3*3
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])
        output = self.linear(h_pool_flat) + self.bias
        return output


class BertTextCnn(nn.Module):
    def __init__(self, config: dict):
        super(BertTextCnn, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(
            self.config.get("pre_model_name", "bert-base-chinese"),
            output_hidden_states=True,
            return_dict=True,
        )
        self.hidden_size = config["hidden_size"]
        self.num_class = config["num_class"]
        self.linear = nn.Linear(self.hidden_size, self.num_class)
        self.text_cnn = TextCNN(config)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x[0], x[1], x[2]
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_status = outputs.hidden_states
        cls_embeddings = hidden_status[1][:, 0, :].unsqueeze(1)  # batch_size, 1, hidden
        for i in range(2,13):
            cls_embeddings = torch.cat([cls_embeddings,hidden_status[i][:, 0, :].unsqueeze(1)],dim=1)
        logits = self.text_cnn(cls_embeddings)
        return logits


if __name__ == "__main__":
    with open("../config/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    tokenizers = AutoTokenizer.from_pretrained(config["pre_model_name"])
    encode_pair = tokenizers(
        "我是上海海事大学信息工程学院的老师",
        padding="max_length",
        truncation=True,
        max_length=150,
        return_tensors="pt",
    )
    input_ids = encode_pair["input_ids"]
    attention_mask = encode_pair["attention_mask"]
    token_type_ids = encode_pair["token_type_ids"]
    bert_text_cnn = BertTextCnn(config)
    bert_text_cnn([input_ids, attention_mask, token_type_ids])
