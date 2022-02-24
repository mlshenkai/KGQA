# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2022/2/22 11:15 AM
# @Organization: YQN
# @Email: mlshenkai@163.com
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from nlu.bert_intent_recognition.model.bert_textcnn_model import BertTextCnn


class BertTextCnnInfer:
    def __init__(self, config: dict):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BertTextCnn(config)
        model.load_state_dict(torch.load("./text_cnn.pth"))
        self.model = model.to(self.device)
        self.model.eval()
        self.id_map = {
            5: "治疗方法",
            3: "临床表现(病症表现)",
            11: "治疗时间",
            12: "其他",
            9: "禁忌",
            2: "预防",
            1: "病因",
            0: "定义",
            8: "治愈率",
            6: "所属科室",
            10: "化验/体检方案",
            7: "传染性",
            4: "相关病症",
        }

    def predict(self, text):
        encode_pair = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=70,
            return_tensors="pt",
        )
        input_ids = encode_pair["input_ids"].to(self.device)
        attention_mask = encode_pair["attention_mask"].to(self.device)
        token_type_ids = encode_pair["token_type_ids"].to(self.device)
        with torch.no_grad():
            pred = self.model([input_ids, attention_mask, token_type_ids])
            pre_id = torch.argmax(pred, dim=-1)
            print(self.id_map[pre_id.data.cpu().numpy()[0]])
        print(pred)


if __name__ == "__main__":
    with open("./config/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    model = BertTextCnnInfer(config)
    model.predict("头颅CT显示：多发颅内出血，蛛网膜下腔出血，脑水肿，中线移位。如何治疗")
