# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2022/2/21 5:53 PM
# @Organization: YQN
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import json
import torch.nn.functional as F
from nlu.bert_intent_recognition.data.data_utils import DataSet, load_data
from nlu.bert_intent_recognition.model.bert_textcnn_model import BertTextCnn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config: dict):
    model = BertTextCnn(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    train_text_list, train_label_list, label_map = load_data(
        "./resources/files/train.csv"
    )
    train_dataset = DataSet(
        config, text_list=train_text_list, label_list=train_label_list
    )
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=1
    )
    eval_text_list, eval_label_list, _ = load_data("./resources/files/test.csv")
    eval_dataset = DataSet(config, text_list=eval_text_list, label_list=eval_label_list)
    eval_dataloader = data.DataLoader(eval_dataset)
    sum_loss = 0.0
    total_step = len(train_dataloader)
    train_curve = []
    for epoch in range(100):
        for i, inputs in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs_tensor = [p.to(device) for p in inputs]
            pred = model([inputs_tensor[0], inputs_tensor[1], inputs_tensor[2]])
            pred_dim = torch.softmax(pred,dim=-1)
            # acc = torchmetrics.functional.accuracy(pred_dim,inputs[3])
            # print(acc)
            loss = loss_fn(pred, inputs_tensor[3])
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print("[{}|{} step: {}/{} loss: {:.4f}]".format(epoch+1, 100, i+1, total_step, loss.item()))
        train_curve.append(sum_loss)
        sum_loss = 0
    torch.save(model.state_dict(),"./text_cnn.pth")
    print(train_curve)


def eval(model, eval_dataloader):
    model.eval()


if __name__ == "__main__":
    with open("./config/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    train(config)
