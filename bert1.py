# Import libraries
import sys
import math
import csv
import torch
import argparse
import pandas as pd
import numpy as np
import pkbar


import config
import os
import torch.nn.functional as F
import torch.utils.data as data_utils

from transformers import BertForSequenceClassification, BertModel, BertTokenizerFast
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from datasets import Dataset
from torch.utils.data import DataLoader
from torch import nn
from types import SimpleNamespace

from dataset import DatasetResolver
from process import PreProcessText


TEXT_COL = "text"
Langue="Chinese"

pretrained_model = '.\models\\bert-base-Chinese'

# Parser
args = SimpleNamespace(
    dataset="hahackathon_zh",
    label="label",
    force=False,
    task="2a",
    evaluate=True
)

batch_size = 64

epochs = 2

# Get device
device = torch.device ('cuda') if torch.cuda.is_available () else torch.device ('cpu')
print(device)

torch.manual_seed (config.seed)

# Preprocess text
preprocess = PreProcessText()

models = []




# ---------------------------------------------------
# 1) 模型：支持传 bert_model（en/zh 自动）
# ---------------------------------------------------

# Get the model
class CustomBERTModelFineTunning(nn.Module):

    def __init__(self, num_labels=None):
        super(CustomBERTModelFineTunning, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model,
                                                                  return_dict=True,
                                                                  num_labels=num_labels,
                                                                  local_files_only=True)
        self.bert.to(device)

    def forward(self, input_ids,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None):
        bert_x = self.bert(input_ids, attention_mask=attention_mask)
        return bert_x.logits

# ---------------------------------------------------
# 2) tokenizer + tokenize：使用你的 TEXT_COL="text"
# ---------------------------------------------------
tokenizer = BertTokenizerFast.from_pretrained (pretrained_model)

def tokenize(batch):
    return tokenizer(batch[TEXT_COL], padding=True, truncation=True)

# ---------------------------------------------------
# 3) 主流程：读数据 -> 预处理 -> tokenize -> dataloader -> 训练/验证 -> 保存
# ---------------------------------------------------
if __name__ == "__main__":
    for key, dataset_options in config.datasets[args.dataset].items():

        # A) 加载 df
        dataset_path = os.path.join(config.directories["datasets"], Langue, "train.csv")
        print("[LOAD]", dataset_path)
        df = pd.read_csv(dataset_path).reset_index(drop=True)

        pd.set_option('display.max_columns', None)  # 显示所有列
        #print("df:",df)


        # 你没有 is_test 列，所以不做这个过滤（保留原处理逻辑，不硬加）
        # 如果以后加了 is_test，这里可以安全过滤：
        if "is_test" in df.columns:
            df = df.drop(df[df["is_test"] == True].index)

        # B) 通过 DatasetResolver 获取任务映射对象
        ds_obj = DatasetResolver().get(args.dataset, dataset_options, args.force)

        # 真正按 task 生成统一标签列 label
        if hasattr(ds_obj, "getDFFromTask"):
            df = ds_obj.getDFFromTask(args.task, df)
        # B) 任务类型
        task_type = "classification" if args.task in ["1a", "1c"] else "regression"

        # C) 必要列检查
        if TEXT_COL not in df.columns:
            raise KeyError(f"缺少文本列 '{TEXT_COL}'，当前列：{list(df.columns)}")

        label_col = "label"
        if "label" not in df.columns:
            raise KeyError(f"缺少标签列 '{label_col}'，当前列：{list(df.columns)}")

        # D)回归任务，删掉label
        if task_type == "regression":
            df = df.drop(df[df["label"] == 0.0].index).reset_index(drop=True)

        # E) 文本预处理：原脚本是对 df["tweet"]，这里改成 df["text"]
        for pipe in [
            "remove_urls", "remove_digits", "remove_whitespaces",
            "remove_elongations", "to_lower", "remove_punctuation"
        ]:
            df[TEXT_COL] = getattr(preprocess, pipe)(df[TEXT_COL])

        df[TEXT_COL] = preprocess.expand_acronyms(df[TEXT_COL], preprocess.EN_ACRONYMS)

        #tokenizer = BertTokenizerFast.from_pretrained(pretrained_model, local_files_only=True)

        if task_type == 'classification':
            df["label"] = df["label"].astype('category').cat.codes


        model = CustomBERTModelFineTunning(num_labels=len(df["label"].unique()) if task_type == 'classification' else 1)
        model.to(device)


        # G)  Dataset -> tokenize -> torch (直接让 HF Dataset 输出 torch 张量)
        dataset = Dataset.from_pandas(df)

        dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))

        dataset.set_format(
            'torch',
            columns=['input_ids', 'attention_mask', "label"],
            output_all_columns=True
        )



        x1 = dataset["input_ids"]
        x2 = dataset["attention_mask"]
        x3 = dataset["label"]

        print("input_ids type:", type(x1))
        print("attention_mask type:", type(x2))
        print(f"label type:", type(x3))

        # 如果是 tensor，就还能打印 shape；如果不是就会走 else
        import torch

        for name, x in [("input_ids", x1), ("attention_mask", x2), ("label", x3)]:
            if isinstance(x, torch.Tensor):
                print(name, "is torch.Tensor, shape =", tuple(x.shape), "dtype =", x.dtype)
            else:
                print(name, "NOT tensor ->", type(x), "has size attr?", hasattr(x, "size"))

        print("b\n")

        input_ids_tensor = torch.stack([dataset[i]["input_ids"] for i in range(len(dataset))])
        attention_mask_tensor = torch.stack([dataset[i]["attention_mask"] for i in range(len(dataset))])

        if task_type == "classification":
            label_tensor = torch.tensor(df["label"].values, dtype=torch.long)
        else:
            label_tensor = torch.tensor(df["label"].values, dtype=torch.float)

        dataset = data_utils.TensorDataset(
            input_ids_tensor,
            attention_mask_tensor,
            label_tensor
        )
        print("a\n")

        if args.evaluate:
            # split 的是 df（为了 stratify），但 sampler 用的必须是“位置索引”
            # ✅关键：确保 df 的 index 和 torch_ds 行号一致
            df = df.reset_index(drop=True)

            train_df, val_df = train_test_split(
                df,
                train_size=dataset_options["train_size"],
                random_state=config.seed,
                stratify=df[["label"]] if task_type == "classification" else None
            )

            train_sampler = torch.utils.data.SubsetRandomSampler(train_df.index.tolist())
            val_sampler = torch.utils.data.SubsetRandomSampler(val_df.index.tolist())

            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False)

        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            val_loader = None
            train_df = df
            val_df = None

        print("H/r")



        # I) optimizer & loss
        optimizer = AdamW(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss() if task_type == "classification" else torch.nn.MSELoss()
        criterion = criterion.to(device)

        train_per_epoch = int(math.ceil(len(train_df) / batch_size))

        metrics = (["epoch", "loss", "acc", "val_loss", "val_acc"] if (args.evaluate and task_type == "classification")
                   else ["epoch", "loss", "val_loss"] if args.evaluate
                   else ["epoch", "loss", "acc"] if task_type == "classification"
                   else ["epoch", "loss"])

        # J) train + (optional) val
        for epoch in range(1, epochs + 1):
            kbar = pkbar.Kbar(target=train_per_epoch, width=32, stateful_metrics=metrics)

            model.train()
            correct = 0
            seen = 0
            train_losses = []

            for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                preds = model(input_ids, attention_mask=attention_mask)

                if task_type == "classification":
                    preds = torch.squeeze(preds)
                    loss = criterion(preds, labels)

                    _, pred_class = torch.max(preds, dim=1)
                    correct += torch.sum(pred_class == labels).item()
                    seen += labels.size(0)
                    acc = correct / max(seen, 1)
                else:
                    preds=preds.squeeze(-1)
                    loss = torch.sqrt(criterion(preds, labels.float()) + 1e-6)

                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()

                kbar_values = [("epoch", epoch), ("loss", loss.item())]
                if task_type == "classification":
                    kbar_values.append(("acc", acc))
                kbar.add(1, values=kbar_values)

            # 验证
            if args.evaluate and val_loader is not None:
                model.eval()
                val_losses = []
                val_correct = 0
                val_seen = 0

                with torch.no_grad():
                    for input_ids, attention_mask, labels in val_loader:
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        labels = labels.to(device)

                        preds = model(input_ids, attention_mask=attention_mask)

                        if task_type == "classification":
                            preds = torch.squeeze(preds)
                            loss = criterion(preds, labels)

                            _, pred_class = torch.max(preds, dim=1)
                            val_correct += torch.sum(pred_class == labels).item()
                            val_seen += labels.size(0)
                        else:
                            preds=preds.squeeze(-1)
                            loss = torch.sqrt(criterion(preds, labels.float()) + 1e-6)

                        val_losses.append(loss.item())

                kbar_values = [("val_loss", float(np.mean(val_losses)))]
                if task_type == "classification":
                    kbar_values.append(("val_acc", val_correct / max(val_seen, 1)))
                kbar.add(0, values=kbar_values)

        # K) 保存模型（不管 evaluate 与否都保存）
        model_dir = os.path.join(
            config.directories["assets"],
            args.dataset,
            key,
            f"bert-finetune-{args.task}"
        )
        os.makedirs(model_dir, exist_ok=True)
        model.bert.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"[OK] saved: {model_dir}  (bert_model={pretrained_model})")