import sys
import math
import csv
import torch
import argparse
import pandas as pd
import numpy as np
import pkbar
from numba.cuda.simulator.cudadrv.nvvm import is_available

import config
import os
import torch.nn.functional as F
import torch.utils.data as data_utils

from transformers import BertForSequenceClassification, BertModel, BertTokenizerFast
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from bert1 import batch_size, TEXT_COL
from datasets import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch
from types import SimpleNamespace

from dataset import DatasetResolver
from process import PreProcessText


Langue="Chinese"
LABEL_COL="label"
TEXT_COL="text"

#pretrained_model='.\\assets\hahackathon_en\\base\\bert-finetune-2a'
pretrained_model='.\\assets\hahackathon_zh\\base\\bert-finetune-2a'
# Parser
args = SimpleNamespace(
    dataset="hahackathon_zh",
    label="label",
    force=False,
    task="2a",
    evaluate=True
)

batch_size=64

epochs=10

preprocess=PreProcessText()
Acronyn_Map = preprocess.EN_ACRONYMS
#Acronyn_Map = preprocess.ZH_ALIASES

models=[]

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

torch.manual_seed(config.seed)


task_type = "classification" if args.task in ["1a", "1c"] else "regression"

# ---------------------------------------------------
# 1) 模型：支持传 bert_model——finetune（en/zh 自动）
# ---------------------------------------------------


class CustomBERTModel(nn.Module):
    """
    CustomBERTModel

    This model mixes the fine tunned BERT model with custom features based
    on linguistic features
    """

    def __init__(self, input_size, num_classes):
        """
        @param input_size
        @param num_classes
        """
        super(CustomBERTModel, self).__init__()

        # Init BERT model
        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model, return_dict=True,
                                                                  output_hidden_states=True)

        # Linguistic features layer
        self.fc1 = nn.Linear(input_size + (768 * 1), 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, lf, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None,
                epoch=0):

        # Get BERT results
        with torch.no_grad():
            sequence_output = self.bert(input_ids, attention_mask=attention_mask)

        # Get BERT hidden_states
        hidden_states = sequence_output.hidden_states

        # @var cls_tokens The first token for each batch
        # @link https://stackoverflow.com/questions/61465103/how-to-get-intermediate-layers-output-of-pre-trained-bert-model-in-huggingface

        # This way works fine, getting the last layer
        cls_tokens = hidden_states[-1][:, 0]

        # To test
        """
        token_embeddings = hidden_states[-2][:, 1:]
        token_embeddings = torch.mean (token_embeddings, dim = -2)
        """

        # Combine BERT with LF
        combined_with_bert_features = torch.cat((cls_tokens, lf), dim=1)

        # Handle LF
        lf_x = F.relu(self.fc1(combined_with_bert_features))
        lf_x = F.relu(self.fc2(lf_x))

        # According to the task type, we need to apply a sigmoid function
        # or return the value as it is
        if task_type == 'classification':
            lf_x = torch.sigmoid(self.fc3(lf_x))
        else:
            lf_x = self.fc3(lf_x)

        return lf_x

# ---------------------------------------------------
# 2) tokenizer + tokenize：使用你的 TEXT_COL="text"
# ---------------------------------------------------
tokenizer = BertTokenizerFast.from_pretrained (pretrained_model)

def tokenize(batch):
    return tokenizer(batch['text'],padding=True,truncation=True)
# ---------------------------------------------------
# 3) 主流程：读数据 -> 预处理 -> tokenize -> dataloader -> 训练/验证 -> 保存
# ---------------------------------------------------
if __name__ == "__main__":

    for key, dataset_options in config.datasets[args.dataset].items():


        # A） 加载 df
        dataset_path = os.path.join(config.directories["datasets"], Langue, "train.csv")
        print("[LOAD]", dataset_path)
        Bert_df = pd.read_csv(dataset_path).reset_index(drop=True)

        #pd.set_option('display.max_columns', None)  # 显示所有列
        # print("df:",df)

        if "is_test" in Bert_df.columns:
            Bert_df = Bert_df.drop(Bert_df[Bert_df["is_test"] == True].index)


        # ---------------------------------
        # B) Load dataset 通过 DatasetResolver 获取任务映射对象
        # ---------------------------------
        ds_obj = DatasetResolver().get(args.dataset, dataset_options, args.force)

       # df = ds_obj.get().reset_index(drop=True)


        # Task-specific label processing
        if hasattr(ds_obj, "getDFFromTask"):
            Bert_df = ds_obj.getDFFromTask(args.task, Bert_df)

        # Required columns check
        if TEXT_COL not in Bert_df.columns:
            raise KeyError(f"缺少文本列 '{TEXT_COL}'，当前列：{list(Bert_df.columns)}")

        if LABEL_COL not in Bert_df.columns:
            raise KeyError(f"缺少标签列 '{LABEL_COL}'，当前列：{list(Bert_df.columns)}")

        # ---------------------------------
        # C) Text preprocessing
        # ---------------------------------
        for pipe in [
            "remove_urls","remove_digits","remove_whitespaces",
            "remove_elongations","to_lower","remove_punctuation"
        ]:
            Bert_df[TEXT_COL] = getattr(preprocess, pipe)(Bert_df[TEXT_COL])


        # Then, expand contractions
        Bert_df['text'] = preprocess.expand_acronyms(Bert_df['text'], Acronyn_Map)


        # ---------------------------------
        # D) Load LF file
        # ---------------------------------
        lf_path = os.path.join(
            config.directories["assets"],
            args.dataset,
            key,
            "lf.csv"
        )

        print(lf_path)

        if not os.path.isfile(lf_path):
            raise FileNotFoundError(
                f"找不到 lf.csv：{lf_path}\n"
                f"bert-final 这条线必须依赖现成的 lf.csv。"
            )

        df_lf = pd.read_csv(lf_path, header=0, sep=",")

        # Keep original compatibility
        if "class" in df_lf.columns and LABEL_COL not in df_lf.columns:
            df_lf = df_lf.rename(columns={"class": LABEL_COL})


        df_lf = df_lf.loc[:, (df_lf != 0).any(axis=0)]
        df_lf = df_lf.loc[Bert_df.index]
        #df_lf_x = df_lf.iloc[:, :-1]
        df_lf_x = df_lf.copy()

        df = pd.concat([Bert_df, df_lf_x], axis=1)
        df = df[[c for c in df if c not in ['label']] + ['label']]

        # ---------------------------------
        # E) Regression special handling
        # ---------------------------------
        '''
            if task_type == "regression":
            df = df.drop(df[df['label'] == 0.0].index).reset_index()
            df_lf_x = df_lf_x.loc[df.index]
            '''
        if task_type == "regression":
            keep_idx = df[df['label'] != 0.0].index
            df = df.loc[keep_idx].reset_index(drop=True)
            df_lf_x = df_lf_x.loc[keep_idx].reset_index(drop=True)

        if task_type == "classification":
            df[LABEL_COL] = df[LABEL_COL].astype("category").cat.codes

        # ---------------------------------
        # G) Build model
        # ---------------------------------
        input_size = df_lf_x.shape[1]
        num_classes = len(df[LABEL_COL].unique()) if task_type == "classification" else 1

        model = CustomBERTModel(
            input_size=input_size,
            num_classes=num_classes
        )
        model.to(device)

        # ---------------------------------
        # H) HuggingFace Dataset + tokenize
        # ---------------------------------
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))

        dataset.set_format(
            "torch",
            columns=["input_ids", "attention_mask", LABEL_COL],
            output_all_columns=True
        )

        x1 = dataset["input_ids"]
        x2 = dataset["attention_mask"]
        x3 = dataset[LABEL_COL]

        print("input_ids type:", type(x1))
        print("attention_mask type:", type(x2))
        print(f"label type:", type(x3))

        # 如果是 tensor，就还能打印 shape；如果不是就会走 else
        import torch

        for name, x in [("input_ids", x1), ("attention_mask", x2), (LABEL_COL, x3)]:
            if isinstance(x, torch.Tensor):
                print(name, "is torch.Tensor, shape =", tuple(x.shape), "dtype =", x.dtype)
            else:
                print(name, "NOT tensor ->", type(x), "has size attr?", hasattr(x, "size"))

        print("b\n")



        # Convert to TensorDataset
        input_ids_tensor = torch.stack([dataset[i]["input_ids"] for i in range(len(dataset))])
        attention_mask_tensor = torch.stack([dataset[i]["attention_mask"] for i in range(len(dataset))])

        if task_type == "classification":
            label_tensor = torch.tensor(df[LABEL_COL].values, dtype=torch.long)
        else:
            label_tensor = torch.tensor(df[LABEL_COL].values, dtype=torch.float)

        dataset = data_utils.TensorDataset(
            torch.tensor(df_lf_x.values, dtype=torch.float),
            input_ids_tensor,
            attention_mask_tensor,
            label_tensor
        )
        print("a\n")

        # ---------------------------------
        # I) Split train / val
        # ---------------------------------
        if args.evaluate:
            df = df.reset_index(drop=True)

            train_df, val_df = train_test_split(
                df,
                train_size=dataset_options["train_size"],
                random_state=config.seed,
                stratify=df[[LABEL_COL]] if task_type == "classification" else None
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

        # ---------------------------------
        # J) Optimizer & loss
        # ---------------------------------
        optimizer = AdamW(model.parameters(), lr=1e-2)

        criterion = torch.nn.CrossEntropyLoss()if task_type == "classification" else torch.nn.MSELoss()
        criterion = criterion.to(device)

        train_per_epoch = int(math.ceil(len(train_df) / batch_size))

        metrics = (
            ["epoch", "loss", "acc", "val_loss", "val_acc"]
            if (args.evaluate and task_type == "classification")
            else ["epoch", "loss", "val_loss"]
            if args.evaluate
            else ["epoch", "loss", "acc"]
            if task_type == "classification"
            else ["epoch", "loss"]
        )

        # For saving best model
        best_val_loss = float("inf")

        # Save path
        save_path = os.path.join(
            config.directories["assets"],
            args.dataset,
            key,
            f"bert-final-{args.task}.pt"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # ---------------------------------
        # K) Training loop
        # ---------------------------------
        for epoch in range(1, epochs + 1):
            kbar = pkbar.Kbar(target=train_per_epoch, width=32, stateful_metrics=metrics)

            model.train()
            correct = 0
            seen = 0
            train_losses = []

            for i, (lf, input_ids, attention_mask, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                lf = lf.to(device).float()
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                preds = model(lf, input_ids, attention_mask=attention_mask)

                if task_type == "classification":
                    preds = torch.squeeze(preds)
                    loss = criterion(preds, labels)

                    _, pred_class = torch.max(preds, dim=1)
                    correct += torch.sum(pred_class == labels).item()
                    seen += labels.size(0)
                    acc = correct / max(seen, 1)
                else:
                    preds = preds.squeeze(-1)
                    loss = torch.sqrt(criterion(preds, labels.float()) + 1e-6)

                train_losses.append(loss.item())

                loss.backward()
                optimizer.step()

                kbar_values = [("epoch", epoch), ("loss", loss.item())]
                if task_type == "classification":
                    kbar_values.append(("acc", acc))
                kbar.add(1, values=kbar_values)

            # -------------------------
            # Validation
            # -------------------------
            if args.evaluate and val_loader is not None:
                model.eval()
                val_losses = []
                val_correct = 0
                val_seen = 0

                with torch.no_grad():
                    for lf, input_ids, attention_mask, labels in val_loader:
                        lf = lf.to(device).float()
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        labels = labels.to(device)

                        preds = model(lf, input_ids, attention_mask=attention_mask)

                        if task_type == "classification":
                            preds = torch.squeeze(preds)
                            loss = criterion(preds, labels)

                            _, pred_class = torch.max(preds, dim=1)
                            val_correct += torch.sum(pred_class == labels).item()
                            val_seen += labels.size(0)
                        else:
                            preds = preds.squeeze(-1)
                            loss = torch.sqrt(criterion(preds, labels.float()) + 1e-6)

                        val_losses.append(loss.item())

                current_val_loss = float(np.mean(val_losses))

                kbar_values = [("val_loss", current_val_loss)]
                if task_type == "classification":
                    kbar_values.append(("val_acc", val_correct / max(val_seen, 1)))
                kbar.add(0, values=kbar_values)

                # Save best model when evaluate=True
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    torch.save(model.state_dict(), save_path)
                    print(f"[OK] Best model saved to: {save_path}, val_loss={best_val_loss:.4f}")

        # ---------------------------------
        # L) Save final model if no validation
        # ---------------------------------
        if not args.evaluate:
            torch.save(model.state_dict(), save_path)
            print(f"[OK] Final model saved to: {save_path}")

