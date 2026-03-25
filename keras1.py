"""
    Keras and Talos for hyper-parameter tuning
    revised version for HaHackathon EN task 1b
"""

import random
import os
import pickle
import datetime

import tensorflow
import talos
import sklearn
import numpy as np
import pandas as pd
import keras

import config
import kerasmodel
import kerasutils

from types import SimpleNamespace
from dataset import DatasetResolver
from process import PreProcessText
from sklearn.linear_model import LassoCV


# =========================
# Global config
# =========================
LANG = "English"
LABEL_COL = "label"
TEXT_COL = "text"

args = SimpleNamespace(
    dataset="hahackathon_en",
    label="label",
    force=False,
    task="1c",
    evaluate=True,
    minutes=60 * 24 * 7,
    permutations=None,
    patience=10
)

max_words = None

random.seed(config.seed)
np.random.seed(config.seed)
tensorflow.random.set_seed(config.seed)

preprocess = PreProcessText()


def safe_select_lf_features(train_df_lf, val_df_lf, label_col, task_type):
    """
    Select LF features using training split only.
    Avoid data leakage.
    """
    candidate_cols = [c for c in train_df_lf.columns if c != label_col]

    # 去掉全 0 列，但只检查特征列，不碰 label
    non_zero_cols = [c for c in candidate_cols if (train_df_lf[c] != 0).any()]
    if not non_zero_cols:
        return train_df_lf, val_df_lf

    X_train = train_df_lf[non_zero_cols]

    if task_type == "classification":
        y_train = train_df_lf[label_col].astype("category").cat.codes
    else:
        y_train = pd.to_numeric(train_df_lf[label_col], errors="coerce").fillna(0.0)

    # LassoCV 只在训练集拟合
    reg = LassoCV(cv=5, random_state=config.seed, n_jobs=-1)
    reg.fit(X_train, y_train)

    selected_cols = [col for col, coef in zip(non_zero_cols, reg.coef_) if coef != 0]

    # 如果一个都没选出来，至少保留原始非零列，避免空输入
    if not selected_cols:
        selected_cols = non_zero_cols

    train_keep_cols = [label_col] + selected_cols
    val_keep_cols = [label_col] + selected_cols

    return train_df_lf[train_keep_cols].copy(), val_df_lf[val_keep_cols].copy()


for key, dataset_options in config.datasets[args.dataset].items():

    resolver = DatasetResolver()

    dataset_name = os.path.join(config.directories["datasets"], LANG, "train.csv")
    print(dataset_name)

    dataset = resolver.get(dataset_name, dataset_options, args.force)

    task_type = "classification" if args.task in ["1a", "1c"] else "regression"

    # -------------------------
    # 1) Load text dataframe
    # -------------------------
    df_embeddings = dataset.get()

    if "is_test" in df_embeddings.columns:
        df_embeddings = df_embeddings.loc[df_embeddings["is_test"] != True].copy()

    df_embeddings = dataset.getDFFromTask(args.task, df_embeddings).copy()
    df_embeddings = df_embeddings.reset_index(drop=True)

    # 文本预处理
    df_embeddings[TEXT_COL] = preprocess.remove_urls(df_embeddings[TEXT_COL])
    df_embeddings[TEXT_COL] = preprocess.remove_mentions(df_embeddings[TEXT_COL])
    df_embeddings[TEXT_COL] = preprocess.remove_whitespaces(df_embeddings[TEXT_COL])
    df_embeddings[TEXT_COL] = preprocess.expand_acronyms(
        df_embeddings[TEXT_COL],
        preprocess.EN_ACRONYMS
    )

    # -------------------------
    # 2) Load LF dataframe
    # -------------------------
    dataset_path = os.path.join(
        config.directories["assets"],
        args.dataset,
        key,
        "lf.csv"
    )

    df_lf = pd.read_csv(dataset_path)
    df_lf = df_lf.reset_index(drop=True)

    # 如果 LF 里没有 label，就从文本数据同步
    if LABEL_COL not in df_lf.columns:
        df_lf[LABEL_COL] = df_embeddings[LABEL_COL].values

    # 对齐长度，防止潜在错位
    min_len = min(len(df_embeddings), len(df_lf))
    df_embeddings = df_embeddings.iloc[:min_len].copy().reset_index(drop=True)
    df_lf = df_lf.iloc[:min_len].copy().reset_index(drop=True)

    # -------------------------
    # 3) Encode labels if classification
    # -------------------------
    lb = None

    if task_type == "classification":
        lb = sklearn.preprocessing.LabelBinarizer()
        lb.fit(df_embeddings[LABEL_COL].unique())

        if len(lb.classes_) > 2:
            df_labels = pd.DataFrame(
                lb.transform(df_embeddings[LABEL_COL]),
                columns=lb.classes_
            )
            df_embeddings = pd.concat([df_embeddings, df_labels], axis=1)
            df_lf = pd.concat([df_lf, df_labels], axis=1)
        else:
            df_embeddings[LABEL_COL] = df_embeddings[LABEL_COL].astype("category").cat.codes
            df_lf[LABEL_COL] = df_lf[LABEL_COL].astype("category").cat.codes

    if task_type == "regression":
        df_embeddings[LABEL_COL] = pd.to_numeric(df_embeddings[LABEL_COL], errors="coerce")
        df_lf[LABEL_COL] = pd.to_numeric(df_lf[LABEL_COL], errors="coerce")

    # -------------------------
    # 4) One split only
    # -------------------------
    all_indices = np.arange(len(df_embeddings))

    train_idx, val_idx = sklearn.model_selection.train_test_split(
        all_indices,
        train_size=dataset_options["train_size"],
        random_state=config.seed,
        stratify=df_embeddings[LABEL_COL] if task_type == "classification" else None
    )

    train_df_embeddings = df_embeddings.iloc[train_idx].copy().reset_index(drop=True)
    val_df_embeddings = df_embeddings.iloc[val_idx].copy().reset_index(drop=True)

    train_df_lf = df_lf.iloc[train_idx].copy().reset_index(drop=True)
    val_df_lf = df_lf.iloc[val_idx].copy().reset_index(drop=True)

    # -------------------------
    # 5) LF feature selection on training only
    # -------------------------
    train_df_lf, val_df_lf = safe_select_lf_features(
        train_df_lf=train_df_lf,
        val_df_lf=val_df_lf,
        label_col=LABEL_COL,
        task_type=task_type
    )

    # -------------------------
    # 6) Tokenizer: fit on train only
    # -------------------------
    tokenizer = keras.preprocessing.text.Tokenizer(
        num_words=max_words,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(train_df_embeddings[TEXT_COL])

    token_filename = os.path.join(
        config.directories["assets"],
        "keras",
        args.dataset,
        key,
        "tokenizer.pickle"
    )
    os.makedirs(os.path.dirname(token_filename), exist_ok=True)
    with open(token_filename, "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for dataframe in [train_df_embeddings, val_df_embeddings]:
        dataframe["tokens"] = tokenizer.texts_to_sequences(dataframe[TEXT_COL])

    # 用训练集长度分位数，而不是最大值
    train_lengths = [len(x) for x in train_df_embeddings["tokens"]]
    maxlen = int(np.percentile(train_lengths, 95))
    maxlen = max(maxlen, 5)

    tokens = []
    for dataframe in [train_df_embeddings, val_df_embeddings]:
        tokens.append(
            keras.preprocessing.sequence.pad_sequences(
                dataframe["tokens"],
                padding="pre",
                truncating="pre",
                maxlen=maxlen
            )
        )

    # -------------------------
    # 7) Classes / output dim
    # -------------------------
    if task_type == "regression":
        number_of_classes = 1
    elif len(lb.classes_) == 2:
        number_of_classes = 1
    else:
        number_of_classes = len(lb.classes_)

    # -------------------------
    # 8) Optimizers and metric
    # -------------------------
    optimizers = [keras.optimizers.Adam]
    if task_type == "regression":
        optimizers.append(keras.optimizers.RMSprop)

    # 这个前提是 kerasmodel.create 里 metric 名真叫 rmse
    reduction_metric = "val_loss" if task_type == "classification" else "val_rmse"

    # -------------------------
    # 9) Search space
    # -------------------------
    parameters_to_evaluate = {
        "task_type": [task_type],
        "tokenizer": [tokenizer],
        "name": [key],
        "dataset": [args.dataset],
        "number_of_classes": [number_of_classes],
        "epochs": [100],

        # 改正常学习率范围
        "lr": [1e-4, 3e-4],

        "optimizer": [keras.optimizers.Adam],
        "trainable": [True],
        "number_of_layers": [2],
        "first_neuron": [64],
        "shape": ["brick"],
        "batch_size": [16, 32],
        "dropout": [0.0, 0.2],
        "kernel_size": [3, 5],
        "maxlen": [maxlen],
        "we_architecture": ["cnn"],
        "activation": ["relu"],
        "pretrained_embeddings": ["fasttext_english"],
        "features": [ "we", "lf+we"],
        "patience": [args.patience]
    }

    # -------------------------
    # 10) Labels
    # -------------------------
    if task_type == "classification" and len(lb.classes_) > 2:
        y = tensorflow.convert_to_tensor(train_df_embeddings[lb.classes_].values)
        y_val = tensorflow.convert_to_tensor(val_df_embeddings[lb.classes_].values)
    else:
        y = tensorflow.convert_to_tensor(train_df_embeddings[LABEL_COL].values)
        y_val = tensorflow.convert_to_tensor(val_df_embeddings[LABEL_COL].values)

    # -------------------------
    # 11) Inputs
    # -------------------------
    if task_type == "classification" and len(lb.classes_) > 2:
        lf_exclude = [LABEL_COL, *lb.classes_]
    else:
        lf_exclude = [LABEL_COL]

    x = [
        tokens[0],
        train_df_lf.loc[:, ~train_df_lf.columns.isin(lf_exclude)]
    ]
    x_val = [
        tokens[1],
        val_df_lf.loc[:, ~val_df_lf.columns.isin(lf_exclude)]
    ]

    # -------------------------
    # 12) Time limit
    # -------------------------
    time_limit = (
        datetime.datetime.now() + datetime.timedelta(minutes=args.minutes)
    ).strftime("%Y-%m-%d %H:%M")

    # -------------------------
    # 13) Talos scan
    # -------------------------
    scan_object = talos.Scan(
        x=x,
        x_val=x_val,
        y=y,
        y_val=y_val,
        params=parameters_to_evaluate,
        model=kerasmodel.create,
        experiment_name=args.dataset,
        time_limit=time_limit,
        reduction_metric=reduction_metric,
        minimize_loss=True,
        print_params=True,
        round_limit=args.permutations,
        save_weights=False,
        seed=config.seed,
        multi_input=True
    )

    # -------------------------
    # 14) Save scan results
    # -------------------------
    for feature in ["shape", "we_architecture", "activation", "pretrained_embeddings"]:
        if feature in scan_object.data.columns:
            scan_object.data = kerasutils.pd_onehot(scan_object.data, feature)

    params_filename = os.path.join(
        config.directories["assets"],
        "keras",
        args.dataset,
        key,
        "hyperparameters-task-" + args.task + ".csv"
    )
    os.makedirs(os.path.dirname(params_filename), exist_ok=True)
    scan_object.data.to_csv(params_filename, index=False)