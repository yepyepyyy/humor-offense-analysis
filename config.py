import sys
import os
import  random
import numpy as np
import pandas as pd

from pathlib import Path

seed=0

# Configure main libraries for reproductibility
# Remember to use this seed in another random functions
os.environ['PYTHONHASHSEED'] = str (seed)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
random.seed (seed)
np.random.seed (seed)

base_path=Path(os.path.realpath(__file__)).parent

directories = {
    'datasets': os.path.join (base_path, 'datasets'),
    'pretrained': os.path.join (base_path, 'embeddings', 'pretrained'),
    'sentence_embeddings': os.path.join (base_path, 'embeddings', 'sentences'),
    'assets': os.path.join (base_path, 'assets'),
    'cache': os.path.join (base_path, 'cache_dir'),
}

pretrained_models = {
    'fasttext_english': {
        'vectors': os.path.join(directories['pretrained'], 'cc.en.300.vec'),
    },

    'glove_english': {
        'vectors': os.path.join(directories['pretrained'], 'glove.6b.300d.txt'),
    },
    'fasttext_chinese': {
        'vectors': os.path.join(directories['pretrained'], 'cc.zh.300.vec'),
    }

}

shared_max_size=sys.maxsize

#先放着，一会换成对应的处理函数
# @var shared_preprocessing Array Used for preprocessing tweets by preserving
#                                 lettercase, mentions, emojis, ...
shared_preprocessing = [
    'lettercase',
    'mentions',
    'emojis',
    'punctuation',
    'digits',
    'msg_language',
    'misspellings',
    'elongation'
]


shared_preprocessing_zh = [
    "mentions",
    "emojis",
    "punctuation",
    "digits",
    "msg_language",
    "elongation",
]
'''# @var shared_umutextstats_preprocessing Array Used for extracting LF
shared_umutextstats_preprocessing = [
    "lettercase",
    "hyperlinks",
    "mentions",
    "emojis",
    "punctuation",
    "digits",
    "msg_language",
    "misspellings",
    "elongation",
    "preserve_multiple_spaces",
    "preserve_blank_lines"
]

# @var shared_postagging_preprocessing Array Used for extracting PoS
shared_postagging_preprocessing = [
    "lettercase",
    "punctuation"
]'''

# Train ratio (float or int, default=None)
shared_train_size = 0.8

# Validation ratio. This value will be used to split the train split again, achieving training 60 testing 20 eval 20
shared_train_val_size = 0.75

datasets = {
  "hahackathon_en": {
    "base": {
      "language": "en",
      "datasetClass": "datasetHahackathon",
      "max": shared_max_size,
      "train_size": 0.8,
      "val_size": 0.2,
      "preprocessing": shared_preprocessing,
      "fields": ' ',
      "label_key": "user",
      "bert_model": "bert-base-uncased",   # 可选：给训练脚本用
    }
  },

  "hahackathon_zh": {
    "base": {
      "language": "zh",
      "datasetClass": "datasetHahackathon",  # 你新增一个类（或同类参数化）
      "max": shared_max_size,
      "train_size": 0.8,
      "val_size": 0.2,
      "preprocessing": shared_preprocessing_zh,
      "fields": '',
      "label_key": "user",
      "bert_model": "bert-base-chinese",  # 可选
    }
  }
}