import pandas as pd

from process import run_pipeline
import config
import requests
import sys
import csv
import os
import argparse
import string

import fasttext

class Dataset:
    """
    Load datasets from local CSV files (train/test/dev).
    Produces a unified dataframe with:
      - split: train/test/dev
      - text column normalized to 'tweet'
      - label column normalized to 'label' (if exists)
    """

    def __init__(self, dataset, options, refresh=False):
        self.dataset = dataset
        self.options = options
        self.refresh = refresh
        self.df = None

    '''    def get(self):
        """
        Retrieve the Pandas Dataframe

        @return dataframe
        """
        if self.df is None:
            # filename
            filename = os.path.join(config.directories['datasets'], self.dataset)
            # Get the dataset as a dataframe
            if not self.refresh and os.path.isfile(filename):
                self.df = pd.read_csv(os.path.join(config.directories['datasets'], self.dataset), header=0, sep=",")
            else:
                self.df = self.compile()


        return self.df'''

    def get(self):
        """
        Retrieve the Pandas Dataframe

        @return dataframe
        """
        if self.df is None:
            filename = os.path.join(config.directories['datasets'], self.dataset)

            if not os.path.isfile(filename):
                raise FileNotFoundError(
                    f"找不到数据集文件：{filename}\n"
                    f"当前版本不会自动 compile，请先确认该 CSV 文件已经放在 datasets 目录下。"
                )

            self.df = pd.read_csv(filename, header=0, sep=",")

        return self.df


class DatasetHahackathon (Dataset):
    def __init__ (self, dataset, options, refresh = False):
        """
        @inherit
        """
        Dataset.__init__ (self, dataset, options, refresh)

    def getDFFromTask(self, task, df):
        """
        Task 1 emulates previous humor detection tasks in which all ratings were averaged to provide mean classification and rating scores.
        Task 2 aims to predict how offensive a text would be (for an average user) with values between 0 and 5.
        """

        # Task 1a: predict if the text would be considered humorous (for an average user). This is a binary task.
        # -------------------------------------------------------------
        # We replace 0 with non-humor and 1 with humor
        if (task == '1a'):
            df['label'] = df['is_humor'].replace([0, 1], ['non-humor', 'humor'])
            return df

        # Task 1b: if the text is classed as humorous, predict how humorous it is (for an average user).
        # The values vary between 0 and 5.
        # -------------------------------------------------------------
        # We move the humor_rating column to the label and fill NaN with zeros
        if (task == '1b'):
            df['label'] = df['humor_rating']
            df['label'] = df['label'].fillna(0.0)
            return df;

        # Task 1c: if the text is classed as humorous, predict if the humor rating would be considered controversial,
        #  i.e. the variance of the rating between annotators is higher than the median. This is a binary task.
        # -------------------------------------------------------------
        # We move the humor_controversy column to the label
        if (task == '1c'):
            # It is possible to transform this problem into multiclass, to do it, just...
            df['label'] = df['is_humor'].replace([0, 1], ['non-humor', 'humor'])
            df.loc[df['humor_controversy'] == 1, 'label'] = 'offensive'

            # It is possible to transform this problem into binary; however
            # work has to be done to prevent the imbalance
            """
            df['label'] = df['humor_controversy'].replace ([0, 1], ['non-controversy', 'controversy'])
            df['label'] = df['label'].fillna ("non-controversy")
            """
            return df

        # Task 2a: predict how generally offensive a text is for users.
        # This score was calculated regardless of whether the text is classed as humorous or offensive overall.
        # -------------------------------------------------------------
        # Similar to 1b, we move the desired column to the labels, and then
        # fill the missing values
        if (task == '2a'):
            df['label'] = df['offense_rating']
            df['label'] = df['label'].fillna(0.0)
            return df


class DatasetResolver():
    """
    DatasetResolver
    """

    def get(self, dataset, options, refresh=False):

        # Default
        if not 'datasetClass' in options:
            return Dataset(dataset, options, refresh)

        # Super
        if (options['datasetClass'] == 'datasetHahackathon'):
            return DatasetHahackathon(dataset, options, refresh)

        else:
            return Dataset(dataset, options, refresh)



