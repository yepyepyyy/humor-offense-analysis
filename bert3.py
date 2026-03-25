# Import libraries
import argparse
import config
import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
import sklearn
import random
import pickle
from types import SimpleNamespace
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
#from tensorflow import keras
from dataset import DatasetResolver
from process import PreProcessText
from sklearn import linear_model
from torch import nn


TEXT_COL = "text"
Langue="English"

#pretrained_model = '.\models\\bert-base-uncased'
pretrained_model='.\\assets\hahackathon_en\\base\\bert-finetune-1a'

# Parser
args = SimpleNamespace(
    dataset="hahackathon_en",
    label="label",
    force=False,
    task="1a",
    evaluate=True,
    methods = ["bert"]  # 当前只跑 BERT；以后可改成 ["keras"] 或 ["random", "bert", "keras"]
)

batch_size = 64

# Get device
device = torch.device ('cuda') if torch.cuda.is_available () else torch.device ('cpu')
print(device)

torch.manual_seed (config.seed)

# Preprocess text
preprocess = PreProcessText()
Acronyn_Map = preprocess.EN_ACRONYMS
models = []


bert_tokenizer = BertTokenizerFast.from_pretrained (pretrained_model)
def tokenize (batch):
    """
    Bert Tokenizer
    """
    return bert_tokenizer (batch[TEXT_COL], padding = True, truncation = True)

class CustomBERTModel(nn.Module):
    """
    BERT + linguistic features
    """

    def __init__(self, input_size, num_classes, pretrained_model_path, task_type):
        super(CustomBERTModel, self).__init__()

        self.task_type = task_type

        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_model_path,
            return_dict=True,
            output_hidden_states=True
        )

        self.fc1 = nn.Linear(input_size + 768, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, lf, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None):
        with torch.no_grad():
            sequence_output = self.bert(input_ids, attention_mask=attention_mask)

        hidden_states = sequence_output.hidden_states
        cls_tokens = hidden_states[-1][:, 0]

        combined_with_bert_features = torch.cat((cls_tokens, lf), dim=1)

        x = F.relu(self.fc1(combined_with_bert_features))
        x = F.relu(self.fc2(x))

        if self.task_type == "classification":
            x = torch.sigmoid(self.fc3(x))
        else:
            x = self.fc3(x)

        return x


def randomMethod(rows):
    """
    Generate random values for each column. Used as a baseline
    """

    return pd.DataFrame({
        'id': list(range(1, rows + 1)),
        'is_humor': [random.randint(0, 1) for dump in range(rows)],
        'humor_rating': [random.uniform(0.0, 5.0) for dump in range(rows)],
        'humor_controversy': [random.randint(0, 1) for dump in range(rows)],
        'offense_rating': [random.uniform(0.0, 5.0) for dump in range(rows)]
    })


def kerasMethod(x, models):
    """
    keras Method evaluation
    """

    # Get predictions for the task 1A
    y_prob = models['1a'].predict(x)
    y = y_prob > 0.5
    y = y.flatten()
    y = [int(item) for item in y]

    # Shame on me
    y = [1 - item for item in y]
    is_humor = pd.Series(y)

    # Get predictions for the task 1B
    y = models['1b'].predict(x)
    y = y.flatten()
    y = [max(item, 0) for item in y]
    humor_rating = pd.Series(y)

    # Get predictions for the task 1C
    y_prob = models['1c'].predict(x)
    y = np.argmax(y_prob, axis=1)

    # As labels are 'humor', 'non-humor', 'offensive', and we need an binary
    # output, we collapse non-humor and humor
    # First step, transform (non-humor, 1) into (humor, 0)
    y = [0 if item == 1 else item for item in y]

    # Second step, transform (offensive, 2) into (offensive, 1)
    y = [1 if item == 2 else 0 for item in y]

    # Shame on me
    y = [1 - item for item in y]

    # Get column
    humor_controversy = pd.Series(y)

    # Get predictions for the task 1B
    y = models['2a'].predict(x)
    y = y.flatten()
    y = [max(item, 0) for item in y]
    offense_rating = pd.Series(y)

    return pd.DataFrame({
        'id': x[1].index + 1,
        'is_humor': is_humor,
        'humor_rating': humor_rating,
        'humor_controversy': humor_controversy,
        'offense_rating': offense_rating
    })


def BERTMethod(df, df_lf, models):
    """
    BERT Method
    """

    # Encode datasets to work with transformers
    dataset = Dataset.from_pandas(df)

    # Tokenizer trainset and test dataframe with the training
    # The tokenize function only takes care of the "tweet"
    # column and will create the input_ids, token_type_ids, and
    # attention_mask
    dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))

    # Finally, we "torch" the new columns. We return the rest
    # of the columns with "output_all_columns"
    #dataset.set_format('torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    '''    input_ids = torch.tensor(dataset['input_ids'])
        attention_mask = torch.tensor(dataset['attention_mask'])
        lf_tensor = torch.tensor(df_lf.values)
    '''

    input_ids = torch.tensor(dataset[:]['input_ids'], dtype=torch.long)
    attention_mask = torch.tensor(dataset[:]['attention_mask'], dtype=torch.long)
    lf_tensor = torch.tensor(df_lf.values, dtype=torch.float)

    dataset = data_utils.TensorDataset(
        lf_tensor,
        input_ids,
        attention_mask,
    )


    # Get the train loader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Columns
    is_humor = []
    humor_rating = []
    humor_controversy = []
    offense_rating = []

    with torch.no_grad():
        for i, (lf, input_ids, attention_mask) in enumerate(train_loader):

            # Move features to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            lf = lf.float().to(device)

            # Get predictions for task 1a
            predictions = models['1a'](lf, input_ids, attention_mask=attention_mask)
            predictions = torch.squeeze(predictions)

            # Use max to get the correct class
            _, preds = torch.max(predictions, dim=1)

            # Attach
            for i, prediction in enumerate(preds):
                is_humor.append(1 - prediction.item())

            # Get predictions for task 1b
            predictions = models['1b'](lf, input_ids, attention_mask=attention_mask)

            for i, prediction in enumerate(predictions):
                humor_rating.append(prediction.item())

            # Get predictions for task 1c
            predictions = models['1c'](lf, input_ids, attention_mask=attention_mask)
            predictions = torch.squeeze(predictions)

            # Use max to get the correct class
            _, preds = torch.max(predictions, dim=1)

            # Attach
            for i, prediction in enumerate(preds):
                # As labels are 'humor', 'non-humor', 'offensive', and we need an binary
                # output, we collapse non-humor and humor
                # First step, transform (non-humor, 1) into (humor, 0)
                prediction = 0 if prediction == 1 else prediction

                # Second step, transform (offensive, 2) into (offensive, 1)
                prediction = 1 if prediction == 2 else 0

                # Shame on me
                prediction = 1 - prediction

                # Reverse
                humor_controversy.append(prediction)

            # Get predictions for task 2a
            predictions = models['2a'](lf, input_ids, attention_mask=attention_mask)

            for i, prediction in enumerate(predictions):
                offense_rating.append(prediction.item())

    """
    print ("is_humor")
    print (pd.Series (is_humor).value_counts ())

    print ("humor_rating")
    print (pd.Series (humor_rating).value_counts (normalize = True))

    print ("humor_controversy")
    print (pd.Series (humor_controversy).value_counts ())

    print ("offense_rating")
    print (pd.Series (offense_rating).value_counts (normalize = True))
    sys.exit ();
    """

    return pd.DataFrame({
        'id': df.index + 1,
        'is_humor': is_humor,
        'humor_rating': humor_rating,
        'humor_controversy': humor_controversy,
        'offense_rating': offense_rating
    })


def main():
    """ To use from command line """



    # @var umucorpus_ids int|string The Corpus IDs
    for key, dataset_options in config.datasets[args.dataset].items():

        # Resolver
        resolver = DatasetResolver()

        # Get the dataset name
        #dataset_name = args.dataset + "-" + key + '.csv'
        dataset_path = os.path.join(config.directories["datasets"], Langue, "train.csv")
        print("[LOAD]", dataset_path)
        Bert_df = pd.read_csv(dataset_path).reset_index(drop=True)

        # Get the dataset
        dataset = resolver.get(dataset_path, dataset_options, args.force)

        # Get dataframe embeddings
        df_embeddings = dataset.get()

        # Get rid of training
        if "is_test" in df_embeddings.columns:
            df_embeddings_train = df_embeddings.drop(df_embeddings[df_embeddings["is_test"] == True].index)
            df_embeddings = df_embeddings.drop(df_embeddings[df_embeddings["is_test"] == False].index)
        else:
            df_embeddings_train = df_embeddings.copy()
        # Get linguistic features
        path1=os.path.join(config.directories['assets'], args.dataset, key, 'lf.csv')
        print(path1)
        df_lf = pd.read_csv(path1, header=0, sep=",")

        df_lf = df_lf.rename(columns={"class": "label"})

        # Get BERT LF
        df_lf_bert = df_lf.loc[:, (df_lf != 0).any(axis=0)]
        #df_lf_bert = df_lf_bert.iloc[:, :-1]

        # Get models
        my_keras_models = {}
        my_bert_models = {}

        # We get the train split to replicate Lasso
        df_lf_train = df_lf[df_lf.index.isin(df_embeddings_train.index)]

        # Perform feature selection over the LF
        reg = sklearn.linear_model.LassoCV()

        # 先给原始 df 生成当前任务的 label
        if hasattr(dataset, "getDFFromTask"):
            df_task = dataset.getDFFromTask(args.task, df_embeddings_train.copy())
        else:
            raise AttributeError("dataset 没有 getDFFromTask，无法生成当前任务 label")

        # 再把 label 拼回 LF 训练子集
        df_lf_train = df_lf[df_lf.index.isin(df_embeddings_train.index)].copy()
        df_lf_train["label"] = df_task["label"].values

        # Get all features that are not the label
        X = df_lf_train.loc[:, ~df_lf_train.columns.isin(['label'])]

        # Encode label
        # @todo. Error with multiclass
        y = df_lf_train['label'].astype('category').cat.codes

        # Fit LassoCV
        reg.fit(X, y)

        # Get Lasso coefficients
        coef = pd.Series(reg.coef_, index=X.columns)

        # Determine which LF does not fit the coef
        lf_columns = [column for column, value in coef.items() if value != 0]

        # Filter those LF
        df_lf = df_lf[lf_columns]

        # Retrieve only the linguistic features that are part of the training
        df_lf = df_lf.loc[df_embeddings.index]
        df_lf_bert = df_lf_bert.loc[df_embeddings.index]

        # Preprocess. First, some basic stuff
        for pipe in ['remove_urls', 'remove_digits', 'remove_whitespaces', 'remove_elongations', 'to_lower',
                     'remove_punctuation']:
            df_embeddings[TEXT_COL] = getattr(preprocess, pipe)(df_embeddings[TEXT_COL])

        # Then, expand contractions
        df_embeddings[TEXT_COL] = preprocess.expand_acronyms(df_embeddings[TEXT_COL], Acronyn_Map)

        for task in ['1a', '1b', '1c', '2a']:
            model_filename = os.path.join(config.directories['assets'], args.dataset, key, 'model-task-' + task + '.h5')
            #my_keras_models[task] = keras.models.load_model(model_filename, compile=False)

        for task in ['1a', '1b', '1c', '2a']:
            # Get the task type
            task_type = 'classification' if task in ['1a', '1c'] else 'regression'

            # Get the number of classes
            num_classes = 1 if task_type == 'regression' else (2 if task == '1a' else 3)

            # @var pretrained_model_filename String
            pretrained_model_filename = os.path.join(config.directories['assets'], args.dataset, key,
                                                     'bert-finetune-' + task)

            # Create the mode
            my_bert_models[task] = CustomBERTModel(df_lf_bert.shape[1], num_classes, pretrained_model_filename,
                                                   task_type).to(device)

            # Get the model information
            model_dict = os.path.join(config.directories['assets'], args.dataset, key, 'bert-final-' + task + '.pt')

            # Load model and put in eval mode
            my_bert_models[task].load_state_dict(torch.load(model_dict,map_location=device))
            my_bert_models[task].eval()


            '''        # @var Tokenizer Retrieve the tokenizer from disk
        token_filename = os.path.join(config.directories['assets'], args.dataset, key, 'tokenizer.pickle')
        with open(token_filename, 'rb') as handle:
            tokenizer = pickle.load(handle)'''

            # 这里只在需要时才加载 keras tokenizer / 生成 tokens
            tokens = []

            for method in args.methods:

                if method in ['random', 'keras']:
                    token_filename = os.path.join(
                        config.directories['assets'], args.dataset, key, 'tokenizer.pickle'
                    )

                    if not os.path.exists(token_filename):
                        raise FileNotFoundError(
                            f"Keras tokenizer not found: {token_filename}"
                        )

                    with open(token_filename, 'rb') as handle:
                        keras_tokenizer = pickle.load(handle)



                    # Update to tokens
                    for dataframe in [df_embeddings]:
                        dataframe['tokens'] = keras_tokenizer.texts_to_sequences(dataframe[TEXT_COL])

                    # Get the max-len size
                    maxlen = max(len(l) for l in df_embeddings['tokens'])
                    maxlen = 62  # fix

                    # Transform sentences to tokens
                    tokens = []
                    #for dataframe in [df_embeddings]:
                       # tokens.append(keras.preprocessing.sequence.pad_sequences(dataframe['tokens'], padding='pre', maxlen=maxlen))

        # Encode labels for classification task
        # For each method
        for method in args.methods:

            if (method == 'random'):
                output_df = randomMethod(tokens[0].shape[0])

            elif (method == 'keras'):
                if len(tokens) == 0:
                    raise RuntimeError("Keras tokens 尚未生成，请恢复 pad_sequences 相关代码后再运行 keras 方法。")


                # Get features
                x = [tokens[0], df_lf.loc[:, ~df_lf.columns.isin([*['label']])]]

                output_df = kerasMethod(x, my_keras_models)

            elif (method == 'bert'):
                output_df = BERTMethod(df_embeddings, df_lf_bert, my_bert_models)

            # Create dataset
            #print(output_df)

            #output_df.to_csv(os.path.join(config.directories['assets'], args.dataset, key, 'output-' + method + '.csv'),index=False, float_format="%.3f")
            print(output_df.head())

            save_path = os.path.join(
                config.directories["assets"],
                args.dataset,
                key,
                f"output-{method}.csv"
            )
            output_df.to_csv(save_path, index=False, float_format="%.3f")
            print(f"[OK] saved: {save_path}")


if __name__ == "__main__":
    main()