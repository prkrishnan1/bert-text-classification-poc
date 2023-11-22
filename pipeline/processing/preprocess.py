# -*- coding: utf-8 -*-
import re
import json
import pandas as pd
import os
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

# Steps to clean and preprocess incident data
# 1. Read into Pandas DataFrame
# 2. Inspect service offerings to filter out null rows
# 3. Remove rows that are nulls for short_description AND description
# 4. Fill nulls
# 5. Remove added spaces from short description, and description columns
# 6. Combine text into one column: input_text
# 7. Filter dataset into following columns: incident_id, Created_time, input_text, service_offering

def clean_data(container_base_path, data_file_name):
    container_file_path = os.path.join(container_base_path, "data", data_file_name)
    
    df = pd.read_csv(container_file_path, encoding="cp1252")
    
    df.rename({"sys_created_on": "created_time", "number" : "incident_id", "assignment_group.u_su_business_unit.u_name" : "business_unit", "cmdb_ci" : "labels"}, axis=1, inplace=True)
    
    columns = ["created_time", "incident_id", "short_description", "description", "business_unit", "labels"]

    df = df[columns]
    df = df.groupby("labels").filter(lambda x: (x[columns[0]].count()>400).any())
    
    df = df[pd.notnull(df["labels"])]
    df = df[pd.notnull(df[columns[2]]) | pd.notnull(df[columns[3]])]
    
    df = df.fillna(value="None")
    
    df["short_description"] = df["short_description"].apply(lambda text: re.sub(' +', ' ', text))
    df["description"] = df["description"].apply(lambda text: re.sub(' +', ' ', text))
    
    return df

def preprocess_data(df: pd.DataFrame, train_ratio: float, valid_ratio: float, random_state: int=138):
    columns = df.columns
    df["input_text"] = df[columns[4]] + " | " + df[columns[2]] + " | " + df[columns[3]]
    le = LabelEncoder()
    le.fit(df["labels"])
    
    id2label = dict(zip(le.transform(le.classes_).astype(str), le.classes_))
    print("There are {} unique labels".format(len(list(le.classes_))))
    print("Here is the encoding to label mapping")
    print("Mapping: {}".format(id2label))
    
    df["labels"] = le.transform(df["labels"])
    
    
    valid_to_test_ratio = valid_ratio / (1 - train_ratio)
    
    train_df, _remaining_df = train_test_split(df, random_state = random_state, train_size = train_ratio)
    valid_df, test_df = train_test_split(_remaining_df, random_state = random_state, train_size = valid_to_test_ratio)
    
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    encoded_dataset_train = train_dataset.map(tokenize_data, batched=True, remove_columns=train_dataset.column_names)
    encoded_dataset_valid = valid_dataset.map(tokenize_data, batched=True, remove_columns=valid_dataset.column_names)
    encoded_dataset_test = test_dataset.map(tokenize_data, batched=True, remove_columns=test_dataset.column_names)
    
    return encoded_dataset_train, encoded_dataset_valid, encoded_dataset_test, id2label
    
def tokenize_data(examples):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # take a batch of texts
    text = examples["input_text"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    # add labels
    encoding["labels"] = examples["labels"]

    return encoding

def save_to_file(encoded_train_df, encoded_valid_df, encoded_test_df, id2label, container_base_path):
    
    training_input_path = os.path.join(container_base_path, "output", "train")
    encoded_train_df.save_to_disk(training_input_path)
    eval_input_path = os.path.join(container_base_path, "output", "validation")
    encoded_valid_df.save_to_disk(eval_input_path)
    test_input_path = os.path.join(container_base_path, "output", "test")
    encoded_test_df.save_to_disk(test_input_path)
    
    id2label_path = os.path.join(container_base_path, "output", "id2label")
    
    os.makedirs(id2label_path, exist_ok=True)
    
    with open(os.path.join(id2label_path, "id2label.json"), "w") as fp:
        json.dump(id2label, fp)
    
        
    print(f'Saved training data to {training_input_path}')
    print(f'Saved evaluation data to {eval_input_path}')
    print(f'Saved test data to {test_input_path}')
    print(f'Saved id2label mapping to {id2label_path}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--container-base-path", type=str)
    parser.add_argument("--data-file-name", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    
    args, _ = parser.parse_known_args()
    
    print("Received arguments {}".format(args))
    
    df = clean_data(args.container_base_path, args.data_file_name)
    encoded_dataset_train, encoded_dataset_valid, encoded_dataset_test, id2label = preprocess_data(df, args.train_ratio, args.val_ratio)
    save_to_file(encoded_dataset_train, encoded_dataset_valid, encoded_dataset_test, id2label, args.container_base_path)
    
