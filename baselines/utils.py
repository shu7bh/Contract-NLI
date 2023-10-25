import os
import json
from nltk import word_tokenize 
import re

cfg = {
    'raw_data_dir': '../dataset/',
    'train_path': 'train.json',
    'test_path': 'test.json',
    'dev_path': 'dev.json',
    "model_name": "bert-base-uncased",
    "max_length": 512,
    "models_save_dir": "/scratch/shu7bh/contract_nli/models",
    "dataset_dir": "/scratch/shu7bh/contract_nli/dataset",
    "results_dir": "/scratch/shu7bh/contract_nli/results",
}

def load_data(path: str) -> json:
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def get_labels() -> dict:
    return {
        'NotMentioned': 0,
        'Entailment': 1,
        'Contradiction': 2,
        'Ignore': 3,
    }

def get_hypothesis(data: dict) -> list:
    hypothesis = {}
    for key, value in data['labels'].items():
        hypothesis[key] = clean_str(value['hypothesis'])
    return hypothesis

def tokenize(str: str) -> str:
    return ' '.join(word_tokenize(str))

def clean_str(str: str) -> str:
    # remove '\n' character
    str = str.replace('\n', ' ')
    # remove '\t' character
    str = re.sub(r'\\t', ' ', str)
    # remove '\r' character
    str = re.sub(r'\\r', ' ', str)
    # remove more than 2 consecutive occcurance of a character
    str = re.sub(r'(.)\1{2,}', r'\1', str)
    return str.strip().lower()