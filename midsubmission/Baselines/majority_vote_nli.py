# %%
import pandas as pd
import json
from utils import load_data, get_labels, cfg

# %%
def get_hypothesis(data: dict) -> list:
    hypothesis = {}
    majority_ct = {} # storing the counts
    for key, value in data['labels'].items():
        hypothesis[key] = value['hypothesis']
        majority_ct[key] = {'NotMentioned': 0, 'Entailment': 0, 'Contradiction': 0}
    return hypothesis, majority_ct

# %%
train = load_data(cfg['train_path'])
hypothesis, majority_ct = get_hypothesis(train)
labels = get_labels()

# %%
for doc in train['documents']:
    for key, value in doc['annotation_sets'][0]['annotations'].items():
        majority_ct[key][value['choice']] += 1

for key, value in majority_ct.items():
    majority_ct[key] = max(value, key=value.get)

# %%
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

wandb.init(entity="contract-nli-db", project="majority_vote")
config = wandb.config

# %%
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# dev = load_data(cfg['dev_path'])
test = load_data(cfg['test_path'])

Y_true = []
Y_pred = []

for doc in test['documents']:
    for key, value in doc['annotation_sets'][0]['annotations'].items():
        Y_true.append(labels[value['choice']])
        Y_pred.append(labels[majority_ct[key]])

report = classification_report(Y_true, Y_pred, output_dict=True)
wandb.log(report)

cm = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('majority_vote.png')
wandb.log({'confusion_matrix': wandb.Image('majority_vote.png')})

# %%
wandb.finish()
os.remove('majority_vote.png')

# %%



