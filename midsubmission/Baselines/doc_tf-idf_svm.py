# %%
import pandas as pd
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import cfg, load_data, get_labels, get_hypothesis, clean_str, tokenize

# %%
def clean_data(data: dict) -> None:
    for i in range(len(data['documents'])):
        data['documents'][i]['text'] = clean_str(data['documents'][i]['text'])
        data['documents'][i]['text'] = tokenize(data['documents'][i]['text'])

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
def get_XY(data: dict, tfidf: TfidfVectorizer, hypothesis: dict, labels: dict) -> (list, list):
    
    X = []
    Y = []
    for i in range(len(data["documents"])):

        premise = data["documents"][i]["text"]
        premise_vector = tfidf.transform([premise])
        
        for key, val in hypothesis.items():
            hypothesis_vector = tfidf.transform([val])

            input_vec = sp.hstack([premise_vector, hypothesis_vector])
            X += [input_vec]
            Y += [labels[data["documents"][i]["annotation_sets"][0]["annotations"][key]["choice"]]]
    
    return sp.vstack(X), Y

# %%
train = load_data(cfg['train_path'])
clean_data(train)
hypothesis = get_hypothesis(train)
labels = get_labels()

# %%
all_text = ""

for i in range(len(train["documents"])):
    all_text += train["documents"][i]["text"] + " "

tfidf = TfidfVectorizer()
tfidf.fit([all_text])

# %%
X_train, Y_train = get_XY(train, tfidf, hypothesis, labels=labels)

# %%
from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(X_train, Y_train)

# %%
# dev = load_data(cfg['dev_path'])
test = load_data(cfg['test_path'])
clean_data(test)
X_dev, Y_dev = get_XY(test, tfidf, hypothesis, labels=labels)

# %%
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

wandb.init(entity="contract-nli-db", project="doc_tf-if_svm")
config = wandb.config

config.kernel = 'linear'
config.train_size = len(train)
config.dev_size = len(X_dev)

wandb.log({'accuracy': model.score(X_dev, Y_dev)})

Y_pred = model.predict(X_dev)
report = classification_report(Y_dev, Y_pred, output_dict=True)
wandb.log(report)

# %%
cm = confusion_matrix(Y_dev, Y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('doc_tf-idf_svm.png')
wandb.log({'confusion_matrix': wandb.Image('doc_tf-idf_svm.png')})

# %%
wandb.finish()

# %%
# remove the png file
os.remove('doc_tf-idf_svm.png')


