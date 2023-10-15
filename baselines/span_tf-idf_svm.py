# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from utils import cfg, load_data, get_labels, get_hypothesis, tokenize, clean_str

# %%
def clean_data(data: dict) -> None:
    for i in range(len(data['documents'])):
        data['documents'][i]['text'] = clean_str(data['documents'][i]['text'])
        data['documents'][i]['text'] = tokenize(data['documents'][i]['text'])

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import scipy.sparse as sp
import nltk
def get_XY(data: dict, tfidf: TfidfVectorizer, hypothesis: dict, labels: dict, n_docs : int, threshold : float = 0.1) -> (list, list):

    X = []
    Y = []

    hypothesis_vecs = {}
    for key, val in hypothesis.items():
        hypothesis_vecs[key] = tfidf.transform([val])

    for i in tqdm(range(min(n_docs, len(data["documents"])))):
        doc_text = data["documents"][i]["text"]

        for key, val in hypothesis.items():
            choice = data["documents"][i]["annotation_sets"][0]["annotations"][key]["choice"]
            if choice == "NotMentioned":
                continue

            spans_for_hypothesis = data["documents"][i]["annotation_sets"][0]["annotations"][key]["spans"]

            for j, span in enumerate(data["documents"][i]["spans"]):
                start_idx = span[0]
                end_idx = span[1]

                span_text = doc_text[start_idx:end_idx]
                span_vector = tfidf.transform([span_text])

                input_vec = sp.hstack([span_vector, hypothesis_vecs[key]])
                # return X, Y
                X += [input_vec]
                Y += [1 if j in spans_for_hypothesis else 0]
        
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
X_train, Y_train = get_XY(train, tfidf, hypothesis, labels=labels, n_docs=10)

# %%
from sklearn.svm import SVC

model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

# %%
test = load_data(cfg['test_path'])
clean_data(test)
X_test, Y_test = get_XY(test, tfidf, hypothesis, labels=labels, n_docs=10)

# %%
Y_pred = model.predict(X_test)

# %%
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score

def precision_at_80_recall(ypred, ytrue):
    precision, recall, thresholds = precision_recall_curve(ytrue, ypred)
    idx = (abs(recall - 0.8)).argmin()
    return precision[idx]

# %%
from sklearn.metrics import average_precision_score
def mean_average_precision(Y_pred, Y_true):
    average_prec_scores = []
    for i in range(len(Y_true)):
        average_prec_scores.append(average_precision_score(Y_true[i], Y_pred[i], average='micro'))
    return np.mean(average_prec_scores)

# %%
all_y_pred_test = np.concatenate(Y_pred)
all_y_true_test = np.concatenate(Y_test)

# %%
prec_arr = []
for i in range(len(Y_test)):
    prec_arr.append(precision_at_80_recall(Y_pred[i], Y_test[i]))

print("Precision @ 80\% recall: ", np.mean(np.array(prec_arr)))
print("Mean Average Precision: ", mean_average_precision(Y_pred, Y_test))


