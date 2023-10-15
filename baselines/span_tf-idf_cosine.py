# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from utils import cfg, load_data, get_labels, get_hypothesis, tokenize, clean_str

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
def clean_data(data: dict) -> None:
    for i in range(len(data['documents'])):
        data['documents'][i]['text'] = clean_str(data['documents'][i]['text'])

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def get_Ypred_Ytrue(data: dict, tfidf: TfidfVectorizer, hypothesis: dict) -> (list, list):

    Y_pred = []
    Y_true = []

    hypothesis_vecs = {}
    for key, val in hypothesis.items():
        hypothesis_vecs[key] = tfidf.transform([val])

    for i in tqdm(range(len(data["documents"]))):
        doc_text = data["documents"][i]["text"]

        Y_pred_curdoc = []
        Y_true_curdoc = []

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

                cosine_sim = cosine_similarity(span_vector, hypothesis_vecs[key])[0][0]
                Y_pred_curdoc.append(cosine_sim)

                if j in spans_for_hypothesis:
                    Y_true_curdoc.append(1)
                else:
                    Y_true_curdoc.append(0)
        
        Y_pred.append(Y_pred_curdoc)
        Y_true.append(Y_true_curdoc)

    return Y_pred, Y_true


# %%
def get_total_no_of_spans(data: dict) -> int:
    total_spans = 0
    for i in range(len(data["documents"])):
        total_spans += len(data["documents"][i]["spans"])
    return total_spans

# %%
train = load_data(cfg['train_path'])
val = load_data(cfg['dev_path'])
test = load_data(cfg['test_path'])
print(len(train['documents']), len(val['documents']), len(test['documents']))
print(get_total_no_of_spans(train), get_total_no_of_spans(val), get_total_no_of_spans(test))
# clean_data(train)
# hypothesis = get_hypothesis(train)
# labels = get_labels()

# %%
all_text = []

for i in range(len(train["documents"])):
    all_text.append(train["documents"][i]["text"])

tfidf = TfidfVectorizer()
tfidf.fit(all_text)

# %%
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score

def precision_at_80_recall(ypred, ytrue):
    precision, recall, thresholds = precision_recall_curve(ytrue, ypred)
    idx = (abs(recall - 0.8)).argmin()
    return precision[idx]

# %%
test = load_data(cfg['test_path'])

# %%
Y_pred_test, Y_true_test = get_Ypred_Ytrue(test, tfidf, hypothesis, labels, threshold=0.4)

# %%
from sklearn.metrics import average_precision_score
def mean_average_precision(Y_pred, Y_true):
    average_prec_scores = []
    for i in range(len(Y_true)):
        average_prec_scores.append(average_precision_score(Y_true[i], Y_pred[i], average='micro'))
    return np.mean(average_prec_scores)

# %%
all_y_pred_test = np.concatenate(Y_pred_test)
all_y_true_test = np.concatenate(Y_true_test)

# %%
prec_arr = []
for i in range(len(Y_true_test)):
    prec_arr.append(precision_at_80_recall(Y_pred_test[i], Y_true_test[i]))

print("Precision @ 80\% recall: ", np.mean(np.array(prec_arr)))
print("Mean Average Precision: ", mean_average_precision(Y_pred_test, Y_true_test))


