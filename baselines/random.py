# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from utils import cfg, load_data, get_labels, get_hypothesis, tokenize, clean_str

# %%
class RandomModel():
    def __init__(self) -> None:
        pass 

    def predict(self, X):
        return np.random.randint(2, size=len(X))

# %%
def clean_data(data: dict) -> None:
    for i in range(len(data['documents'])):
        data['documents'][i]['text'] = clean_str(data['documents'][i]['text'])

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def get_Ypred_Ytrue(data: dict, hypothesis: dict) -> (list, list):

    Y_pred = []
    Y_true = []

    model = RandomModel()

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

                if j in spans_for_hypothesis:
                    Y_true_curdoc.append(1)
                else:
                    Y_true_curdoc.append(0)
        
        Y_pred.append(model.predict(Y_true_curdoc))
        Y_true.append(Y_true_curdoc)

    return Y_pred, Y_true


# %%
train = load_data(cfg['train_path'])
clean_data(train)
hypothesis = get_hypothesis(train)
labels = get_labels()

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
Y_pred_test, Y_true_test = get_Ypred_Ytrue(test, hypothesis)

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

print("Precision @ 80% recall: ", np.mean(np.array(prec_arr)))
print("Mean Average Precision: ", mean_average_precision(Y_pred_test, Y_true_test))

# %%



