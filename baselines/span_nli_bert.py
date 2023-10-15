# %%
import numpy as np
from utils import cfg, load_data, get_labels, get_hypothesis, tokenize, clean_str

# %%
def Dynamic_Context_Segmentation(B, T, n, l):
    contexts = []
    start = 0

    while len(B) > 0:
        for bi in list(B):
            if bi - start <= l:
                B.remove(bi - 1)
                end = bi - 1
        contexts.append(T[start : (start + l)])
        start = end - n

    return contexts

# %%
train = load_data(cfg['train_path'])

# %%
doc_spans = []
for doc in train["documents"]:
    cur_doc_spans = []
    for span in doc["spans"]:
        cur_doc_spans.append(span)
    doc_spans.append(cur_doc_spans)

# %%
contexts = Dynamic_Context_Segmentation(doc_spans[0], train["documents"][0]["text"], 2, 512)

# %%



