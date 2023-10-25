# %%
from transformers import AutoTokenizer, AutoModelForMaskedLM
import logging as log
log.basicConfig(level=log.DEBUG)

# %%
import sys
sys.path.append('../')
from baselines.utils import *
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# %%
import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE

# %%
cfg['model_name'] = 'bert-base-uncased'
cfg['batch_size'] = 32
cfg

# %%
# create dir if not exists
from pathlib import Path
Path(cfg["models_save_dir"]).mkdir(parents=True, exist_ok=True)
Path(cfg["dataset_dir"]).mkdir(parents=True, exist_ok=True)

# %%
tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
bert = AutoModelForMaskedLM.from_pretrained(cfg['model_name'])

tokenizer.save_pretrained(cfg['models_save_dir'])
bert.save_pretrained(cfg['models_save_dir'])

# %%
tokenizer = AutoTokenizer.from_pretrained(cfg['models_save_dir'])
bert = AutoModelForMaskedLM.from_pretrained(cfg['models_save_dir'])

# %%
from icecream import ic

# %%
from torch.utils.data import Dataset
import torch

class NLIDataset(Dataset):
    def __init__(self, documents, tokenizer, hypothesis, context_sizes, surround_character_size):
        label_dict = get_labels()
        self.tokenizer = tokenizer

        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[SPAN]']})

        data_points = []
        contexts = [{}]

        for context_size in context_sizes:
            for i, doc in enumerate(documents):
                ic(i)
                char_idx = 0
                while char_idx < len(doc['text']):
                    ic(char_idx)
                    document_spans = doc['spans']
                    cur_context = {
                        'doc_id': i,
                        'start_char_idx': char_idx,
                        'end_char_idx': char_idx + context_size,
                        'spans' : [],
                    }

                    for j, (start, end) in enumerate(document_spans):
                        ic(j)
                        if end <= char_idx:
                            continue

                        cur_context['spans'].append({
                            'start_char_idx': max(start, char_idx),
                            'end_char_idx': min(end, char_idx + context_size),
                            'marked': start >= char_idx and end <= char_idx + context_size,
                            'span_id': j
                        })

                        if end > char_idx + context_size:
                            break

                    if cur_context == contexts[-1]:
                        char_idx = cur_context['end_char_idx'] - surround_character_size
                        continue

                    contexts.append(cur_context)
                    if len(cur_context['spans']) == 1 and not cur_context['spans'][0]['marked']:
                        char_idx = cur_context['end_char_idx'] - surround_character_size
                    else:
                        char_idx = cur_context['spans'][-1]['start_char_idx'] - surround_character_size

        contexts.pop(0)

        for nda_name, nda_desc in hypothesis.items():
            for i, context in enumerate(contexts):
                data_point = {}
                data_point['hypotheis'] = nda_desc
                cur_premise = ""
                data_point['marked_beg'] = context['spans'][0]['marked']
                data_point['marked_end'] = context['spans'][-1]['marked']

                if len(context['spans']) == 1:
                    data_point['marked_end'] = True

                span_labels = []

                for span in context['spans']:
                    if span['marked']:
                        span_labels.append(int(span['span_id'] in documents[context['doc_id']]['annotation_sets'][0]['annotations'][nda_name]['spans']))

                    cur_premise += ' [SPAN] '
                    cur_premise += documents[context['doc_id']]['text'][span['start_char_idx']:span['end_char_idx']]

                evidence = any(span_labels)

                data_point['premise'] = cur_premise

                nli_label = label_dict[documents[context['doc_id']]['annotation_sets'][0]['annotations'][nda_name]['choice']]

                if not evidence and nli_label != label_dict['NotMentioned']:
                    nli_label = label_dict['Ignore']

                data_point['nli_label'] = torch.tensor(nli_label, dtype=torch.long)
                data_point['span_labels'] = torch.tensor(span_labels, dtype=torch.long)

                data_points.append(data_point)

        self.data_points = data_points
        self.span_token_id = self.tokenizer.convert_tokens_to_ids('[SPAN]')

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        tokenized_data = self.tokenizer(
            [self.data_points[idx]['hypotheis']],
            [self.data_points[idx]['premise']],
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        tokenized_data['input_ids'] = tokenized_data['input_ids'].squeeze()
        tokenized_data['attention_mask'] = tokenized_data['attention_mask'].squeeze()
        tokenized_data['token_type_ids'] = tokenized_data['token_type_ids'].squeeze()

        span_indices = torch.where(tokenized_data['input_ids'] == self.span_token_id)[0]

        if not self.data_points[idx]['marked_beg']:
            span_indices = span_indices[1:]
        
        if not self.data_points[idx]['marked_end'] or tokenized_data['attention_mask'][-1] == 0:
            span_indices = span_indices[:-1]

        return {
            'input_ids': tokenized_data['input_ids'],
            'attention_mask': tokenized_data['attention_mask'],
            'token_type_ids': tokenized_data['token_type_ids'],
            'span_indices': span_indices,
            'nli_label': self.data_points[idx]['nli_label'],
            'span_labels': self.data_points[idx]['span_labels'][:len(span_indices)]
        }

# %%
train_data = load_data(os.path.join(cfg['raw_data_dir'], cfg['train_path']))
dev_data = load_data(os.path.join(cfg['raw_data_dir'], cfg['dev_path']))
test_data = load_data(os.path.join(cfg['raw_data_dir'], cfg['test_path']))

hypothesis = get_hypothesis(train_data)

train_data = train_data['documents']
dev_data = dev_data['documents']
test_data = test_data['documents']

# train_data = train_data[:2]
# dev_data = dev_data[:2]
# test_data = test_data[:2]

ic.disable()

ic(len(train_data), len(dev_data), len(test_data))
train_dataset = NLIDataset(train_data, tokenizer, hypothesis, [1000, 1100, 1200], 50)
dev_dataset = NLIDataset(dev_data, tokenizer, hypothesis, [1000, 1100, 1200], 50)
test_dataset = NLIDataset(test_data, tokenizer, hypothesis, [1000, 1100, 1200], 50)

ic.enable()

del train_data
del dev_data
del test_data
del hypothesis
# save the datasets
torch.save(train_dataset, os.path.join(cfg['dataset_dir'], 'train_dataset.pt'))
torch.save(dev_dataset, os.path.join(cfg['dataset_dir'], 'dev_dataset.pt'))
torch.save(test_dataset, os.path.join(cfg['dataset_dir'], 'test_dataset.pt'))

# %%
# load the datasets
train_dataset = torch.load(os.path.join(cfg['dataset_dir'], 'train_dataset.pt'))
dev_dataset = torch.load(os.path.join(cfg['dataset_dir'], 'dev_dataset.pt'))
test_dataset = torch.load(os.path.join(cfg['dataset_dir'], 'test_dataset.pt'))

# %%
bert.resize_token_embeddings(len(train_dataset.tokenizer), pad_to_multiple_of=8)

# %%
from torch import nn
class ContractNLI(nn.Module):
    def __init__(self, bert, num_labels, ignore_index):
        super().__init__()
        self.bert = bert
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False

        self.embedding_dim = self.bert.config.hidden_size
        self.num_labels = num_labels
        self.lambda_ = 1
        self.nli_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.span_criterion = nn.BCEWithLogitsLoss()

        self.span_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, 1)
        )

        self.nli_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, self.num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, span_indices):
        outputs = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True).hidden_states[-1]

        gather = torch.gather(outputs, 1, span_indices.unsqueeze(2).expand(-1, -1, outputs.shape[-1]))
        ic(gather.shape)

        masked_gather = gather[span_indices != 0]
        ic(masked_gather.shape)
        span_logits = self.span_classifier(masked_gather)
        ic(span_logits.shape)
        
        nli_logits = self.nli_classifier(outputs[:, 0, :])

        return span_logits, nli_logits

# %%
from transformers import Trainer

class ContractNLITrainer(Trainer):
    def __init__(self, *args, data_collator=None, **kwargs):
        super().__init__(*args, data_collator=data_collator, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        span_label = inputs.pop('span_labels')
        nli_label = inputs.pop('nli_label')

        outputs = model(**inputs)
        span_logits, nli_logits = outputs[0], outputs[1]

        span_loss = self.model.span_criterion(span_logits, span_label.reshape(-1, 1).float())
        nli_loss = self.model.nli_criterion(nli_logits, nli_label)

        if torch.isnan(nli_loss):
            nli_loss = torch.tensor(0, dtype=torch.float32, device=DEVICE)

        loss = span_loss + self.model.lambda_ * nli_loss

        if torch.isnan(loss):
            ic(inputs['input_ids'])
            ic(nli_label)
            ic(nli_logits)

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def collate_fn(features):
        span_indices_list = [feature['span_indices'] for feature in features]
        max_len = max([len(span_indices) for span_indices in span_indices_list])
        span_indices_list = [torch.cat([span_indices, torch.zeros(max_len - len(span_indices), dtype=torch.long)]) for span_indices in span_indices_list]

        input_ids = torch.stack([feature['input_ids'] for feature in features])
        attention_mask = torch.stack([feature['attention_mask'] for feature in features])
        token_type_ids = torch.stack([feature['token_type_ids'] for feature in features])
        span_indices = torch.stack(span_indices_list)
        nli_label = torch.stack([feature['nli_label'] for feature in features])
        span_label = torch.cat([feature['span_labels'] for feature in features], dim=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'span_indices': span_indices,
            'span_labels': span_label,
            'nli_label': nli_label
        }

# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
    auto_find_batch_size=True,
    output_dir=cfg['results_dir'],   # output directory
    num_train_epochs=5,              # total number of training epochs
    warmup_steps=10,                 # number of warmup steps for learning rate scheduler
    gradient_accumulation_steps=4,   # number of updates steps to accumulate before performing a backward/update pass
    logging_steps=0.25,
    eval_steps=0.25,
    evaluation_strategy='steps',
    save_strategy='steps',
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,
    run_name='1',
    label_names=['nli_label', 'span_labels'],
    report_to='wandb'
)

# %%
trainer = ContractNLITrainer(
    model=ContractNLI(bert, len(get_labels()), ignore_index=get_labels()['Ignore']).to(DEVICE),
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset,            # evaluation dataset
    data_collator=ContractNLITrainer.collate_fn,
)

# %%
ic.disable()
trainer.train()
ic.enable()

# %%



