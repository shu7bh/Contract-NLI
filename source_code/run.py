# %%
from transformers import AutoTokenizer, AutoModelForMaskedLM
import logging as log
log.basicConfig(level=log.DEBUG)

# %%
# model = AutoModelForMaskedLM.from_pretrained(artifact_dir)

# %%
# import wandb
# run = wandb.init()
# artifact = run.use_artifact('contract-nli-db/contract-nli/checkpoint:v1', type='model')
# artifact_dir = artifact.download()

# %%
import sys
sys.path.append('../')
from baselines.utils import *
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ['WANDB_ENTITY'] = 'contract-nli-db'
os.environ['WANDB_PROJECT'] = 'contract-nli'
os.environ['WANDB_LOG_MODEL'] = 'end'

# %%
import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE

# %%
cfg['model_name'] = 'bert-base-uncased'
cfg['trained_model_dir'] = '/scratch/shu7bh/contract_nli/trained_model/'
cfg['batch_size'] = 32
cfg

# %%
# create dir if not exists
from pathlib import Path
Path(cfg["models_save_dir"]).mkdir(parents=True, exist_ok=True)
Path(cfg["dataset_dir"]).mkdir(parents=True, exist_ok=True)

# %%
tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

tokenizer.save_pretrained(cfg['models_save_dir'])

# %%
tokenizer = AutoTokenizer.from_pretrained(cfg['models_save_dir'])

# %%
from icecream import ic

# %%
def get_hypothesis_idx(hypothesis_name):
    return int(hypothesis_name.split('-')[-1])

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
                doc_id = context['doc_id']
                hypothesis_id = get_hypothesis_idx(nda_name)
                span_ids = []

                if len(context['spans']) == 1:
                    data_point['marked_end'] = True

                span_labels = []

                for span in context['spans']:
                    if span['marked']:
                        span_labels.append(int(span['span_id'] in documents[context['doc_id']]['annotation_sets'][0]['annotations'][nda_name]['spans']))
                        span_ids.append(span['span_id'])

                    cur_premise += ' [SPAN] '
                    cur_premise += documents[context['doc_id']]['text'][span['start_char_idx']:span['end_char_idx']]

                evidence = any(span_labels)

                data_point['premise'] = cur_premise

                nli_label = label_dict[documents[context['doc_id']]['annotation_sets'][0]['annotations'][nda_name]['choice']]

                if not evidence and nli_label != label_dict['NotMentioned']:
                    nli_label = label_dict['Ignore']

                data_point['nli_label'] = torch.tensor(nli_label, dtype=torch.long)
                data_point['span_labels'] = torch.tensor(span_labels, dtype=torch.long)
                data_point['doc_id'] = torch.tensor(doc_id, dtype=torch.long)
                data_point['hypothesis_id'] = torch.tensor(hypothesis_id, dtype=torch.long)
                data_point['span_ids'] = torch.tensor(span_ids, dtype=torch.long)

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
        
        span_ids = self.data_points[idx]['span_ids']
        span_ids = span_ids[:len(span_indices)]

        return {
            'input_ids': tokenized_data['input_ids'],
            'attention_mask': tokenized_data['attention_mask'],
            'token_type_ids': tokenized_data['token_type_ids'],
            'span_indices': span_indices,
            'nli_label': self.data_points[idx]['nli_label'],
            'span_labels': self.data_points[idx]['span_labels'][:len(span_indices)],
            'data_for_metrics': {
                'doc_id': self.data_points[idx]['doc_id'],
                'hypothesis_id': self.data_points[idx]['hypothesis_id'],
                'span_ids': span_ids,
            }
        }

# %%
train_data = load_data(os.path.join(cfg['raw_data_dir'], cfg['train_path']))
dev_data = load_data(os.path.join(cfg['raw_data_dir'], cfg['dev_path']))
test_data = load_data(os.path.join(cfg['raw_data_dir'], cfg['test_path']))

hypothesis = get_hypothesis(train_data)

train_data = train_data['documents']
dev_data = dev_data['documents']
test_data = test_data['documents']

train_data = train_data[:100]
dev_data = dev_data[:100]
test_data = test_data[:100]

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
cfg

# %%
from transformers import PreTrainedModel, PretrainedConfig

class ContractNLIConfig(PretrainedConfig):
    def __init__(self, lambda_ = 1, bert_model_name = cfg['model_name'], num_labels = len(get_labels()), ignore_index = get_labels()['Ignore'], **kwargs):
        super().__init__(**kwargs)
        self.bert_model_name = bert_model_name
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.lambda_ = lambda_

# %%
from transformers import AutoModel
from torch import nn

class ContractNLI(PreTrainedModel):
    config_class = ContractNLIConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModel.from_pretrained(config.bert_model_name)
        self.bert.resize_token_embeddings(self.bert.config.vocab_size + 1, pad_to_multiple_of=8)
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False

        self.embedding_dim = self.bert.config.hidden_size
        self.num_labels = config.num_labels
        self.lambda_ = config.lambda_
        self.nli_criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
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

        # initialize weights
        self.init_weights()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # use the same initialization as bert
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids, span_indices):
        outputs = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True).hidden_states[-1]

        gather = torch.gather(outputs, 1, span_indices.unsqueeze(2).expand(-1, -1, outputs.shape[-1]))

        masked_gather = gather[span_indices != 0]
        span_logits = self.span_classifier(masked_gather)
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
        inputs.pop('data_for_metrics')

        outputs = model(**inputs)
        span_logits, nli_logits = outputs[0], outputs[1]

        span_loss = self.model.span_criterion(span_logits, span_label.reshape(-1, 1).float())
        nli_loss = self.model.nli_criterion(nli_logits, nli_label)

        if torch.isnan(nli_loss):
            nli_loss = torch.tensor(0, dtype=torch.float32, device=DEVICE)

        loss = span_loss + self.model.lambda_ * nli_loss

        if torch.isnan(loss):
            print("here")

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def collate_fn(features):
        span_indices_list = [feature['span_indices'] for feature in features]
        max_len = max([len(span_indices) for span_indices in span_indices_list])
        span_indices_list = [torch.cat([span_indices, torch.zeros(max_len - len(span_indices), dtype=torch.long)]) for span_indices in span_indices_list]

        span_ids_list = [feature['data_for_metrics']['span_ids'] for feature in features]
        max_len = max([len(span_ids) for span_ids in span_ids_list])
        span_ids_list = [torch.cat([span_ids, torch.zeros(max_len - len(span_ids), dtype=torch.long)]) for span_ids in span_ids_list]

        input_ids = torch.stack([feature['input_ids'] for feature in features])
        attention_mask = torch.stack([feature['attention_mask'] for feature in features])
        token_type_ids = torch.stack([feature['token_type_ids'] for feature in features])
        span_indices = torch.stack(span_indices_list)
        nli_label = torch.stack([feature['nli_label'] for feature in features])
        span_label = torch.cat([feature['span_labels'] for feature in features], dim=0)
        data_for_metrics = {
            'doc_id': torch.stack([feature['data_for_metrics']['doc_id'] for feature in features]),
            'hypothesis_id': torch.stack([feature['data_for_metrics']['hypothesis_id'] for feature in features]),
            'span_ids': torch.stack(span_ids_list),
        }

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'span_indices': span_indices,
            'nli_label': nli_label,
            'span_labels': span_label,
            'data_for_metrics': data_for_metrics,
        }

# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
    auto_find_batch_size=True,
    output_dir=cfg['results_dir'],   # output directory
    num_train_epochs=15,            # total number of training epochs
    gradient_accumulation_steps=4,   # number of updates steps to accumulate before performing a backward/update pass
    logging_strategy='epoch',
    # eval_steps=0.25,
    # save_steps=0.25,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,
    label_names=['nli_label', 'span_labels', 'data_for_metrics'],
    report_to='wandb',
)

# %%
from transformers import EarlyStoppingCallback

# %%
def wandb_hp_space(trial):
    return {
        "method": "random",
        "metric": {
            "name": "eval/loss",
            "goal": "minimize"
        },
        "parameters": {
            "learning_rate": {
                "values": [1e-5, 3e-5, 5e-5]
            },
            "lambda_": {
                "values": [0.05, 0.1, 0.4]
            },
        }
    }

# %%
def model_init(trial):
    if trial is None:
        return ContractNLI(ContractNLIConfig())

    return ContractNLI(ContractNLIConfig(lambda_=trial['lambda_']))

# %%
# config = ContractNLIConfig()

# model = ContractNLI(config).to(DEVICE)
trainer = ContractNLITrainer(
    model=None,                          # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset,            # evaluation dataset
    data_collator=ContractNLITrainer.collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.01)],
    model_init=model_init,
)

# %%
# trainer.train()

# %%
trainer.hyperparameter_search(
    hp_space=wandb_hp_space,
    backend='wandb',
    n_trials=5,
)

# %%
# model = ContractNLI.from_pretrained(cfg['trained_model_dir']).to(DEVICE)

# %%
# trainer = ContractNLITrainer(
#     model=model,                  # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=train_dataset,         # training dataset
#     eval_dataset=dev_dataset,            # evaluation dataset
#     data_collator=ContractNLITrainer.collate_fn,
# )

# %%
# trainer.train()


