#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install scikit-multilearn
#!pip install -U transformers==4.35.2
#!pip install tensorflow


#!pip install scikit-multilearn
#!pip install -U transformers==4.38.2
#!pip install tensorflow
#!pip install accelerate==0.27.2


# In[2]:


import os
import random
import functools
import csv
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from skmultilearn.model_selection import iterative_train_test_split
from datasets import Dataset, DatasetDict
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
import transformers
print(transformers.__version__)

from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory


# In[3]:


#from accelerate import Accelerator
#from datetime import timedelta
#from accelerate import InitProcessGroupKwargs

# Create the custom configuration
#process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))  # 0.5 hours

# Instantiate Accelerator with the custom configuration
#accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])



# In[4]:


def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.stack(d['labels'])
    return d


def compute_metrics(p):
    predictions, labels = p
    f1_micro = f1_score(labels, predictions > 0, average = 'micro')
    f1_macro = f1_score(labels, predictions > 0, average = 'macro')
    f1_weighted = f1_score(labels, predictions > 0, average = 'weighted')
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


# create custom trainer class to be able to pass label weights and calculate multilabel loss
class CustomTrainer(Trainer):

    def __init__(self, label_weights, **kwargs):
        super().__init__(**kwargs)
        self.label_weights = label_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32), pos_weight=self.label_weights)
        #accelerator.backward(loss, retain_graph=True)
        
        return (loss, outputs) if return_outputs else loss



# In[5]:


#!unzip ../2024-03-08_data_multiple-fields_german-only-20240319T075025Z-001.zip


# In[6]:


datadir = "2024-03-08_data_multiple-fields_german-only"
trainfile = datadir + "/train_data.csv"
with open(trainfile, newline='') as csvfile:
    fulldata = list(csv.reader(csvfile, delimiter=','))        
    header_row = fulldata.pop(0)
 
traindata=[]
for line in fulldata:   
    #if (len(line[1])>5000 and len(line[1])<=200000):
    if True:
        datum=[]
        datum.append(line[1])
        for elm in line[2:]:
            datum += list(elm)

        traindata.append(datum)
    

#print (header_row)
#print ("full" , len(fulldata))

#trainmetafile = datadir + "/train_metadata.csv"
#with open(trainmetafile, newline='') as csvfile:
#    fulldata = list(csv.reader(csvfile, delimiter=','))        
#    header_row = fulldata.pop(0)

#print (header_row)
#print ("full" , len(fulldata))
random.seed(0)
random.shuffle(traindata)
data=traindata #[0:20000]

print (len(data))

# In[7]:


# reshape
text, labels = list(zip(*[(f'{row[0].strip()}', row[1:]) for row in data]))
labels = np.array(labels, dtype=int)


# create label weights
label_weights = 1 - labels.sum(axis=0) / labels.sum()

#print(label_weights)

# stratified train test split for multilabel ds
row_ids = np.arange(len(labels))
train_idx, y_train, val_idx, y_val = iterative_train_test_split(row_ids[:,np.newaxis], labels, test_size = 0.2)
x_train = [text[i] for i in train_idx.flatten()]
x_val = [text[i] for i in val_idx.flatten()]

# create hf dataset
ds = DatasetDict({
    'train': Dataset.from_dict({'text': x_train, 'labels': y_train}),
    'val': Dataset.from_dict({'text': x_val, 'labels': y_val})
})


# In[9]:
#device_index = Accelerator().process_index
#device_map = {"": device_index}
device_map ="auto"
#device_map = {"": "cuda:0"}



# model name
model_name = 'mistralai/Mistral-7B-v0.1'
#model_name = 'HuggingFaceH4/zephyr-7b-beta'

# preprocess dataset with tokenizer
def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['text'])
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
tokenized_ds = tokenized_ds.with_format('torch')


quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
)

# lora config
lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
)


# load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    #quantization_config=quantization_config,
    num_labels=labels.shape[1],
    device_map=device_map       
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id


OUTPUTDIR='./multilabel_classification'


training_args = TrainingArguments(
    output_dir = OUTPUTDIR,
    learning_rate = 1e-4,
    per_device_train_batch_size = 4, 
    per_device_eval_batch_size = 4,
    num_train_epochs = 6,
    weight_decay = 0.01,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    ddp_find_unused_parameters=False
)

# train
trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_ds['train'],
    eval_dataset = tokenized_ds['val'],
    tokenizer = tokenizer,
    data_collator = functools.partial(collate_fn, tokenizer=tokenizer),
    compute_metrics = compute_metrics,
    label_weights = torch.tensor(label_weights, device=model.device)
    
)


trainer.train()


# save model
peft_model_id = OUTPUTDIR
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
torch.save(model.score.state_dict(), OUTPUTDIR + '/score-params_edu.pt')

