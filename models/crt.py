import os
import sys
from typing import List

import time
import fire
import torch
import transformers
from datasets import load_dataset
from transformers import Trainer

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModel

from utils.prompter import Prompter

HEAD_PROMPT = 'The subject of the target inference statement is {}'
PATH_PROMPT = 'To prove the existence of relation {}, we use the reasoning path: {}. '
TAIL_PROMPT = 'The object of the target inference statement is {}'

# 用同一个LM还是用不同的LM

# TODO: 先用相同的
# TODO: LLM挑选最合适的path

class CrtModelConfig:
    def __init__(self, lm_path, hidden_size=None):
        self.lm_path = lm_path
        self.hidden_size = hidden_size

class CrtModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm = AutoModel.from_pretrained(config.lm_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.lm_path)
        self.classifier = torch.nn.Linear(self.lm.config.hidden_size, 1)
        self.dropout = torch.nn.Dropout(0.3)
        

    def forward(self, batch, labels=None, use_sigmoid=False):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        outputs = outputs[1]
        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
        if use_sigmoid:
            outputs = torch.sigmoid(outputs)
        # outputs = self.sigmoid(outputs)
        return outputs
        

    def print_trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f'{name}: {param.size()}')

class CrtTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs
        loss_fct = torch.nn.BCELoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
        