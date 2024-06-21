import transformers
import torch
import torch.nn as nn
import openai
import torch.nn.functional as F
# from models.LSTM_Classifier import LSTMClassifier
from utils.knowledge_graph import KG
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class LSTMDataset(Dataset):
    def __init__(self, kg, dataset, max_hop=5):
        self.kg = kg
        self.max_hop = max_hop
        self.dataset = dataset
        
    def generate_lstm_path(self, index):
        id_path = self.dataset[index]['id_path']
        rq = self.dataset[index]['id_query'][2]
        lstm_path = []
        for i in range(self.max_hop):
            if i < len(id_path):
                eh, et, ri = id_path[i]
                lstm_path.append((eh, et, ri, rq))
            else: 
                lstm_path.append((id_path[-1][0], id_path[-1][0], 2, rq))
        return lstm_path
    
    def __getitem__(self, index):  
        return {
            'lstm_path': torch.tensor(self.generate_lstm_path(index)),
            'id_query': torch.tensor(self.dataset[index]['id_query']),
            'label': torch.tensor([self.dataset[index]['label']], dtype=torch.float32),
            'query': self.dataset[index]['query'],
            'path': self.dataset[index]['path'],
        }
    
    def __len__(self):
        return len(self.dataset)

class BertDataset(LSTMDataset):
    def __init__(self, dataset, tokenizer=None, max_length=512, max_hop=5, generate_lstm_path=True):
        self.dataset = dataset
        self.max_length = max_length
        self.max_hop = max_hop
        self.glp = generate_lstm_path
        if not tokenizer:
            self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        lstm_path = self.generate_lstm_path(index) if self.glp else self.dataset[index]['lstm_path']
        e1, e2, query_relation = self.dataset[index]['query']
        query_text = 'Question: ' + e1 + ' ' + query_relation + ' what? Is the answer ' + e2 + '?'
        return {
            'lstm_path': torch.tensor(lstm_path),
            'id_query': torch.tensor(self.dataset[index]['id_query']),
            'label': torch.tensor([self.dataset[index]['label']], dtype=torch.float32),
            'query': self.dataset[index]['query'],
            'path': self.dataset[index]['path'],
            'query_text': query_text,
        }
    
    def __len__(self):
        return len(self.dataset)
