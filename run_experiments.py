import transformers
import torch
import torch.nn as nn
import openai
import torch.nn.functional as F
from models.LSTM_Classifier import LSTMClassifier
from utils.knowledge_graph import KG
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
import argparse
import json
import os
from tqdm import tqdm
from models.crt import CrtModelConfig, CrtModel, CrtTrainer
from collections import defaultdict
LLM_PATH_CLASSIFY_PROMPT = "I want to prove that there is a {r_q} relationship between {e_h} and {e_t}. Please select those unreasonable paths from the following possible proof paths: {paths}.\n Only output the path numbers and strictly follow the output format: \"[ID1], [ID2], ... [IDn]\".\nExample: 1, 10, 5"

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
                # ri = self.kg.kg_graph[eh][et][ri]['relation']
                # if '_inv' in ri:
                #     et, eh = eh, et
                # ri = self.kg.relation2id[ri]
                lstm_path.append((eh, et, ri, rq))
            else: 
                lstm_path.append((id_path[-1][0], id_path[-1][0], 2, rq))
        return lstm_path
    
    def __getitem__(self, index):  
        return {
            'lstm_path': torch.tensor(self.generate_lstm_path(index)).clone().detach(),
            'id_query': torch.tensor(self.dataset[index]['id_query']).clone().detach(),
            'label': torch.tensor([self.dataset[index]['label']], dtype=torch.float32).clone().detach(),
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
            'lstm_path': torch.tensor(lstm_path).clone().detach(),
            'id_query': torch.tensor(self.dataset[index]['id_query']).clone().detach(),
            'label': torch.tensor([self.dataset[index]['label']], dtype=torch.float32).clone().detach(),
            'query': self.dataset[index]['query'],
            'path': self.dataset[index]['path'],
            'query_text': query_text,
        }
    
    def __len__(self):
        return len(self.dataset)

def list_to_tuple(lst):
    return tuple(list_to_tuple(x) if isinstance(x, list) else x for x in lst)

def get_responses(
    client,
    model='gpt-4-0125-preview',
    LLM_query='',
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': LLM_query
            }
        ]
    )
    return response.choices[0].message.content

def sequeenize_dataset(
    dataset, 
    relabel=False, 
    filter_by_relation=False,
    max_num_per_relation=500,
):
    relation_count = defaultdict(int)
    sequeenized_dataset = []
    client = openai.OpenAI(api_key='sk-xxx')
    for item in tqdm(dataset):
        if len(item['paths']) == 0:
            continue
        if filter_by_relation and relation_count[item['id_query'][2]] > max_num_per_relation:
            continue
        if relabel:
            e1, e2, query_relation = item['query']
            paths_text = ''
            for i, path in enumerate(item['paths']):
                paths_text += f'{i + 1}. {path}\n'
            response = get_responses(
                # model='gpt-3.5-turbo-0125',
                LLM_query=LLM_PATH_CLASSIFY_PROMPT.format(
                r_q=query_relation,
                e_h=e1,
                e_t=e2,
                paths=paths_text,
            ), client=client,)
            if len(response) > 0 and \
               response[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                neg_ids = response.split(',')
                item['label'] = [1] * len(item['paths'])
                for neg_id in neg_ids:
                    if int(neg_id) > len(item['paths']):
                        break
                    item['label'][int(neg_id) - 1] = 0
        
        for i in range(len(item['paths'])):    
            relation_count[item['id_query'][2]] += 1
            sequeenized_dataset.append({
                'id_path': item['id_paths'][i],
                'label': item['label'][i],
                'path': item['paths'][i],
                'query': item['query'],
                'id_query': item['id_query'],
            })
    return sequeenized_dataset

def bert_collate(batch, tokenizer):
    label, path, query_text = zip(*[(item['label'], 'Context: ' + item['path'], item['query_text']) for item in batch])
    encoded_text = tokenizer(
        text=query_text,
        text_pair=path, 
        return_tensors='pt', padding=True, truncation=True, max_length=512)
    return {
        'label': torch.stack(label),
        'input_ids': encoded_text['input_ids'],
        'attention_mask': encoded_text['attention_mask'],
        'token_type_ids': encoded_text['token_type_ids'],
    }
    
if __name__ == '__main__':
    
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='WN18RR')
    parser.add_argument('--num_hop', type=int, default=3)
    parser.add_argument('--num_candidates', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--ablation', type=str, default='')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.device))
    head_type = ['head', 'tail']
    kg_path = os.path.join('kg', args.dataset_name)
    kg = KG(kg_path=kg_path)
    num_entity, num_relation = max(kg.entity2id.values()) + 1, max(kg.relation2id.values()) + 1
    if args.ablation == 'ne':
        lstm_dataset_path = os.path.join('data', args.dataset_name, args.ablation + f'lstm_{head_type}.json')
    else:
        lstm_dataset_path = os.path.join('data', args.dataset_name, f'lstm_{head_type}.json')
    print('lstm_dataset_path:', lstm_dataset_path)
    if os.path.exists(lstm_dataset_path):
        lstm_dataset = json.load(open(lstm_dataset_path))
    else:
        triples = kg.triples
        lstm_dataset = kg.generate_train_paths(
            triples,
            neg_num=1,
            num_hop=args.num_hop,
            kept_pos_path_num=10,
            kept_neg_path_num=10,
            mask_entity=args.ablation == 'ne',)
        json.dump(lstm_dataset, open(lstm_dataset_path, mode='w'), indent=4)
    lstm_relabeled_dataset_path = os.path.join('data', args.dataset_name, f'lstm_relabeled{head_type}.json')
    print('lstm_relabeled_dataset_path:', lstm_relabeled_dataset_path)
    if os.path.exists(lstm_relabeled_dataset_path):
        lstm_dataset = json.load(open(lstm_relabeled_dataset_path))
    else:
        lstm_dataset = sequeenize_dataset(
            lstm_dataset, 
            relabel=True, filter_by_relation=True, 
            max_num_per_relation=args.num_candidates)
        json.dump(lstm_dataset, open(lstm_relabeled_dataset_path, mode='w'), indent=4)
    lstm_dataset = LSTMDataset(
        kg,
        lstm_dataset,
        max_hop=args.num_hop,
    )
    
    classifier = LSTMClassifier(
        input_size=kg.entity_embedding_dim, 
        hidden_size=50, 
        num_layers=args.num_hop, 
        num_relation=num_relation,
        num_entity=num_entity,
        output_dim=1)
    classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = F.binary_cross_entropy_with_logits
    lstm_dataloader = DataLoader(
        dataset=lstm_dataset,
        batch_size=16,
        shuffle=True,)
    # if os.path.exists('models/LSTM_Classifier.pth'):
    if False:
        classifier.load_state_dict(torch.load('models/LSTM_Classifier.pth'))
    else:
        for epoch in range(args.num_epochs):
            for i, batch in enumerate(lstm_dataloader):
                for key in ['lstm_path', 'id_query', 'label']:
                    batch[key] = batch[key].to(device)
                logits = classifier(batch)
                loss = criterion(logits, batch['label'])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f'lstm: epoch {epoch} loss: {loss.item()}')
        torch.save(classifier.state_dict(), 'models/LSTM_Classifier.pth')
    if args.ablation == 'ne':
        bert_dataset_path = os.path.join('data', args.dataset_name, f'nebertdata{head_type}.json')
    else:
        bert_dataset_path = os.path.join('data', args.dataset_name, f'bertdata{head_type}.json')
    print('bert_dataset_path:', bert_dataset_path)
    if not os.path.exists(bert_dataset_path):
        bert_dataset = kg.generate_train_paths(
            kg.triples,
            neg_num=5,
            num_hop=args.num_hop,
            kept_pos_path_num=10,
            mask_entity=args.ablation == 'ne',
        )
        json.dump(bert_dataset, open(bert_dataset_path, 'w'), indent=4)
    else:
        bert_dataset = json.load(open(bert_dataset_path))
    bert_dataset = BertDataset(
        sequeenize_dataset(bert_dataset, relabel=False, filter_by_relation=False),
        max_hop=args.num_hop,
    )
    bert_dataloader = DataLoader(
        dataset=bert_dataset,
        batch_size=128,
        shuffle=False,)
    bert_filter_dataset = []
    # classifier.eval()
    
    
    print('********Data Filtering********')
    original_id_paths_dict = defaultdict(int)
    filtered_id_paths_dict = defaultdict(int)
    for i, batch in enumerate(tqdm(bert_dataloader)):
        for key in ['lstm_path', 'id_query', 'label']:
            batch[key] = batch[key].to(device)
        sc_score = classifier(batch).squeeze()
        # sc_score = torch.rand(batch['label'].shape[0])
        batch_size = batch['lstm_path'].shape[0]
        for j in range(sc_score.shape[0]): 
            if batch['label'][j].item() == 1:
                lstm_path = repr(list_to_tuple(batch['lstm_path'][j].tolist()))
                original_id_paths_dict[lstm_path] += 1 
                if args.ablation == 'np' or sc_score[j].item() > args.threshold:
                    bert_filter_dataset.append(bert_dataset[i * batch_size + j])
                    filtered_id_paths_dict[lstm_path] += 1
            elif batch['label'][j].item() == 0:
                if args.ablation == 'nn' or sc_score[j].item() < 1 - args.threshold:
                    bert_filter_dataset.append(bert_dataset[i * batch_size + j])
    json.dump(original_id_paths_dict, open(args.dataset_name + str(args.threshold) + '_original.json', 'w'), indent=4)
    json.dump(filtered_id_paths_dict, open(args.dataset_name + str(args.threshold) + '_filtered.json', 'w'), indent=4)
    bert_filter_dataset = bert_dataset
    bert_filter_dataset = BertDataset(
        bert_filter_dataset,
        max_hop=args.num_hop,
        generate_lstm_path=False,
    )    
    bert_dataloader = DataLoader(
        dataset=bert_filter_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: bert_collate(batch, tokenizer=bert_dataset.tokenizer),
    )
    
    config = CrtModelConfig(
        lm_path='bert-base-uncased',
    )
    crt_model = CrtModel(config)
    crt_model.to(device)
    optimizer = torch.optim.Adam(crt_model.parameters(), lr=0.00001)
    criterion = F.binary_cross_entropy_with_logits
    print('********Bert Training********')
    for epoch in range(2):
        for i, batch in enumerate(tqdm(bert_dataloader)):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            logits = crt_model(batch)
            loss = criterion(logits, batch['label'])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 50 == 0:
                print(f'bert: batch {i} loss: {loss.item()}')
    torch.save(crt_model.state_dict(), os.path.join('saved_models', args.dataset_name + args.ablation + '_'.join(head_type) + 'CrtModel.pth'))