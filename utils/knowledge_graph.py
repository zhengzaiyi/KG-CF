import wordninja
import networkx as nx
import os
import random
from tqdm import trange
import torch
from tqdm import tqdm
import torch.nn as nn
import json

def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            v, w = line.strip().split()
            w = int(w)
            index[v] = w
            rev_index[w] = v
    return index, rev_index

def nell995_preprocess(e1, e2, r, entity2text, relation2text):
    if e1 not in entity2text:
        if 'concept' in e1:
            entity2text[e1] = ' '.join(wordninja.split(e1[8:]))
        else:
            entity2text[e1] = e1
    if e2 not in entity2text:
        if 'concept' in e2:
            entity2text[e2] = ' '.join(wordninja.split(e2[8:]))
        else:
            entity2text[e2] = e2
    if r not in relation2text:
        relation2text[r] = ' '.join(wordninja.split(r[8:]))
    return entity2text[e1], entity2text[e2], relation2text[r]
    
def wn18rr_preprocess(e1, e2, r, entity2text, relation2text):
    if len(entity2text.keys()) > 0:
        return entity2text[e1], entity2text[e2], relation2text[r]
    with open('kg/WN18RR/entity2text.txt') as f:
        for l in f:
            e, text = l.strip().split('\t')
            name = text.split(',')[0]
            description = text[len(name) + 1:]
            entity2text[e] = name
            entity2text[e + '_desc'] = description
    
    with open('kg/WN18RR/relation2text.txt') as f:
        for l in f:
            r, text = l.split(',')
            relation2text[r] = text.split('\n')[0]
            
    return entity2text[e1], entity2text[e2], relation2text[r]

def fb237_preprocess(e1, e2, r, entity2text, relation2text):
    if len(entity2text.keys()) > 0:
        return entity2text[e1], entity2text[e2], relation2text[r]
    with open('kg/FB15K-237/entity2text.txt') as f:
        for l in f:
            e, text = l.strip().split('\t')
            entity2text[e] = text
    
    with open('kg/FB15K-237/relation2text.txt') as f:
        for l in f:
            r, text = l.split('\t')
            relation2text[r] = text
            
    return entity2text[e1], entity2text[e2], relation2text[r]
            
def write_entity2text(entity2text, output_path):
    with open(output_path, 'w') as f:
        for e, text in entity2text.items():
            f.write(e + '\t' + text + '\n')

def write_relation2text(relation2text, output_path):
    with open(output_path, 'w') as f:
        for r, text in relation2text.items():
            f.write(r + '\t' + text + '\n')

preprocess = {
    'NELL-995': nell995_preprocess,
    'WN18RR': wn18rr_preprocess,
    'FB15K-237': fb237_preprocess,
}

class KG(nn.Module):
    def __init__(
        self, 
        kg_path: str, 
        mode='train',
        entity_embedding_dim=128,
        relation_embedding_dim=128,
    ) -> None:
        super(KG, self).__init__()
        self.kg_graph = nx.MultiDiGraph() # all splited_text
        self.entity2text = {} # str(entity_id) -> splited_text
        self.relaiton2text = {} # str(relation_id) -> splited_text
        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.preprocess = preprocess[kg_path.split('/')[-1].split('_')[0]]
        self.triples, self.test_triples = [], []
        self.kg_path = kg_path
        self.entity2id, self.id2entity = load_index(os.path.join(kg_path, 'entity2id.txt'))
        self.relation2id, self.id2relation = load_index(os.path.join(kg_path, 'relation2id.txt'))
        self.train_entities = set()
        with open(os.path.join(kg_path, 'train.triples'), 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                e1, e2, r = self.preprocess(line[0], line[1], line[2], self.entity2text, self.relaiton2text)
                self.triples.append([line[0], line[2], line[1]])
                self.kg_graph.add_edges_from([(line[0], line[1], dict(relation=line[2]))])
                self.kg_graph.add_edges_from([(line[1], line[0], dict(relation=line[2]+'_inv'))]) 
        self.entity_embeddings = nn.Embedding(len(self.entity2text), entity_embedding_dim)
        self.relation_embedding_dim = relation_embedding_dim
        self.relation_embeddings = nn.Embedding(len(self.relaiton2text), relation_embedding_dim)
        
    def is_triple_in_graph(self, e1, e2, r):
        return e1 in self.kg_graph and e2 in self.kg_graph[e1] and r in [r_dict['relation'] for r_dict in self.kg_graph[e1][e2].values()]
    
    def path2idpath(self, path):
        id_path = []
        for edge in path:
            e1, e2, r = edge
            r = self.kg_graph[e1][e2][r]['relation']
            e1, e2, r = self.entity2id[e1], self.entity2id[e2], self.relation2id[r]
            id_path.append((e1, e2, r))
        return id_path   
                      
    def generate_train_paths(
        self, 
        triples, 
        num_hop=3, 
        kept_pos_path_num=10, 
        neg_num=10,
        kept_neg_path_num=3,
        mode='train',
        mask_entity=False,
        head_type = ['head', 'tail']
    ):
        print('generating paths...')
        def path2idpath(path):
            id_path = []
            for edge in path:
                e1, e2, r = edge
                r = self.kg_graph[e1][e2][r]['relation']
                e1, e2, r = self.entity2id[e1], self.entity2id[e2], self.relation2id[r]
                id_path.append((e1, e2, r))
            return id_path
        
        def get_valid_paths(exclude_edge, paths, query_relation=None, label=False, keep_num=6):
            # exclude_edge: (e1, e2, r)
            e1, e2, r_ind = exclude_edge
            if query_relation is None: # True
                query_relation = self.kg_graph[e1][e2][r_ind]['relation']
            id_query = (self.entity2id[e1], self.entity2id[e2], self.relation2id[query_relation])
            query = self.preprocess(e1, e2, query_relation, self.entity2text, self.relaiton2text)
            valid_paths= []
            for path in paths:
                if not any((e1, e2, r_ind) == edge for edge in path):
                    valid_paths.append(path)
            valid_paths.sort(key=lambda x: len(x))
            valid_paths = valid_paths[:keep_num] if len(valid_paths) > keep_num else valid_paths
            valid_id_paths = [path2idpath(path) for path in valid_paths]
            return {
                'paths': [self.generate_path_text(path, mask_entity=mask_entity) for path in valid_paths],
                'id_paths': valid_id_paths,
                'query': query,
                'id_query': id_query,
                'label': [label] * len(valid_paths),
            }
        
        dataset = []
        kept_neg_path_num = kept_pos_path_num if kept_neg_path_num is None else kept_neg_path_num
        for i in trange(len(triples)):
            # pos samples
            e1, r, e2 = triples[i]
            pos_paths = [*nx.algorithms.all_simple_edge_paths(self.kg_graph, source=e1, target=e2, cutoff=num_hop)]
            no_path_count = 0
            if len(pos_paths) == 1:
                no_path_count += 1
            # exclude the path (e1, e2, r)
            if mode == 'train':
                r_inds = [k for k, r_dict in self.kg_graph[e1][e2].items() if r_dict['relation'] == r]  # r_ind in biG local graph
                assert len(r_inds) == 1
                r_ind = r_inds[0]
            elif mode == 'test':
                r_ind = 0
            
            
            # get shortest kept_path_num paths
            pos_paths = get_valid_paths(
                (e1, e2, r_ind), 
                pos_paths, 
                query_relation=r,
                label=True,
                keep_num=kept_pos_path_num)

            dataset.append(pos_paths) 
            
            # neg samples
            j = 0
            ego_graph = nx.ego_graph(self.kg_graph, e1, radius=num_hop)
            e1_neigh_to_dis = list(set(nx.ego_graph(self.kg_graph, e1, radius=num_hop).nodes()) - set([e1, e2]))
            e1_neigh_to_dis = [e2_neg for e2_neg in e1_neigh_to_dis 
                if not self.is_triple_in_graph(e1, e2_neg, r)
                and e2_neg != e2 and e2_neg != e1]
            e2_neigh_to_dis = list(set(nx.ego_graph(self.kg_graph, e2, radius=num_hop).nodes()) - set([e1, e2]))
            e2_neigh_to_dis = [e1_neg for e1_neg in e2_neigh_to_dis
                if not self.is_triple_in_graph(e1_neg, e2, r)
                and e1_neg != e2 and e1_neg != e1]
                
            if 'tail' in head_type:
                while j < neg_num and j < len(e1_neigh_to_dis):
                    if len(e1_neigh_to_dis) == 0:
                        break
                    e2_neg = random.choice(e1_neigh_to_dis)
                    if self.is_triple_in_graph(e1, e2_neg, r):
                        continue
                    neg_paths = [*nx.algorithms.all_simple_edge_paths(self.kg_graph, source=e1, target=e2_neg, cutoff=num_hop)]
                    neg_paths = get_valid_paths(
                        (e1, e2_neg, r_ind), 
                        neg_paths,
                        query_relation=r,
                        label=False,
                        keep_num=kept_neg_path_num
                    )
                    dataset.append(neg_paths)
                    j += 1
            if 'head' in head_type:
                j = 0
                while j < neg_num and j < len(e2_neigh_to_dis):
                    if len(e2_neigh_to_dis) == 0:
                        break
                    e1_neg = random.choice(e2_neigh_to_dis)
                    if self.is_triple_in_graph(e1_neg, e2, r):
                        continue
                    neg_paths = [*nx.algorithms.all_simple_edge_paths(self.kg_graph, source=e1_neg, target=e2, cutoff=num_hop)]
                    neg_paths = get_valid_paths(
                        (e1_neg, e2, r_ind), 
                        neg_paths,
                        query_relation=r,
                        label=False,
                        keep_num=kept_neg_path_num
                    )
                    dataset.append(neg_paths)
                    j += 1
        print('no path count:', no_path_count)    
        return dataset
    
    def generate_test_paths(
        self,
        candidates_triples,
        num_hop=3,
        kept_path_num=3,
        mode='test',
    ):
        def get_valid_paths(exclude_edge, paths, label=False):
            valid_paths = []
            for path in paths:
                if not any(exclude_edge == edge for edge in path):
                    valid_paths.append((exclude_edge, path, label))
            return valid_paths
        
        paths = []
        triple2path_idx_start = []
        no_path_count = 0
        for i in trange(len(candidates_triples)):
            # pos samples
            e1, e2, r = candidates_triples[i]
            tmp_paths = [*nx.algorithms.all_simple_edge_paths(self.kg_graph, source=e1, target=e2, cutoff=num_hop)]
            tmp_paths = get_valid_paths(
                (e1, e2, r), 
                tmp_paths, 
            )
            no_path_count += 1 if len(tmp_paths) == 0 else 0
            triple2path_idx_start.append(len(paths))
            paths.extend(tmp_paths)
        print('no path count:', no_path_count)
        return paths, triple2path_idx_start               
    
    def generate_triple_text(self, e1, e2, r_idx, mask_entity=False, i=0):
        if self.kg_graph[e1][e2][r_idx]['relation'].endswith('_inv'):
            r = self.kg_graph[e1][e2][r_idx]['relation'][: -4]
            e1, e2 = e2, e1
        else:
            r = self.kg_graph[e1][e2][r_idx]['relation']
        e1, e2, r = self.preprocess(e1, e2, r, self.entity2text, self.relaiton2text)
            
        if mask_entity:
            return 'entity{} {} entity{}, '.format(i, r, i + 1)
        return "{} {} {};".format(e1, r, e2)
    
    def generate_path_text(self, path, mask_entity=False):
        path_text = ''
        for i, edge in enumerate(path):
            path_text += self.generate_triple_text(*edge, mask_entity=mask_entity, i = i)
        return path_text + '.\n'
    
    def generate_json(
        self, 
        paths, 
        output_path=None,
        template_name='alpaca',
        instruction='Question: is the following statement true or false?',
        mask_entity=False,
    ):
         
        if template_name == 'crt':
            json_data = []
            # path: ((e1, e2, r), [(e1, e2, r), ...], label)
            for path in paths:
                e1, e2, query_relation = path[0]
                id_query = (self.entity2id[e1], self.entity2id[e2], self.relation2id[query_relation])
                json_data.append({
                    'path': self.generate_path_text(path[1], mask_entity=mask_entity),
                    'id_path': self.path2idpath(path[1]),
                    'query': self.preprocess(*path[0], self.entity2text, self.relaiton2text),
                    'id_query': id_query,
                    'label': 1.0 if path[2] else 0.0,
                })
            
        
        else: 
            raise NotImplementedError
        return json_data
    
    def load_test_triples(self):
        self.ranking_head = []
        self.ranking_head_gt = []
        self.ranking_tail = []
        self.ranking_tail_gt = []
        with open(os.path.join(self.kg_path, 'ranking_head.txt'), 'r') as f:
            for i, line in enumerate(f):
                line = line.strip().split('\t')
                if i % 50 == 0:
                    self.ranking_head_gt.append([line[0], line[2], line[1]])
                self.ranking_head.append([line[0], line[2], line[1]])
        with open(os.path.join(self.kg_path, 'ranking_tail.txt'), 'r') as f:
            for i, line in enumerate(f):
                line = line.strip().split('\t')
                if i % 50 == 0:
                    self.ranking_tail_gt.append([line[0], line[2], line[1]])
                self.ranking_tail.append([line[0], line[2], line[1]])
        print('test triples:', len(self.test_triples))
                               
    
    def forward(self, batch):
        pass         