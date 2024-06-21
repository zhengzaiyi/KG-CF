import os
import sys
from typing import List
from accelerate import Accelerator
import time
import fire
import torch
import transformers
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from models.crt import CrtModelConfig, CrtModel
from run_experiments import BertDataset
import argparse
import json
from typing import Optional
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
from utils.knowledge_graph import KG
from peft import PeftModel, PeftConfig
from run_experiments import bert_collate

def judge_from_ans_text(ans: str):
    # TODO: cosine similarity
    return 'true' in ans.lower()

def generate_and_tokenize_prompt(
    data_point,
    prompter,
    tokenize,
    train_on_inputs=False,
    add_eos_token=True,
):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

def tokenize(prompt, add_eos_token=True, tokenizer=None, cutoff_len=512):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def evaluate_by_crt_scores(
    candidate_triples,
    kg: KG,
    num_hop: int = 4,
    num_candidates: int = 50,
    hits_metric = [1, 5, 10],
    lm_model = "bert-base-uncased",
    dataset_name: str = "WN18RR",
    model_path: str = None,
    type: Optional[str] = None,
):
    accelerator = Accelerator()
    config = CrtModelConfig(
        lm_path=lm_model,
    )
    if model_path is None:
        crt_path = os.path.join('saved_models', dataset_name + 'CrtModel.pth')
    else:
        crt_path = model_path
    text_model = CrtModel(config)
    text_model.load_state_dict(torch.load(crt_path))
    text_model.eval()
    text_model = text_model.to('cuda:0')
    tokenizer = text_model.tokenizer
    
    hits = [0] * (hits_metric[-1] + 1)
    mrr = 0
    paths_scores = []      # test_triples * candidates * paths_for_each
    candidates_scores = [] # test_triples * candidates
    ranks = []             # test_triples
    test_paths_di = os.path.join('kg', dataset_name, f'{type}_num_hop{num_hop}_test.json')
    triple2path_idx_start_di = os.path.join('kg', dataset_name, f'{type}_triple2path_idx_start.json')
    if not os.path.exists(test_paths_di):
        possible_paths, triple2path_idx_start = kg.generate_test_paths(
            candidates_triples=candidate_triples,
        )
        json.dump(possible_paths, open(test_paths_di, 'w'), indent=4)
        json.dump(triple2path_idx_start, open(triple2path_idx_start_di, 'w'))
    else:
        possible_paths = json.load(open(test_paths_di, 'r'))
        triple2path_idx_start = json.load(open(triple2path_idx_start_di, 'r'))
        
    bert_dataset = kg.generate_json(possible_paths, template_name='crt',)
    bert_dataset = BertDataset(bert_dataset)
    test_dataloader = DataLoader(
        bert_dataset,
        shuffle=False,
        batch_size=128,
        collate_fn=lambda batch: bert_collate(batch, tokenizer=bert_dataset.tokenizer),
    )
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = {k: v.to('cuda:0') for k, v in batch.items()}
            outputs = text_model(batch, use_sigmoid=True)
            paths_scores.extend(outputs.squeeze().cpu().numpy().tolist()) 
    
    def get_next_idx(idx):
        if idx == len(triple2path_idx_start) - 1:
            return len(paths_scores)
        else:
            return triple2path_idx_start[idx + 1]
    
    # paths_scores -> candidates_scores
    for i in tqdm(range(len(triple2path_idx_start))):
        if get_next_idx(i) == triple2path_idx_start[i]:
            candidates_scores.append(0)
            continue
        # TODO: try original model
        candidates_scores.append(max(paths_scores[
            triple2path_idx_start[i]:
            len(paths_scores) if i == len(triple2path_idx_start) - 1 else triple2path_idx_start[i + 1]
        ]))
    
    # candidates_scores -> ranks    
    for i in tqdm(range(0, len(candidates_scores), num_candidates)):
        group = candidates_scores[i: i + num_candidates]
        true_entity_score = group[0]
        if true_entity_score == 0:
            start = sorted(group, reverse=True).index(0) + 1
            rank = int((start + num_candidates) / 2)
        else: 
            after = sorted(group, reverse=True)
            rank = sorted(group, reverse=True).index(true_entity_score) + 1
            # while rank < len(after) - 1 and after[rank - 1] == after[rank]:
            #     print('rank:', rank, 'after[rank]:', after[rank])
            #     rank += 1
        ranks.append(rank)
        for j in hits_metric:
            if rank <= j:
                hits[j] += 1.0
                
    for rank in ranks:
        mrr += 1.0 / rank
    mrr /= len(ranks)
    
    for i in [1, 5, 10]:
        hits[i] = (hits[i] + 0.0) / len(ranks)
    
    print(f'hits@1: {hits[1]}, hits@5: {hits[5]}, hits@10: {hits[10]}, mrr: {mrr}')    
    return {
        'hits@1': hits[1],
        'hits@5': hits[5],
        'hits@10': hits[10],
        'mrr': mrr,
    } 
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='WN18RR')
    parser.add_argument('--num_hop', type=int, default=3)
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()
    kg = KG(os.path.join('kg', args.dataset_name))
    kg.load_test_triples()
    
    head_metrics = evaluate_by_crt_scores(
        kg.ranking_head,
        kg, 
        num_hop=args.num_hop, 
        dataset_name=args.dataset_name,
        model_path=args.model_path,
        type='head',
    )
    tail_metrics = evaluate_by_crt_scores(
        kg.ranking_tail,
        kg, 
        num_hop=args.num_hop, 
        dataset_name=args.dataset_name,
        model_path=args.model_path,
        type='tail',
    )
    
    avg_metrics = [(head_metrics[i] + tail_metrics[i]) / 2 for i in head_metrics.keys()]
    print(f'Average metrics: {avg_metrics}')
    