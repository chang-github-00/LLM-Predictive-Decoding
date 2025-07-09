import sys
import os
import numpy as np
import re
import warnings
import yaml
import json
import time
import math
import torch
import datasets
import argparse
import timeout_decorator
from dotenv import load_dotenv
from llm import load_llm
from algorithms import load_algorithm
from tqdm import tqdm
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM


from utils.math.math_utils import parse_question, parse_ground_truth, math_equal, call_with_timeout

model_name = '/root/huggingface/Meta-Llama-3-8B-Instruct/'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16).eval()


def _get_start_end_token_id(original_text, text, tokens):
    if text not in original_text:
        text = text.strip()
        if text not in original_text:
            return 0, -1
    cnt_length = [len(token) for token in tokens]
    cumulated_cnt_length = np.cumsum(cnt_length)
    index_start =  original_text.index(text)#processed action index in action
    index_end = index_start + len(text)
        
    token_start = np.argmax(cumulated_cnt_length >= index_start)
    token_end = np.argmax(cumulated_cnt_length >= index_end)
    if token_end < token_start:
        token_end = -1
    return token_start+1, token_end+1


def load_analyze_data(task, path='result/...'):
    examples = []
    path = os.path.join(path)
    with open(path, "r") as f:
        for line in f:
            js = json.loads(line)
            examples.append(js)
    if task == "humaneval": examples = examples[:164]
    return examples

def parse_annotation_file(path):
    f = open(path, "r")
    lines = f.readlines()
    wrong_lines = []
    start_parsing = False
    for line in lines:
        if '[f]' in line:
            start_parsing = True
            
        if start_parsing:
            wrong_lines.append(line)
            
    return wrong_lines

task = "gsm8k"
path = "results/run_anlaysis_important_results"
file_autoregressive = f"{path}/{task}_autoregressive_output.jsonl"
file_beamsearch = f"{path}/{task}_beamsearch_output.jsonl"
file_mpc = f"{path}/{task}_mpc_output.jsonl"

output_file = f"{path}/{task}_corrected_output_probs_per_step.jsonl"
autoregressive_data = load_analyze_data(task, file_autoregressive)
beamsearch_data = load_analyze_data(task, file_beamsearch)
mpc_data = load_analyze_data(task, file_mpc)

correct_autoregressive = np.array([data["success_rate"] for data in autoregressive_data])
correct_mpc = np.array([data["success_rate"] for data in mpc_data])
corrected_samples = correct_mpc & np.logical_not(correct_autoregressive)


all_output_probs = []

sample_50_correct_autoregressive = np.random.choice(np.where(correct_autoregressive)[0], 50, replace=False)

for i in range(len(autoregressive_data)):
    # if i > 100:
    #     break
    if i > 423:
        break
    if not corrected_samples[i] and not i in sample_50_correct_autoregressive:
        continue
    
    
    
    autoregressive_item = autoregressive_data[i]
    beamsearch_item = beamsearch_data[i]
    
    prompt = beamsearch_item["prompt"]
    answer_prefix = beamsearch_item["answer_prefix"]
    system_msg = beamsearch_item["system_msg"]
    
    success = bool(beamsearch_item["success_rate"])
    
    output = autoregressive_item["output"]
    step_probs = []
    
    all_output_lines = output.split("\n")
    all_output_lines = [line for line in all_output_lines if line.strip() not in answer_prefix]
    
    if corrected_samples[i]:
        path = f"results/run_anlaysis_important_results/gsm8k_prm/corrected_{i}.txt"
        wrong_lines = parse_annotation_file(path)

    for target in all_output_lines:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output}
        ]

        if target.strip() == "":
            continue
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors="pt"
        )[:,:-1] # no eot id
        # .to(model.device)

        tokens = [tokenizer.decode(i) for i in input_ids[0]]
        sentence = tokenizer.decode(input_ids[0])
        
        if answer_prefix in output:
            completion_after_prefix = output[output.index(answer_prefix)+len(answer_prefix):]
        else:
            completion_after_prefix = output

        start_id, _ = _get_start_end_token_id(sentence, completion_after_prefix, tokens)

        _, end_id = _get_start_end_token_id(sentence, target, tokens)

        input_ids = input_ids.to(model.device)
        target_ids = input_ids.clone()
        target_ids[:,:start_id] = -100
        target_ids[:,end_id:] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            probs = torch.exp(-outputs.loss)

        gt = True
        if corrected_samples[i]:
            for line in wrong_lines:
                if target in line:
                    gt = False
                    break
        step_probs.append((probs.item(), gt))  
        print(f"[EXP] {i+1} Prob of autoregressive: {probs.item()}, GT: {gt}")

    all_output_probs.append(step_probs)
    
    # print in one line
    # print(f"[EXP] {i+1} Prob of autoregressive: {outputs_probs[0]}, Prob of beamsearch: {outputs_probs[1]}, Prob of mpc: {outputs_probs[2]}")

with open(output_file, "w") as f:
    json.dump(all_output_probs, f)
