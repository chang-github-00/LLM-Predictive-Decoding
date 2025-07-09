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


task = "math" #"gsm8k"
path = "results/run_anlaysis_important_results"
file_autoregressive = f"{path}/{task}_autoregressive_output.jsonl"
file_beamsearch = f"{path}/{task}_beamsearch_output.jsonl"
file_mpc = f"{path}/{task}_mpc_output.jsonl"

output_file = f"{path}/{task}_output_probs.jsonl"
autoregressive_data = load_analyze_data(task, file_autoregressive)
beamsearch_data = load_analyze_data(task, file_beamsearch)
mpc_data = load_analyze_data(task, file_mpc)

all_output_probs = []
for i in range(len(autoregressive_data)):
    # if i > 100:
    #     break
    autoregressive_item = autoregressive_data[i]
    beamsearch_item = beamsearch_data[i]
    mpc_item = mpc_data[i]
    
    prompt = autoregressive_item["prompt"]
    answer_prefix = autoregressive_item["answer_prefix"]
    system_msg = autoregressive_item["system_msg"]
    
    success = bool(autoregressive_item["success_rate"])
    
    outputs = [autoregressive_item["output"], beamsearch_item["output"], mpc_item["output"]]
    outputs_probs = []

    for output in outputs:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output}
        ]


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

        start_id, end_id = _get_start_end_token_id(sentence, completion_after_prefix, tokens)

        input_ids = input_ids.to(model.device)
        target_ids = input_ids.clone()
        target_ids[:,:start_id] = -100
        target_ids[:,end_id:] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            probs = torch.exp(-outputs.loss)


        outputs_probs.append(probs.item())  

    all_output_probs.append(outputs_probs)
    
    # print in one line
    print(f"[EXP] {i+1} Prob of autoregressive: {outputs_probs[0]}, Prob of beamsearch: {outputs_probs[1]}, Prob of mpc: {outputs_probs[2]}")

with open(output_file, "w") as f:
    json.dump(all_output_probs, f)
