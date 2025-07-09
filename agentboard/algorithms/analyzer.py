import pdb

from common.registry import registry
# from rouge import Rouge
import json
import random
import re
import io
import numpy as np
import argparse
import torch

@registry.register_algorithm("Logprob_Analyzer")

class Logprob_Analyzer:
    def __init__(self, 
                 llm_model, 
                 prompt_path=None):
        self.llm_model = llm_model
    
    def parallel_run(self, system_msgs, prompts, completions, answer_prefixes=None):
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
        
        # get logp of completion given prompt
        before_generation_prefixes = answer_prefixes
        all_metrics = []
        success, all_logprobs = self.llm_model.encode(system_msgs, prompts, answer_prefixes=completions)
        
        for i, logprobs_ in enumerate(all_logprobs):
            metric = {}
            tokens = logprobs_["tokens"]
            logprobs = logprobs_["logprobs"]
            # first calculate the cumulative probability of the entire completion
            completion_prob = np.exp(sum(logprobs)) 
            completion_prob = completion_prob ** (1/len(tokens))
            metric["completion_prob"] = completion_prob
            # then calculate the probability of the completion given the prompt
            prompt = "".join(tokens)
            
            full_completion = completions[i]
            prefix = before_generation_prefixes[i]
            if prefix in full_completion:
                completion_after_prefix = full_completion[full_completion.index(prefix)+len(prefix):]
            else:
                completion_after_prefix = full_completion
            start_id, end_id = _get_start_end_token_id(prompt, completion_after_prefix, tokens)
            start_generation_id = start_id
            if end_id != -1:
                answer_logprob = np.exp(sum(logprobs[start_id:end_id]))
                answer_logprob = answer_logprob ** (1/(end_id-start_id))
            else:
                answer_logprob = completion_prob
            metric["answer_logprob"] = answer_logprob
            
            # finally divide the completion line by line and calculate the probability until the end of each line
            lines = completion_after_prefix.split("\n")
            lines = [line for line in lines if line.strip() != ""]
            for j, line in enumerate(lines):
                start_id, end_id = _get_start_end_token_id(prompt, line, tokens)
                if end_id != -1:
                    line_logprob = np.exp(sum(logprobs[start_generation_id:end_id]))
                    line_logprob = line_logprob ** (1/(end_id-start_generation_id))
                else:
                    line_logprob = answer_logprob
                num_lines_left = j - len(lines)
                metric[f"line_{num_lines_left}_logprob"] = line_logprob
            
            all_metrics.append(metric)
            
        return True, all_metrics
    
    def from_config(llm_model, config):
        return Logprob_Analyzer(llm_model)
    