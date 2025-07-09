import pdb

from common.registry import registry
# from rouge import Rouge
import json
import random
import re
import io
import argparse
import numpy as np

from utils.logging.token_logger import token_count, count_flag

@registry.register_algorithm("Self_Infill")
class Self_Infill:  # only support humaneval and mbpp
    def __init__(self,
                 llm_model,
                 prompt_path=None,
                 beam_temperature=0.7,
                 n_generate_sample=8,
                 do_sample=True,
                 beam_search=False,
                 num_iters=4,
                 task="gsm8k"
                 ):
        
        self.llm_model = llm_model
        
        if prompt_path is not None:
            self.prompts = json.load(open(prompt_path, 'r'))
        else:
            self.prompts = {}
        
        self.task = task
        
        self.do_sample = do_sample
        self.beam_temperature = beam_temperature
        self.n_generate_sample = n_generate_sample
        self.beam_search = beam_search
        self.num_iters = num_iters
        
    def format_humaneval_completion(self, code, prefix):
        all_prompt_lines = prefix.split("\n")
        all_lines = code.split("\n")
        cleaned_lines = []
        # all line before def solution(): should be removed
        for line in all_lines:
            complete_line = False  
            for prompt_line in all_prompt_lines:
                if line.strip() in prompt_line:
                    complete_line = True
                    break
            if not complete_line:
                line = line.lstrip('\n')
                line = line.replace("`", "")
                cleaned_lines.append(line)
                # if "return" in line:
                #     break
                
        with io.StringIO() as f:
            f.write(f"{prefix}")
            for a  in cleaned_lines:
                if a is not None:
                    f.write(f"{a}\n")

            full_output = f.getvalue()
            return full_output
        
    def run(self, question, prompts=None, **kwargs):
        
        raise NotImplementedError("This method is not implemented")
    
    def record_generated_num_tokens(self, results):
        if not count_flag:
            return
        
        for res in results:
            for result in res:
                token = self.llm_model.tokenizer.encode(result)
                token_count.add_generation_tokens(len(token))
            
    def parallel_run(self, questions, prompts=None, **kwargs):
        
        args = {
            "n_generate_sample":self.n_generate_sample,
            "max_tokens": 500,
            "temperature": self.beam_temperature,
            "top_p": 1.0,
            "stop": [],            
            "logprobs": 0,
            "use_beam_search": self.beam_search
        }
        
        args = argparse.Namespace(**args)
        
        
        generation_config = {"n": args.n_generate_sample, 
                            "stop": args.stop, 
                            "top_p": args.top_p,
                            "max_tokens": args.max_tokens, 
                            "temperature": args.temperature,
                            "do_sample": True,
                            "logprobs": args.logprobs,
                            "use_beam_search": args.use_beam_search
                            }
        
        
        if prompts is not None:
            self.prompts = prompts
        
        
        prompts = questions
        answer_prefixes = self.prompts["prompt"]
        suffixes = [""] * len(prompts)
        all_system_messages = [self.prompts["system_msg"]] * len(prompts)
        
        for iter in range(self.num_iters):
            
            all_prompts = []
            for prompt, suffix in zip(prompts, suffixes):
                input_prompt = prompt + "Fill in the code below:\n" + suffix
                all_prompts.append(input_prompt)
        
            success, all_code_samples = self.llm_model.parallel_generate_with_config(all_system_messages, all_prompts, generation_config, answer_prefixes)
            
            if not success:
            
                return False, None
            
            if args.n_generate_sample == 1: 
                all_code_samples = [[code_sample] for code_sample in all_code_samples]
            
            self.record_generated_num_tokens(all_code_samples)
        
            # split in half and keep the second half
            suffixes = []
            for id in range(len(questions)):
                code_sample = all_code_samples[id][0]
                half = len(code_sample.split("\n")) // 2
                suffix = "\n".join(code_sample.split("\n")[half:])
                
                with io.StringIO() as f:
                    f.write(answer_prefixes[id])
                    f.write("...\n")
                    f.write(suffix)
                    suffix = f.getvalue()
                suffixes.append(suffix)
                
        
        all_outputs = []
        for prefix, code_samples in zip(self.prompts["prompt"], all_code_samples):
            
            formatted_code_samples = []
            for code in code_samples:
                
                code = self.format_humaneval_completion(code, prefix)
                formatted_code_samples.append(code)
                
            all_outputs.append(formatted_code_samples)
        
        return True, all_outputs
    
                
                
    @classmethod
    def from_config(cls, llm_model, config):
        return cls(llm_model, 
                   prompt_path=config.get("prompt_path", None),
                   beam_temperature=config.get("beam_temperature", 0.7),
                   n_generate_sample=config.get("n_generate_sample", 8),
                   do_sample=config.get("do_sample", True),  
                   task=config.get("task", "gsm8k"),
                   beam_search=config.get("beam_search", False),
                   num_iters=config.get("num_iters", 2)
                   )
        