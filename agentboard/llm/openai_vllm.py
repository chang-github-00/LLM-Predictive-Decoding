# from vllm import LLM, SamplingParams
import pdb
import sys
import numpy as np
sys.path.append('.')
from common.registry import registry
from prompts.prompt_template import prompt_templates
from typing import Optional, Union, Literal
import torch
import re
import copy
from openai import OpenAI
import time
import openai
import os
import tiktoken
from transformers import AutoTokenizer


@registry.register_llm("openai_vllm")
class OPENAI_VLLM:
    def __init__(self,
                 model,
                 temperature=0,
                 max_tokens=100,
                 system_message="You are a helpful assistant.",
                 use_azure=True,
                 top_p=1,
                 stop='\n',
                 retry_delays=5, # in seconds
                 max_retry_iters=5,
                 context_length=4096,
                 ):
        
        
        self.engine =  model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.use_azure = use_azure
        self.top_p = top_p
        self.stop = stop
        self.retry_delays = retry_delays
        self.max_retry_iters = max_retry_iters
        self.context_length = context_length
        
        self.client = OpenAI(
            api_key = "EMPTY",
            base_url = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1"),
            # organization='',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.engine)
                
    def chat_inference(self, messages):

        response = self.client.chat.completions.create(
            model=self.engine,
            messages=messages,
            stop = self.stop,
            temperature = self.temperature,
            max_tokens = self.max_tokens,
        )

        return response.choices[0].message.content
    
    def chat_inference_with_config(self, messages, config):
        stop = config.get("stop", self.stop)
        temperature = config.get("temperature", self.temperature)
        max_tokens = config.get("max_tokens", self.max_tokens)
        n = config.get("n", 1)
        
        stop = config.get("stop", self.stop)
        temperature = config.get("temperature", self.temperature)
        max_tokens = config.get("max_tokens", self.max_tokens)
        n = config.get("n", 1)
        top_logprobs = config.get("logprobs", 5)
        do_sample = config.get("do_sample", True)
        top_p = config.get("top_p", 1)
        
        logprobs = top_logprobs > 0
        
        
        response = self.client.chat.completions.create(
                        model=self.engine,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=n,
                        stop=stop,
                        top_logprobs=top_logprobs,
                        logprobs=logprobs,
        )
        if logprobs == 0:
            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
        else:
            
            # return a list of {"text":xxx, "logprobs":xxx, "tokens": xxx}
            
            outputs = []
            for i in range(n):
                choice = response.choices[i]
                item = {}
                item["text"] = choice.message.content
                raw_log_prob = choice.logprobs.content
                item["logprobs"] = [token.logprob for token in raw_log_prob]
                item["tokens"] = [token.token for token in raw_log_prob]
                outputs.append(item)
            
            if n == 1:
                return outputs[0]
            else:
                return outputs
    
    def generate(self, system_message, prompt, answer_prefix=None):
        prompt=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
        ]
        if answer_prefix is not None:
            prompt.append({"role": "assistant", "content": answer_prefix})
        for attempt in range(self.max_retry_iters):  
            try:
                output = self.chat_inference(prompt)
                # output = output.split("\n")[0]
                return True, output # return success, completion
            except Exception as e:
                print(f"Error on attempt {attempt + 1}") 
                if attempt < self.max_retry_iters - 1:  # If not the last attempt
                    time.sleep(self.retry_delays)  # Wait before retrying
                
                else:
                    print("Failed to get completion after multiple attempts.")
                    # raise e
                    
        return False, None
    
    def encode(self, prompt): # to get logprobs
        response = self.client.completions.create(
            model=self.engine,
            prompt=prompt,
            max_tokens=1,
            logprobs=20,
            echo=True
        )
        
        all_logprobs = []
        all_tokens = []
        for res in response.choices:
            logprobs = res.logprobs.top_logprobs
            tokens = res.logprobs.tokens
            all_logprobs.append(logprobs)
            all_tokens.append(tokens)
        return all_logprobs, all_tokens
            
    
    def generate_with_config(self, system_message, prompt, config):
        prompt=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
        ]
        for attempt in range(self.max_retry_iters):  
            try:
                output = self.chat_inference_with_config(prompt, config)
                # output = output.split("\n")[0]
                return True, output # return success, completion
            except Exception as e:
                print(f"Error on attempt {attempt + 1}") 
                if attempt < self.max_retry_iters - 1:  # If not the last attempt
                    time.sleep(self.retry_delays)  # Wait before retrying
                
                else:
                    print("Failed to get completion after multiple attempts.")
                    # raise e
        return False, None
    
    
    def num_tokens_from_messages(self, messages):
        # prompt = self.tokenizer.encode(self.tokenizer.apply_chat_template(messages))
        prompt = ""
        for message in messages:
            prompt += message["role"] + ": " + message["content"] + "\n"
        prompt = self.tokenizer.encode(prompt)
        
        return len(prompt)+100
    
    def num_tokens(self, prompt):
        return len(self.tokenizer.encode(prompt))

    @classmethod
    def from_config(cls, config):

        engine = config.get("engine", "gpt-35-turbo")
        temperature = config.get("temperature", 0)
        max_tokens = config.get("max_tokens", 100)
        top_p = config.get("top_p", 1)
        stop = config.get("stop", ["\n"])
        context_length = config.get("context_length", 4096)
        retry_delays = config.get("retry_delays", 5)
        max_retry_iters = config.get("max_retry_iters", 5)
        system_message = config.get("system_message", "You are a helpful assistant.")

        return cls(
            engine,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            context_length=context_length,
            retry_delays=retry_delays,
            max_retry_iters=max_retry_iters,
            system_message=system_message
        )