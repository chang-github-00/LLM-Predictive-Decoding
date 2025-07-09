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

@registry.register_algorithm("Generation")
class Generation:  # the agent should receive goal, state and action, then return the next state
    def __init__(self,
                 llm_model,
                 prompt_path
                 ):
        
        self.llm_model = llm_model
        
        self.prompts = json.load(open(prompt_path, 'r'))
    
    def make_prompt(self, question, prompt):
        query = ""
        query += prompt["instruction"] + '\n'
        query += "\nHere is an example:\n" + prompt["examples"][0] + '\n'
        
        input_prompt = query + question
        input_prompt = input_prompt + "\nAnswer:"

        return input_prompt

    def parse_integer_lists(self, string):
        pattern = r'\[([\d\s,]+)\]'  # Regular expression pattern to match integer lists
        matches = re.findall(pattern, string)  # Find all matches

        # Parse each match
        result = []
        for match in matches:
            integers = re.findall(r'\d+', match)  # Find all integers in the match
            integers = [int(num) for num in integers]  # Convert strings to integers
            result.append(integers)

        return result
    def run(self, question):
        
        system_message = self.prompts["system_msg"]
        input_prompt = self.make_prompt(question, self.prompts)
        success, results = self.llm_model.generate( system_message, input_prompt)
        
        if success:
            result_lists = self.parse_integer_lists(results)
            if len(result_lists) > 0:
                return True, result_lists[-1]
        return False, None

    @classmethod
    def from_config(cls, llm_model, config):
        return cls(llm_model, config["prompt_path"])
    

@registry.register_algorithm("Self_Consistency")
class Self_Consistency:  # the algorithm should be stateless, and generates a whole plan / code / chain of actions at once.
    def __init__(self,
                 llm_model,
                 prompt_path=None,
                 beam_temperature=0.7,
                 n_generate_sample=8,
                 do_sample=True,
                 beam_search=False,
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
        
        
    def make_prompt(self, prompt):
        if self.task == "gsm8k":
            with io.StringIO() as f:
                f.write(prompt)
                f.write("\n\n\n\n\n")
                # f.write(f'Q: {self.example}\n\n# solution in Python:\n\n\ndef solution():\n    """{self.example}"""\n')
                f.write(f'Solve this problem following previous examples:\nQ: {self.prompts["question"]}\n\n# Finish the solution in Python:\n\n\ndef solution():\n')

                # get the prompt
                model_input = f.getvalue()
            return model_input
        
        if self.task == "math":
            with io.StringIO() as f:
                f.write(prompt)
                f.write("\n\n")
                f.write(f'Solve this problem following previous examples:\nQ: {self.prompts["question"]}\n# solution in Python:\n')
                model_input = f.getvalue()
            return model_input
    
    def format_code(self, code):
        # one typical format is ```...```, parse the code inside
        # use re to find the code
        code_blocks = re.findall(r'```(.*?)```', code, re.DOTALL)
        if len(code_blocks) > 0:
            return code_blocks[0]
        
        # another possible format is start generation after def solution():
        all_lines = code.split("\n")
        cleaned_lines = []
        # all line before def solution(): should be removed
        start_line = False  
        for line in all_lines:
            if "def solution():" in line:
                start_line = True
            elif start_line:
                line = line.lstrip('\n')
                line = line.replace("`", "")
                cleaned_lines.append(line)
                if "return" in line:
                    break
                
        with io.StringIO() as f:
            f.write("def solution():\n")
                # iterate through the state
            for a  in cleaned_lines:
                if a is not None:
                    f.write(f"{a}\n")

            full_output = f.getvalue()
            return full_output
        
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
    
    def record_generated_num_tokens(self, results):
        if not count_flag:
            return
        
        for res in results:
            for result in res:
                token = self.llm_model.tokenizer.encode(result)
                token_count.add_generation_tokens(len(token))
                
    def run(self, question, prompts=None, **kwargs):
        
        if prompts is not None:
            self.prompts = prompts
        self.prompts["question"] = question
        
        self.trajectory_pool = []
        
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
        
        if self.task == "humaneval":
            input_prompt = question
            answer_prefix = self.prompts["prompt"]
        else:
            input_prompt = self.make_prompt(self.prompts["prompt"])
            answer_prefix = None
        
        system_message = self.prompts["system_msg"]
        success, code_samples = self.llm_model.generate_with_config(system_message, input_prompt, generation_config, answer_prefix)
        
        if not success:
            
            return False, None
        
        if args.n_generate_sample == 1: code_samples = [code_samples]
        
        all_outputs = []
        for code in code_samples:
            
            code = self.format_code(code)
        
            all_outputs.append(code)
        return True, all_outputs
            
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
        
        all_prompts = []
        if self.task in ["humaneval", "mbpp"]:
            all_prompts = questions
            answer_prefixes = self.prompts["prompt"]
        else:
            for i in range(len(questions)):
                self.prompts["question"] = questions[i]
                input_prompt = self.make_prompt(self.prompts["prompt"])
                all_prompts.append(input_prompt)
            answer_prefixes = None

        all_system_messages = [self.prompts["system_msg"]] * len(all_prompts)
        
        success, all_code_samples = self.llm_model.parallel_generate_with_config(all_system_messages, all_prompts, generation_config, answer_prefixes)
            
        if not success:
            
            return False, None
        
        if args.n_generate_sample == 1: all_code_samples = [[code_sample] for code_sample in all_code_samples]

        self.record_generated_num_tokens(all_code_samples)
        
        if self.task in ["humaneval", "mbpp"]:
            all_outputs = []
            for prefix, code_samples in zip(self.prompts["prompt"], all_code_samples):
                
                formatted_code_samples = []
                for code in code_samples:
                    
                    code = self.format_humaneval_completion(code, prefix)
                    formatted_code_samples.append(code)
                    
                all_outputs.append(formatted_code_samples)
                
            return True, all_outputs
        else:   
            all_outputs = []
            for code_samples in all_code_samples:
                
                formatted_code_samples = []
                for code in code_samples:
                    code = self.format_code(code)
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
                   beam_search=config.get("beam_search", False)
                   )


@registry.register_algorithm("Self_Consistency_old")
class Self_Consistency_old:  # the algorithm should be stateless, and generates a whole plan / code / chain of actions at once.
    def __init__(self,
                 llm_model,
                 prompt_path=None,
                 beam_temperature=0.7,
                 n_generate_sample=8,
                 do_sample=True,
                 beam_search=False,
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
        
        
    def make_prompt(self, prompt):
        if self.task == "gsm8k":
            with io.StringIO() as f:
                f.write(prompt)
                f.write("\n\n\n\n\n")
                # f.write(f'Q: {self.example}\n\n# solution in Python:\n\n\ndef solution():\n    """{self.example}"""\n')
                f.write(f'Solve this problem following previous examples:\nQ: {self.prompts["question"]}\n\n# Finish the solution in Python:\n\n\ndef solution():\n')

                # get the prompt
                model_input = f.getvalue()
            return model_input
        
        if self.task == "math":
            with io.StringIO() as f:
                f.write(prompt)
                f.write("\n\n")
                f.write(f'Solve this problem following previous examples:\nQ: {self.prompts["question"]}\n# solution in Python:\n')
                model_input = f.getvalue()
            return model_input
    
    def format_code(self, code):
        # one typical format is ```...```, parse the code inside
        # use re to find the code
        code_blocks = re.findall(r'```(.*?)```', code, re.DOTALL)
        if len(code_blocks) > 0:
            return code_blocks[0]
        
        # another possible format is start generation after def solution():
        all_lines = code.split("\n")
        cleaned_lines = []
        # all line before def solution(): should be removed
        start_line = False  
        for line in all_lines:
            if "def solution():" in line:
                start_line = True
            elif start_line:
                line = line.lstrip('\n')
                line = line.replace("`", "")
                cleaned_lines.append(line)
                if "return" in line:
                    break
                
        with io.StringIO() as f:
            f.write("def solution():\n")
                # iterate through the state
            for a  in cleaned_lines:
                if a is not None:
                    f.write(f"{a}\n")

            full_output = f.getvalue()
            return full_output
        
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
                if "return" in line:
                    break
                
        with io.StringIO() as f:
            f.write(f"{prefix}")
            for a  in cleaned_lines:
                if a is not None:
                    f.write(f"{a}\n")

            full_output = f.getvalue()
            return full_output
        
    def run(self, question, prompts=None, **kwargs):
        
        if prompts is not None:
            self.prompts = prompts
        self.prompts["question"] = question
        
        self.trajectory_pool = []
        
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
        
        if self.task == "humaneval":
            input_prompt = question
            answer_prefix = self.prompts["prompt"]
        else:
            input_prompt = self.make_prompt(self.prompts["prompt"])
            answer_prefix = None
        
        system_message = self.prompts["system_msg"]
        success, code_samples = self.llm_model.generate_with_config(system_message, input_prompt, generation_config, answer_prefix)
        
        if not success:
            
            return False, None
        
        if args.n_generate_sample == 1: code_samples = [code_samples]
        
        all_outputs = []
        for code in code_samples:
            
            code = self.format_code(code)
        
            all_outputs.append(code)
        return True, all_outputs
    
    
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
        
        all_prompts = []
        if self.task == "humaneval":
            all_prompts = questions
            answer_prefixes = self.prompts["prompt"]
        else:
            for i in range(len(questions)):
                self.prompts["question"] = questions[i]
                input_prompt = self.make_prompt(self.prompts["prompt"])
                all_prompts.append(input_prompt)
            answer_prefixes = None

        all_system_messages = [self.prompts["system_msg"]] * len(all_prompts)
        
        success, all_code_samples = self.llm_model.parallel_generate_with_config(all_system_messages, all_prompts, generation_config, answer_prefixes)
                
        if not success:
            
            return False, None
        
        if args.n_generate_sample == 1: all_code_samples = [[code_sample] for code_sample in all_code_samples]
        
        if self.task == "humaneval":
            all_outputs = []
            for prefix, code_samples in zip(self.prompts["prompt"], all_code_samples):
                
                formatted_code_samples = []
                for code in code_samples:
                    
                    code = self.format_humaneval_completion(code, prefix)
                    formatted_code_samples.append(code)
                    
                all_outputs.append(formatted_code_samples)
            
            return True, all_outputs
        else:   
            all_outputs = []
            for code_samples in all_code_samples:
                
                formatted_code_samples = []
                for code in code_samples:
                    code = self.format_code(code)
                    formatted_code_samples.append(code)
                    
                all_outputs.append(formatted_code_samples)
            if count_flag:
                self.record_generated_num_tokens(all_outputs)
            return True, all_outputs
                
                
    @classmethod
    def from_config(cls, llm_model, config):
        return cls(llm_model, 
                   prompt_path=config.get("prompt_path", None),
                   beam_temperature=config.get("beam_temperature", 0.7),
                   n_generate_sample=config.get("n_generate_sample", 8),
                   do_sample=config.get("do_sample", True),  
                   task=config.get("task", "gsm8k"),
                   beam_search=config.get("beam_search", False)
                   )
        