import sys
import os
import re
import math
import warnings
import yaml
import json
import time
import datasets
import argparse
import timeout_decorator
from dotenv import load_dotenv
from llm import load_llm
from algorithms import load_algorithm
from tqdm import tqdm
from typing import Optional

from utils.human_eval.evaluation import evaluate_functional_correctness, parse_code_prefix
from utils.logging.token_logger import token_count, count_flag


os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def load_dataset(task, path=''):
    if task == "humaneval":
        dataset =  open(path).readlines()
        dataset = [json.loads(item) for item in dataset]
        return dataset   
    elif task == "mbpp":
        dataset =  open(path).readlines()
        dataset = [json.loads(item) for item in dataset]
        return dataset  
    else:
        raise ValueError("Task not supported")
    
class Iterator:
    def __init__(self, data, step=200):
        self.data = data
        self.index = 0
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        max_index = min(self.index+self.step, len(self.data))
        value = self.data[self.index:max_index]
        self.index += self.step
        return value

def entry_point(
    sample_file: str,
    problem_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    is_mbpp = "mbpp" in problem_file
    results = evaluate_functional_correctness(sample_file, k=k, n_workers=n_workers, timeout=timeout, problem_file=problem_file, is_mbpp=is_mbpp)
    results = {k:v*100 for k,v in results.items()}
    print(results)
    
    return results
    
   
class EvalReasoning:
    def __init__(self,
                 task="humaneval",
                 run_config=None,
                 llm_config=None,
                 algorithm_config=None,
                 ):
        
        self.llm = load_llm(llm_config.get("name", "gpt"), llm_config)
        
        self.log_path = run_config["log_path"]
        self.task = task
        
        self.batch_size = run_config["batch_size"]
        self.algorithm = load_algorithm(algorithm_config["name"], algorithm_config, self.llm)
        
        self.dataset = load_dataset(task, run_config["data_path"])
        self.dataset_path = run_config["data_path"]
        # with open(algorithm_config["prompt_path"], 'r') as f:
        #     self.prompts = json.load(f)
        self.prompts = {}
        if self.task in ["humaneval"]:
            self.prompts["system_msg"] = "Finish writing the python function. You will only write code blocks. Write # finish after the last line of the function." #"Finish writing the python function. You will only write code blocks."
        elif self.task in ["mbpp"]:
            self.prompts["system_msg"] = "Finish writing the python function. You will only write code blocks. Write # finish after the last line of the function."#"Finish writing the python function. You will only write code blocks. There should only be one return statement in the function."
        
    def evaluate(self):
        
        dataset_name = os.path.basename(self.dataset_path).split(".")[0]
        result = []
        
        # create a new empty file for logging
        f = open(os.path.join(self.log_path,f"{self.task}_{dataset_name}.txt"), "w")
        
        output_filepath = os.path.join(self.log_path,f"{self.task}_output.jsonl")
        
        for id, item in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            
            task_id = item["task_id"]
            if self.task == "humaneval":
                question = item["text"]
                self.prompts["prompt"] = item["prompt"]
            elif self.task == "mbpp":
                question = item["prompt"]
                self.prompts["prompt"] = item["code"]
            else:
                raise ValueError("Task not supported")
            # print(question)
            success, output = self.algorithm.run(question, prompts=self.prompts, end_suffix="return")
            
            assert success
            if type(output) == list: output = output[0]
            
            
            with open(output_filepath, "a+") as f:
                write_dict = {"task_id":item["task_id"], "generation":output, "prompt":item["prompt"]}
                f.write(json.dumps(write_dict) + '\n')
            
        res = entry_point(output_filepath, self.dataset_path)
        with open(os.path.join(self.log_path,f"all_results.txt"), "a+") as f:
            for key, value in res:
                metrics = {"task":self.task+'_'+dataset_name, key: value}
                f.write(json.dumps(metrics) + "\n")
        
        return res
    
    def parallel_evaluate(self):
        
        dataset_name = os.path.basename(self.dataset_path).split(".")[0]
        result = []
        
        # create a new empty file for logging
        f = open(os.path.join(self.log_path,f"{self.task}_{dataset_name}.txt"), "w")
        
        output_filepath = os.path.join(self.log_path,f"{self.task}_output.jsonl")
        
        # output_f = open(output_filepath, "w")
        # output_f.close()
        
        item_iter = Iterator(self.dataset, self.batch_size)
        
        id = 0
        
        for test_items in tqdm(item_iter, total=math.ceil(len(self.dataset)/self.batch_size)):
            if count_flag:
                token_count.add_instance(len(test_items))
                
            if self.task == "humaneval":
                prefixes = [item["prompt"] for item in test_items]
                questions = [item["text"] for item in test_items]
                self.prompts["prompt"] = prefixes
            elif self.task == "mbpp":
                prefixes = [parse_code_prefix(item["code"]) for item in test_items]
                questions = [item["prompt"] for item in test_items]
                self.prompts["prompt"] = prefixes
            else:
                raise ValueError("Task not supported")
            
            success, all_outputs = self.algorithm.parallel_run(questions, prompts=self.prompts, end_suffix="finish") # process all questions in parallel

            assert success
            
            assert len(test_items) == len(all_outputs)
            
            
            with open(output_filepath, "a+") as f:
                for id, item in enumerate(test_items):
                    output = all_outputs[id]
                    if type(output) == list: 
                        for out in output:
                            write_dict = item.copy()
                            write_dict["generation"] = out
                            f.write(json.dumps(write_dict) + '\n')
                    else:
                        write_dict = item.copy()
                        write_dict["generation"] = output
                        f.write(json.dumps(write_dict) + '\n')
            
            count_stats = token_count.get_count()
            token_count.print()
        
            
        res = entry_point(output_filepath, self.dataset_path)
        with open(os.path.join(self.log_path,f"all_results.txt"), "a+") as f:
            for key, value in res.items():
                metrics = {"task":self.task+'_'+dataset_name, key: value}
                f.write(json.dumps(metrics) + "\n")
            if count_flag:
                f.write(f"[FLOPS] {count_stats}\n")
        return res
    
def parse_args():
    parser = argparse.ArgumentParser(description="Testing")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--tasks", required=True, type=str, help="specify the tasks")
    parser.add_argument("--algorithm", required=True, type=str, help="specify the algorithm")
    parser.add_argument("--model", required=True ,help="specify the models, available models are stated in the configuration file")
    parser.add_argument("--log_path", required=False, default='', help="specify the place to store the resuls")
    parser.add_argument("--data_path", required=False, default='/root/huggingface/gsm8k', help="specify the test data file")
    parser.add_argument("--batch_size", required=False, default=50, type=int,help="number of problems processed together")
    # parser.add_argument("--prompt_path", required=False, default='', help="specify the prompt")
    args = parser.parse_args()

    return args


def path_constructor(loader, node):
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    ''' Extract the matched value, expand env variable, and replace the match '''
    value = node.value
    match = path_matcher.match(value)
    env_var = match.group()[2:-1]
    return os.environ.get(env_var) + value[match.end():]

def load_config(cfg_path, args):
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    yaml.add_implicit_resolver('!path', path_matcher)
    yaml.add_constructor('!path', path_constructor)
    with open(cfg_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    llm_config = config["llm"]
    algorithm_config = config["algorithm"]
    run_config = config["run"]
    algorithm_config["name"] = args.algorithm
    if args.log_path != '':
        run_config["log_path"] = args.log_path
    if args.data_path != '':
        run_config["data_path"] = args.data_path
    run_config["batch_size"] = args.batch_size
    # if args.prompt_path != '':
    #     algorithm_config["prompt_path"] = args.prompt_path
    return llm_config, algorithm_config, run_config

def check_log_paths_are_ready(log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
            
    return True


def main():
    load_dotenv()  # take environment variables from .env., load openai api key, tool key, wandb key, project path...

    args = parse_args()
    
    llm_config, algorithm_config, run_config = load_config(args.cfg_path, args) 
    
    task = args.tasks
    llm_config = llm_config[args.model]
    
    check_log_paths_are_ready(run_config["log_path"])
    
    # save the configuration file
    with open(os.path.join(run_config["log_path"], "config.yaml"), "w") as f:
        yaml.dump({"llm":llm_config, "algorithm":algorithm_config, "run":run_config}, f)
    
    eval_reasoning = EvalReasoning(task, run_config, llm_config, algorithm_config)
    
    for i in range(5):
        metrics = eval_reasoning.parallel_evaluate()
    
    
if __name__ == "__main__":
    main()