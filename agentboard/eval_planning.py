import pdb
import sys
import os
import re
import wandb
import warnings
import yaml
import json
import time
import argparse
from dotenv import load_dotenv
from llm import load_llm
from algorithms import load_algorithm
from tqdm import tqdm

def load_dataset(task, path):
    if task == "path_finding":
        dataset = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                edges = ", ".join([f"({x[0]}, {x[1]})" for x in item["edges"]])
                start, end = item["start_end"]
                question_prompt = f"Given a graph where each node is represented by an integer. Undirected edge between nodes i and j is represented by a tuple (i, j). Given all the edges {edges}, generate a path from node {start} to {end}. Represent the path with a list of integers [{start}, ..., {end}]."
                
                item["question"] = question_prompt
                dataset.append(item)
        return dataset[:200]
    
    if task == "dp":
        dataset = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                numbers = item["raw_input"]   
                # question_prompt = f"Given a sequence of integers, find a subsequence with the highest sum, such that no two numbers in the subsequence are adjacent in the original sequence.\n\nOutput a list with \"1\" for chosen numbers and \"2\" for unchosen ones. If multiple solutions exist, select the lexicographically smallest. input = {numbers} "
                question_prompt = f"input = {numbers}"
                item["question"] = question_prompt
                dataset.append(item)
        return dataset[:200]
        
def evaluate_results(task, item, result): #result is a list
    if task == "path_finding":
        edges = item["edges"]
        start, end = item["start_end"]
        if result[0]!=start or result[-1]!=end:
            return False
        for i in range(len(result) - 1):
            if (result[i], result[i+1]) not in edges and (result[i+1], result[i]) not in edges:
                return False
        return True
    
    if task == "dp":
        # ground_truth = item["raw_output"]
        # if ground_truth != result:
        #     return False
        mapping = {1: 1, 0: 2, 2: 2}
        result = [mapping[x] for x in result if x in mapping]
        
        # check all neighbor results are not both 1
        for i in range(len(result) - 1):
            if result[i] == 1 and result[i+1] == 1:
                return False
            
        # check the sum of the chosen numbers is the largest
        numbers = item["raw_input"]
        ground_truth = item["raw_output"]
        sum_result = sum([numbers[i] for i in range(len(numbers)) if result[i] == 1])
        sum_ground_truth = sum([numbers[i] for i in range(len(numbers)) if ground_truth[i] == 1])
        if sum_result != sum_ground_truth:
            return False
        return True
    
class EvalPlanning:
    def __init__(self,
                 task="dp",
                 run_config=None,
                 llm_config=None,
                 algorithm_config=None,
                 ):
        
        self.llm = load_llm(llm_config.get("name", "gpt"), llm_config)
        
        self.log_path = run_config["log_path"]
        self.task = task
        
        self.algorithm = load_algorithm(algorithm_config["name"], algorithm_config, self.llm)
        
        self.dataset = load_dataset(task, run_config["data_path"])
        self.dataset_path = run_config["data_path"]
        with open(algorithm_config["prompt_path"], 'r') as f:
            self.prompts = json.load(f)
    
    def evaluate(self):
        
        dataset_name = os.path.basename(self.dataset_path).split(".")[0]
        result = []
        for id, item in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            question = item["question"]
            # print(question)
            success, output = self.algorithm.run(question)
            if success:
                evaluation = evaluate_results(self.task, item, output)
                result.append(evaluation)
            else:
                evaluation = None   
            
            with open(os.path.join(self.log_path,f"{self.task}_{dataset_name}.txt"), "a+") as f:
                f.write(f"[EXP] {id}: [success_rate]: {evaluation}, [output]: {output} \n")

        
        metrics = {"task":self.task+'_'+dataset_name, "success_rate": sum(result) / len(result)}
        
        with open(os.path.join(self.log_path,f"all_results.txt"), "a+") as f:
            f.write(json.dumps(metrics) + "\n")
        
        return metrics
    

def parse_args():
    parser = argparse.ArgumentParser(description="Testing")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--tasks", required=True, type=str, help="specify the tasks")
    parser.add_argument("--algorithm", required=True, type=str, help="specify the algorithm")
    parser.add_argument("--model", required=True ,help="specify the models, available models are stated in the configuration file")
    parser.add_argument("--log_path", required=False, default='', help="specify the place to store the resuls")
    parser.add_argument("--data_path", required=False, default='', help="specify the test data file")
    parser.add_argument("--prompt_path", required=False, default='', help="specify the prompt")
    parser.add_argument("--lookahead_length", type=int, default=3, help="random seed")
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
    if args.prompt_path != '':
        algorithm_config["prompt_path"] = args.prompt_path
    algorithm_config["lookahead_length"] = args.lookahead_length
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
    
    eval_planning = EvalPlanning(task, run_config, llm_config, algorithm_config)
    
    metrics = eval_planning.evaluate()
    
if __name__ == "__main__":
    main()