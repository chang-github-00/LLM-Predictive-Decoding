import sys
import os
import re
import warnings
import yaml
import json
import time
import math
import datasets
import argparse
import timeout_decorator
from dotenv import load_dotenv
from llm import load_llm
from algorithms import load_algorithm
from tqdm import tqdm
from typing import Optional

from utils.math.math_utils import parse_question, parse_ground_truth, math_equal, call_with_timeout
from utils.logging.token_logger import token_count, count_flag

def load_dataset(task, path='/root/huggingface/gsm8k'):
    if task == "gsm8k":
        full_dataset = datasets.load_dataset(path, 'main', split='test')
        dataset = [{"question": a["question"], "answer": a["answer"]} for a in full_dataset]
        return dataset
    
    if task == "gsm8k-subset":
        full_dataset = datasets.load_dataset(path, 'main', split='test')
        dataset = [{"question": a["question"], "answer": a["answer"]} for a in full_dataset]
        # keep every 7 questions
        indexes = list(range(0, len(dataset), 7))
        dataset = [dataset[i] for i in indexes]
        return dataset
    
    if task == "math":
        examples = []
        with open(path, "r") as f: 
            for line in f:
                js = json.loads(line)
                examples.append(js)
        
        dataset = []
        for example in examples:
            idx = example['idx']
            example['question'] = parse_question(example, "math")
            gt_cot, gt_ans = parse_ground_truth(example, "math")
            example = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'answer': gt_ans}
            dataset.append(example)  

        return dataset

def retrieve_answer_from_dataset(answer: str) -> str:
    return re.match(r'[\S\s]*#### (.*)$', answer)[1]
    
def evaluate_results(task, item, result): #result is a list
    def judge_gsm8k_answer(output: Optional[str], answer: str) -> bool:
        answer = str(answer)
        return answer in output

    if task == "gsm8k":
        answer = retrieve_answer_from_dataset(item["answer"])
        answer = answer.strip()
        if type(result) == str:
            return judge_gsm8k_answer(result, answer), result
        elif type(result) == list:
            count = dict()
            output_answer = dict()
            for res in result:
                match = re.search(r'answer is (\d+)', res)
                if match:
                    output = eval(match.group(1))
                    if output not in count:
                        count[output] = 1
                        output_answer[output] = [res]
                    else:
                        count[output] += 1
                        output_answer[output].append(res)
            if len(count) == 0:
                output = result[0]
            else:
                self_consistency_result = max(count, key=count.get)
                output = output_answer[self_consistency_result]
            return judge_gsm8k_answer(str(output), answer), output
        
    else:
        raise NotImplementedError
        

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

   
class EvalReasoning:
    def __init__(self,
                 task="dp",
                 run_config=None,
                 llm_config=None,
                 reward_llm_config=None,
                 algorithm_config=None,
                 ):
        
        self.llm = load_llm(llm_config.get("name", "gpt"), llm_config)
        self.reward_llm = load_llm(reward_llm_config.get("name", "gpt"), reward_llm_config)
        
        self.log_path = run_config["log_path"]
        self.task = task
        
        self.batch_size = run_config["batch_size"]
        self.algorithm = load_algorithm(algorithm_config["name"], algorithm_config, self.llm, reward_model=self.reward_llm)
        
        self.dataset = load_dataset(task+'-subset', run_config["data_path"])
        self.dataset_path = run_config["data_path"]
        # with open(algorithm_config["prompt_path"], 'r') as f:
        #     self.prompts = json.load(f)
        self.prompts = {}
        if self.task == "gsm8k":
            from prompts.Reasoning.gsm8k_prompt import cot_prompt
            self.prompts["prompt"] = cot_prompt #code_prompt#pal_prompt
            self.prompts["system_msg"] = "You will solve math problems following examples."
        
    def evaluate(self):
        
        dataset_name = os.path.basename(self.dataset_path).split(".")[0]
        result = []
        
        # create a new empty file for logging
        f = open(os.path.join(self.log_path,f"{self.task}_{dataset_name}.txt"), "w")
            
        for id, item in tqdm(enumerate(self.dataset), total=len(self.dataset)):

            question = item["question"]
            # print(question)
            success, output = self.algorithm.run(question, prompts=self.prompts, end_suffix="answer")
            
            if success:
                evaluation, final_output = evaluate_results(self.task, item, output)
                result.append(evaluation)
            else:
                evaluation = None  
            if type(output) == list: output = "\n".join(output) 
            output = output + "\n Voted result: " + str(final_output)
            with open(os.path.join(self.log_path,f"{self.task}_{dataset_name}.txt"), "a+") as f:
                if self.task == "math":
                    answer = item["answer"]
                elif self.task == "gsm8k":
                    answer = retrieve_answer_from_dataset(item["answer"])
                f.write(f"[EXP] {id}: [success_rate]: {evaluation}, [answer]: {answer}, [output]: {output}\n")

        
        metrics = {"task":self.task+'_'+dataset_name, "success_rate": sum(result) / len(result)}
        
        with open(os.path.join(self.log_path,f"all_results.txt"), "a+") as f:
            f.write(json.dumps(metrics) + "\n")
        
        return metrics
    
    def parallel_evaluate(self):
        
        dataset_name = os.path.basename(self.dataset_path).split(".")[0]
        result = []
        
        # create a new empty file for logging
        f = open(os.path.join(self.log_path,f"{self.task}_{dataset_name}.txt"), "w")
        
        log_file = open(os.path.join(self.log_path,f"{self.task}_{dataset_name}_generation_output.txt"), "w")
        
        item_iter = Iterator(self.dataset, self.batch_size)
        
        id = 0
        
        for test_items in tqdm(item_iter, total=math.ceil(len(self.dataset)/self.batch_size)):
            if count_flag:
                token_count.add_instance(len(test_items))
                
            questions = [item["question"] for item in test_items]
            success, all_outputs = self.algorithm.parallel_run(questions, prompts=self.prompts, end_suffix="answer") # process all questions in parallel

            # for output in all_outputs:
            #     log_line = json.dumps(output) + "\n"
            #     log_file.write(log_line)
            
            assert len(all_outputs) == len(test_items)
            for batch_id, item in tqdm(enumerate(test_items), total=len(test_items)):
                output = all_outputs[batch_id]
                try:
                    evaluation, final_output = evaluate_results(self.task, item, output)
                except Exception as e:
                    evaluation = False  
                    
                if self.task == "math":
                    answer = item["answer"]
                elif self.task == "gsm8k":
                    answer = retrieve_answer_from_dataset(item["answer"])
                if type(output) == list: 
                    output = "\n".join(output) 
                    output = output + "\n Voted result: " + str(final_output)
                
                
                id += 1
                
                if "idx" in item:
                    index = item["idx"]
                else:
                    index = id
                    
                count_stats = token_count.get_count()
                    
                f.write(f"[EXP] {index}: [success_rate]: {evaluation}, [answer]: {answer}, [output]: {output}\n")
                result.append(evaluation)
            
            if count_flag:
                token_count.print()
                f.write(f"[FLOPS] {count_stats}\n")

        
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
    parser.add_argument("--reward_model", required=True ,help="specify the models, available models are stated in the configuration file")
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
    reward_model_config = llm_config[args.reward_model]
    llm_config = llm_config[args.model]
    
    
    check_log_paths_are_ready(run_config["log_path"])
    
    # save the configuration file
    with open(os.path.join(run_config["log_path"], "config.yaml"), "w") as f:
        yaml.dump({"llm":llm_config, "reward_llm": reward_model_config, "algorithm":algorithm_config, "run":run_config}, f)
    
    eval_reasoning = EvalReasoning(task, run_config, llm_config, reward_model_config, algorithm_config)
    
    for i in range(3):
        metrics = eval_reasoning.parallel_evaluate()
    
    
if __name__ == "__main__":
    main()