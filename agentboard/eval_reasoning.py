import sys
import os
import re
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

from utils.math.math_utils import parse_question, parse_ground_truth, math_equal, call_with_timeout

def load_dataset(task, path='/root/huggingface/gsm8k'):
    if task == "gsm8k":
        full_dataset = datasets.load_dataset(path, 'main', split='test')
        dataset = [{"question": a["question"], "answer": a["answer"]} for a in full_dataset]
        return dataset[6:]
    
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
        if output is None:
            return False
        try: 
            if isinstance(output, str):
                output = eval(output)
            if isinstance(answer, str):
                answer = eval(answer)
            if output < 0 and answer > 0:
                output = - output
            return abs(output-answer) < 0.01
        except:
            pass
        try:
            output = int(output)
            answer = int(answer)
            return output == answer
        except:
            pass
        try:
            output = float(output)
            answer = float(answer)
            return output == answer
        except:
            pass
        try:
            return output == answer
        except ValueError:
            return False

    if task == "gsm8k":
        answer = retrieve_answer_from_dataset(item["answer"])
        if type(result) == str:
            try:
                executed_solution = execute_solution(result, execute=True)
            except Exception as e:
                executed_solution = "Error: time out"
        elif type(result) == list:
            executed_solution = []
            for r in result:
                try:
                    executed_solution.append(execute_solution(r, execute=True))
                except Exception as e:
                    executed_solution.append("Error: time out")
            # majority vote
            count = dict()
            for a in executed_solution:
                if a not in count:
                    count[a] = 1
                else:
                    count[a] += 1
            executed_solution = max(count, key=count.get)
        return judge_gsm8k_answer(executed_solution, answer), executed_solution
    elif task == "math":
        answer = item["answer"]
        if type(result) == str:
            try:
                executed_solution = str(execute_solution(result, execute=True))
                if "=" in str(executed_solution):
                    executed_solution = executed_solution.split("=")[1].strip()
            except Exception as e:
                executed_solution = "Error: time out"
        elif type(result) == list:
            executed_solution = []
            for r in result:
                try:
                    result = str(execute_solution(r, execute=True))
                    if "=" in str(result):
                        result = result.split("=")[1].strip()
                    executed_solution.append(result)
                except Exception as e:
                    executed_solution.append("Error: time out")
            # majority vote
            count = dict()
            for a in executed_solution:
                if a not in count:
                    count[a] = 1
                else:
                    count[a] += 1
            executed_solution = max(count, key=count.get)
        return math_equal(executed_solution, answer), executed_solution
        
    else:
        raise NotImplementedError
        
@timeout_decorator.timeout(20, use_signals=False)
def execute_solution(code, execute=True):
    
    full_output = code
    if execute:
        try:
            # Create a dictionary to serve as the global and local scope for the exec call
            exec_globals = {}

            # Execute the function definition
            exec(full_output, exec_globals)

            # Call the function and get the output
            output = exec_globals['solution']()
            return output
        except Exception as e:
            # return the error message
            # try execute again without the function definition
            
            # first parse the return statement
            line_by_line_output = full_output.split("\n")
            return_statement = [a for a in line_by_line_output if "return" in a]
            if len(return_statement) > 0:
                return_statement = return_statement[0]
                # return_statement = re.match(r'return (.*)', return_statement).group(1)
                return_variable = return_statement[return_statement.find("return")+len("return"):].strip()
            
                # execute the code line by line 
                exec_globals = {}
                for a in line_by_line_output:
                    try:
                        exec(a.strip(), exec_globals)
                    except Exception as e:
                        pass
                if return_variable in exec_globals:
                    return exec_globals[return_variable]
        
            return "Error: return error, fail to execute"
    else:
        return full_output
    
      
class EvalReasoning:
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
        # with open(algorithm_config["prompt_path"], 'r') as f:
        #     self.prompts = json.load(f)
        self.prompts = {}
        if self.task == "gsm8k":
            from prompts.Reasoning.gsm8k_prompt import code_prompt, evaluate_prompt, pal_prompt
            self.prompts["prompt"] = pal_prompt #code_prompt#pal_prompt
            self.prompts["evaluate"] = evaluate_prompt
            self.prompts["system_msg"] = "You will write python program to solve math problems. You will only write code blocks."
        if self.task == "math":
            from prompts.Reasoning.math_prompt import math_deepseekpal_prompt 
            self.prompts["prompt"] = math_deepseekpal_prompt  #code_prompt#pal_prompt
            self.prompts["system_msg"] = "You will write python program to solve math problems. You will only write imports and code blocks ."

    def evaluate(self):
        
        dataset_name = os.path.basename(self.dataset_path).split(".")[0]
        result = []
        
        # create a new empty file for logging
        f = open(os.path.join(self.log_path,f"{self.task}_{dataset_name}.txt"), "w")
            
        for id, item in tqdm(enumerate(self.dataset), total=len(self.dataset)):

            question = item["question"]
            # print(question)
            success, output = self.algorithm.run(question, prompts=self.prompts, end_suffix="return")
            
            if success:
                evaluation, executed_output = evaluate_results(self.task, item, output)
                result.append(evaluation)
            else:
                evaluation = None  
            if type(output) == list: output = "\n".join(output) 
            output = output + "\n Executed result: " + str(executed_output)
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
    

def parse_args():
    parser = argparse.ArgumentParser(description="Testing")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--tasks", required=True, type=str, help="specify the tasks")
    parser.add_argument("--algorithm", required=True, type=str, help="specify the algorithm")
    parser.add_argument("--model", required=True ,help="specify the models, available models are stated in the configuration file")
    parser.add_argument("--log_path", required=False, default='', help="specify the place to store the resuls")
    parser.add_argument("--data_path", required=False, default='/root/huggingface/gsm8k', help="specify the test data file")
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
    
    metrics = eval_reasoning.evaluate()

def test_execution():
    code = """
    def solution():
        # Let the current age of the son be x
        x = int(input("Enter the current age of the son: "))

        # The current age of the father is 5 times the son's age
        father_age = 5 * x

        # The sum of their ages 3 years ago is 30
        total_age_3_years_ago = 30

        # The sum of their ages 3 years ago is the sum of their current ages minus 6
        current_age_sum = x + father_age + 6

        # Solve for x
        x = (total_age_3_years_ago - 6) / 2

        # The son's current age is x + 3
        son_age = x + 3

        # Format the result in LaTeX
        result = r'The son is {} years old.'.format(son_age)
        return result
    """
    result = execute_solution(code)
    
    
if __name__ == "__main__":
    main()
    # test_execution()