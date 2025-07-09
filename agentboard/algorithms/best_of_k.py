import pdb

from common.registry import registry
# from rouge import Rouge
import json
import random
import re
import argparse

@registry.register_algorithm("BestK")
class BestK:  # the agent should receive goal, state and action, then return the next state
    def __init__(self,
                 llm_model,
                 prompt_path,
                 ):
        
        self.llm_model = llm_model
        
        self.prompts = json.load(open(prompt_path, 'r'))
        
        self.task = "dp" if "dp" in prompt_path else "pf"
    
        self.problem_size = 10
        
        
    def make_prompt(self, prompt):
        query = ""
        query += prompt["instruction"] + '\n'
        query += "\nHere are examples:\n" + prompt["examples"][0] + '\n'
        
        input_prompt = query + "Question:" + prompt["question"]
        
        return input_prompt
    
    def get_samples(self, n_generate_sample, prompt, stop=None):
        if stop is not None:
            config = {"n": n_generate_sample, "stop": stop, "max_tokens": 30, "temperature": 0.7}
        else:
            config = {"n": n_generate_sample, "max_tokens": 30, "temperature": 0.7}
            
        success, samples = self.llm_model.generate_with_config(self.prompts["system_msg"], prompt, config)
        return success, samples

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
            
    def parse_action_sequence(self,action): 
        
        lists_action = self.parse_integer_lists(action)
        if len(lists_action) == 0:
            return None, None
        action_extracted = lists_action[-1]
        
        reward = None
        try:
            if "reward" in action.lower():
                text_after_reward = action[action.lower().index("reward"):]
                reward = self.find_numbers(text_after_reward)[0]
        except:
            pass
        
        action_rollouts = [{"Action": action_extracted[id], "Reward": reward} for id in range(len(action_extracted))]
        return action_rollouts, action_extracted[0]
    
    
    def eval(self, trajectory):
        if self.task == "dp":
            return self.eval_dp(trajectory)
        elif self.task == "pf":
            return self.eval_pf(trajectory)
        else:
            raise NotImplementedError
    
    def find_numbers(self, text):
        # Regular expression pattern to match integers and floats
        # Explanation:
        # -?        Optional negative sign
        # \d+       One or more digits
        # (?:       Start of non-capturing group for decimal part
        #   \.\d+   Decimal point followed by one or more digits
        # )?        The decimal part is optional
        pattern = r'-?\d+(?:\.\d+)?'
        
        # Find all matches using re.findall
        numbers = re.findall(pattern, text)
        
        # Convert all found strings to float or int as appropriate
        converted_numbers = []
        for number in numbers:
            # Convert to float if it contains a decimal point, else to int
            if '.' in number:
                converted_numbers.append(float(number))
            else:
                converted_numbers.append(int(number))
        
        return converted_numbers
    
    def eval_dp(self, action_list):
        input = eval(self.prompts["question"].split("=")[1])
        num_sum = sum([input[i] for i in range(len(input)) if action_list[i] == 1])
        
        for i in range(len(action_list)-1):
            if action_list[i] == 1 and action_list[i+1] == 1:
                return True, -50
        return True, num_sum
        
        prompt = self.make_prompt(self.prompts)
        
            # if "Observation" in item and item["Observation"] is not None:
            #     prompt += f"\nObservation: {item['Observation']}"
        
        prompt += f"\n{self.prompts['evaluation_prompt']}\nAnswer: {action_list}. "
        prompt += f"\nReward:"
        
        config = {"stop": '', "max_tokens": 50}
        success, output = self.llm_model.generate_with_config(self.prompts["system_msg"], prompt, config)
        
        if "is" in output:
            output = "is".join(output.split("is")[1:])
        if ":" in output:
            output = ":".join(output.split(":")[1:])
        numbers = self.find_numbers(output)
        
        if len(numbers) > 0:
            return True, numbers[0]
        else:
            return False, None
        
    def eval_pf(self, trajectory):
        return True, 0
    
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
        
        self.trajectory_pool = []
        self.trajectory_reward = []
        
        self.memory = [None]*self.problem_size
        self.prompts["question"] = question
        iter = 0 
        
        args = {
            "n_generate_sample":10,
            "max_iters": self.problem_size
        }
        
        args = argparse.Namespace(**args)
        
        
            
        system_message = self.prompts["system_msg"]
            
        input_prompt = self.make_prompt(self.prompts)
        wrapped_prompt = f"{input_prompt}\nYou could only change None in the constraints, constraints:{self.memory}\nAnswer="
    
        success, samples = self.get_samples(args.n_generate_sample, wrapped_prompt, stop="\n")
        if not success:
            return False, None
        if isinstance(samples, str):
            samples = [samples]
        
        samples = set(list(samples))
        evaluations = []
        for sample in samples:
            try:
                action = self.parse_integer_lists(sample)[0]
                if len(action) != self.problem_size:
                    continue
                success, evaluation = self.eval(action)
            except:
                continue
            
            evaluations.append((action, evaluation))
            
        evaluations = sorted(evaluations, key=lambda x: x[1], reverse=True)
        
        action = evaluations[0][0]
        
                    
        # answer_prompt = self.make_prompt(self.prompts) + "\n output="
        # success, answer = self.llm_model.generate(self.prompts["system_msg"], answer_prompt)
            
        if success:
            # result_lists = self.parse_integer_lists(answer)
            # if len(result_lists) > 0:
            #     return True, result_lists[-1]
            return True, action
        
            
        return False, None
    
    @classmethod
    def from_config(cls, llm_model, config):
        return cls(llm_model, config["prompt_path"])