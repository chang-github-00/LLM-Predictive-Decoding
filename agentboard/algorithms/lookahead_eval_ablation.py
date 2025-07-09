import pdb

from common.registry import registry
# from rouge import Rouge
import json
import random
import re
import argparse

@registry.register_algorithm("Lookahead_Eval_Ablation")
class Lookahead_Eval_Ablation:  # the agent should receive goal, state and action, then return the next state
    def __init__(self,
                 llm_model,
                 prompt_path,
                 n_gram=3,
                 reward_threshold=1.0,
                 window_size=2
                 ):
        
        self.llm_model = llm_model
        
        self.prompts = json.load(open(prompt_path, 'r'))
        
        self.task = "dp" if "dp" in prompt_path else "pf"
        
        self.reward_threshold = reward_threshold 
        self.window_size = window_size
        
        self.problem_size = 10
        
        self.n_gram = self.problem_size
        
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
    
    def update_trajectory_pool(self, action):
        
        # update the trajectory pool with the generated action rollouts by llm
        
        action_rollouts, new_action = self.parse_action_sequence(action)
        
        if action_rollouts is None:
            return  
        
        action_rollouts.insert(0, {"Action":None, "Reward": None})
        
        self.trajectory_pool.append(action_rollouts)
        
        self.update_trajectory_reward()
    
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
    
    def update_trajectory_reward(self):
        
        # calculate the reward for new action rollouts, this function is called after each action execution

        new_trajectory = self.trajectory_pool[-1]
        
        reward_sequence = [new_trajectory[id]["Reward"] for id in range(len(new_trajectory)) if "Reward" in new_trajectory[id] and new_trajectory[id]["Reward"] is not None]
        
        reward = None
        if len(reward_sequence) > 0:
            
            all_rewards = [float(reward) for reward in reward_sequence]
            reward = all_rewards[-1]

        
        if reward is None:
            success, reward = self.eval(new_trajectory)
        
        self.trajectory_reward.append(reward)
        
        
        
        for id in range(len(self.trajectory_pool[-1])):
            if "Reward" in self.trajectory_pool[-1][id]:
                self.trajectory_pool[-1][id]["Reward"] = reward
                self.trajectory_pool[-1][id]["Normalized_Reward"] = reward
        
        if self.task == "dp":
            max_reward = max(self.trajectory_reward)
            for trajectory in self.trajectory_pool:
                for id in range(len(trajectory)):
                    if "Normalized_Reward" in trajectory[id]:
                        if max_reward > 0:
                            trajectory[id]["Normalized_Reward"] = trajectory[id]["Reward"] / max_reward
                            
                        if max_reward == 0:
                            trajectory[id]["Normalized_Reward"] = 1
                        
                        if max_reward < 0:
                            trajectory[id]["Normalized_Reward"] = max_reward / trajectory[id]["Reward"]
            
    def eval(self, trajectory, start_id):
        if self.task == "dp":
            return self.eval_dp(trajectory, start_id)
        elif self.task == "pf":
            return self.eval_pf(trajectory, start_id)
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
    
    def eval_dp(self, action_list, start_id):
        input = eval(self.prompts["question"].split("=")[1])
        action_list = self.memory[:start_id]+action_list[start_id:]
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
    
    
    def lookahead_decision_model(self, reward_threshold=1.0):
        # given the look ahead predictions, make the next action
        
        # ! todo: choose the best action when there are multiple options
        
        action_history = [None] + [action for action in self.memory if action is not None]
        
        for traj_id, trajectory in enumerate(self.trajectory_pool):
            
            start = max(1, len(trajectory) - self.n_gram + 1)
            
            for id in range(start):
                
                n = min([len(trajectory) - id, self.n_gram, len(action_history)+1])
                n_gram_list = [trajectory[id+s]["Action"] for s in range(n)]
                n_gram_reward_normalized = [trajectory[id+s]["Normalized_Reward"] for s in range(n)]
                n_gram_reward = [trajectory[id+s]["Reward"] for s in range(n)][-1]
                
                match =  (action_history[-n+1:] == n_gram_list[:-1])
                
                reward_bad = n_gram_reward_normalized[-1] < reward_threshold
                
                if match and not reward_bad:
                    return n_gram_list[-1]
                
        if len(self.trajectory_pool) > 0:
            print(self.prompts["question"])
            print("Warning: no action found")
            # print(self.trajectory_pool)
        return None
    
    def reflection_tip(self, reward_threshold=1.0, window_size=2):
        # generate reflection tips for the user
        action_history = [item[1] for item in self.memory if item[0] == "Action"]
        if len(action_history) > window_size:
            last_actions = action_history[-window_size:] 
            for traj_id, trajectory in enumerate(self.trajectory_pool):

                trajectory = trajectory[1:] # remove the first action, which is None
                
                for id in range(len(trajectory) - window_size + 1):
                    
                    n_gram_list = [trajectory[id+s]["Action"] for s in range(window_size)]
                    n_gram_reward_normalized = [trajectory[id+s]["Normalized_Reward"] for s in range(window_size)]
                    n_gram_reward = [trajectory[id+s]["Reward"] for s in range(window_size)][-1]
                    
                    match = (last_actions == n_gram_list)
                    reward_bad = n_gram_reward_normalized[-1] < reward_threshold
                    
                    if match and reward_bad:
                        
                        # find a trajectory in the pool that has reward=1
                        better_trajectory = None
                        for trajectory in self.trajectory_pool:
                            if trajectory[-1]["Normalized_Reward"] == 1:
                                better_trajectory = trajectory
                                break
                        
                        trajectory_text = "".join([f"Action: {item['Action']}" for item in better_trajectory])
                        return True, f"Your recent actions are not optimal. A better solution is {trajectory_text}. Please consider revising your actions."
                    
        return False, None 
    
    def world_model_step(self, action):
        prompt = self.make_prompt(self.prompts) + '\n Action:' + action + f"\n Predict the outcome observation of action {action}. Observation:"
        config = {"stop": '\n', "max_tokens": 20}
        success, output = self.llm_model.generate_with_config(self.system_message, prompt, config)
        
        if ":" in output:
            observation = ":".join(output.split(":")[1:])
        else:
            observation = output
        return success, observation.strip()
    
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
        
        all_iter = 0
        iter = 0
        while iter < args.max_iters :
            
            system_message = self.prompts["system_msg"]
            
            input_prompt = self.make_prompt(self.prompts)
            wrapped_prompt = f"{input_prompt}\nYou could only change None in the constraints, constraints:{self.memory}\nAnswer="
        
            success, samples = self.get_samples(args.n_generate_sample, wrapped_prompt, stop="\n")
            if isinstance(samples, str):
                samples = [samples]
            samples = set(list(samples))
            evaluations = []
            for sample in samples:
                try:
                    action = self.parse_integer_lists(sample)[0]
                    if len(action) != self.problem_size:
                        continue
                    success, evaluation = self.eval(action, iter)
                    evaluations.append((action, evaluation))
                except:
                    continue
            
            try:
                evaluations = sorted(evaluations, key=lambda x: x[1], reverse=True)
                action = evaluations[0][0][iter]
            
                self.memory[iter] = action
                iter += 1
            except:
                pass
            
            all_iter += 1
            if all_iter > 12:
                return False, None
                    
        # answer_prompt = self.make_prompt(self.prompts) + "\n output="
        # success, answer = self.llm_model.generate(self.prompts["system_msg"], answer_prompt)
            
        if success:
            # result_lists = self.parse_integer_lists(answer)
            # if len(result_lists) > 0:
            #     return True, result_lists[-1]
            return True, self.memory
        
            
        return False, None
    
    @classmethod
    def from_config(cls, llm_model, config):
        return cls(llm_model, config["prompt_path"])