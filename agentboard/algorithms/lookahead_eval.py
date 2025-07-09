import pdb

from common.registry import registry
# from rouge import Rouge
import json
import random
import re
import argparse

@registry.register_algorithm("Lookahead_Eval")
class Lookahead_Eval:  # the agent should receive goal, state and action, then return the next state
    def __init__(self,
                 llm_model,
                 prompt_path,
                 n_gram=3,
                 reward_threshold=1.0,
                 window_size=2
                 ):
        
        self.llm_model = llm_model
        
        self.prompts = json.load(open(prompt_path, 'r'))
        
        self.n_gram = n_gram
        
        self.task = "dp" if "dp" in prompt_path else "pf"
        
        self.reward_threshold = reward_threshold 
        self.window_size = window_size
        
    def make_prompt(self, prompt):
        query = ""
        query += prompt["instruction"] + '\n'
        query += "\nHere is an example:\n" + prompt["examples"][0] + '\n'
        
        input_prompt = query 
        
        for item in self.memory:
            input_prompt += f"\n{item[0]}: {item[1]}"
        
        return input_prompt
    
    def get_samples(self, n_generate_sample, prompt, stop=None):
        if stop is not None:
            config = {"n": n_generate_sample, "stop": stop, "max_tokens": 200, "temperature": 1}
        else:
            config = {"n": n_generate_sample, "max_tokens": 200, "temperature": 1}
            
        success, samples = self.llm_model.generate_with_config(self.prompts["system_msg"], prompt, config)
        return success, samples
    
    def update_trajectory_pool(self, action):
        
        # update the trajectory pool with the generated action rollouts by llm
        
        action_rollouts, new_action = self.parse_action_sequnece(action)
        
        begin_observation = self.memory[-1][1]
        
        action_rollouts.insert(0, {"Action":None, "Verified": True, "Observation": begin_observation, "Reward": None})
        
        self.trajectory_pool.append(action_rollouts)
        
        self.update_trajectory_reward()
        
    def parse_action_sequnece(self,action): 
        
        # parse the llm generated action sequence into a trajectory list
        
        action_sequences = action.split('\n')
        all_actions = []
        for action in action_sequences:
            try:
                if "action:" in action.lower():
                    new_action = action.split(":")[1]
                    new_action = new_action.strip()
                    all_actions.append({"Action": new_action, "Verified": None, "Observation": None, "Reward": None, "Normalized_Reward": None})
                elif ":" in action.lower():
                    type = action.split(":")[0]
                    content = action.split(":")[1]
                    type = type.strip()
                    content = content.strip()
                    all_actions[-1][type.capitalize()] = content
                else:
                    continue
            except:
                continue
            
        if len(all_actions)>0: 
            first_action = all_actions[0]["Action"] 
        else:
            first_action = action_sequences[0]
            all_actions.append({"Action": first_action, "Verified": None, "Observation": None, "Reward": None, "Normalized_Reward": None})
        return all_actions, first_action
    
    def update_trajectory_reward(self):
        
        # calculate the reward for new action rollouts, this function is called after each action execution

        new_trajectory = self.trajectory_pool[-1]
        
        reward_sequence = [new_trajectory[id]["Reward"] for id in range(len(new_trajectory)) if "Reward" in new_trajectory[id] and new_trajectory[id]["Reward"] is not None]
        
        reward = None
        if len(reward_sequence) > 0:
            all_rewards = [self.find_numbers(item) for item in reward_sequence]
            all_rewards = [item[0] for item in all_rewards if len(item) > 0]
            
            if len(all_rewards) > 0:
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
    
    def eval_dp(self, trajectory):
        prompt = self.make_prompt(self.prompts)
        for item in trajectory:
            if "Action" in item and item["Action"] is not None:
                prompt += f"\nAction: {item['Action']}"
            # if "Observation" in item and item["Observation"] is not None:
            #     prompt += f"\nObservation: {item['Observation']}"
        
        prompt += f"\nPlease calculate the reward for the action sequence above. {self.prompts['evaluation_prompt']}"
        prompt += f"\nReward:"
        
        config = {"stop": '\n', "max_tokens": 20}
        success, output = self.llm_model.generate_with_config(self.prompts["system_msg"], prompt, config)
        
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
        
        action_history = [item[1] for item in self.memory if item[0] == "Action"]
        
        for traj_id, trajectory in enumerate(self.trajectory_pool):
            
            trajectory = trajectory[1:] # remove the first action, which is None
            
            for id in range(len(trajectory) - self.n_gram + 1):
                
                n_gram_list = [trajectory[id+s]["Action"] for s in range(self.n_gram)]
                n_gram_observation = [trajectory[id+s]["Observation"] for s in range(self.n_gram)]
                n_gram_reward_normalized = [trajectory[id+s]["Normalized_Reward"] for s in range(self.n_gram)]
                n_gram_reward = [trajectory[id+s]["Reward"] for s in range(self.n_gram)][-1]
                
                match =  (action_history[-self.n_gram+1:] == n_gram_list[:-1])
                
                reward_bad = n_gram_reward_normalized[-1] < reward_threshold
                
                if match and not reward_bad:
                    return n_gram_list[-1], n_gram_observation[-1]

        return None, None
    
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
        
    def run(self, question):
        
        self.trajectory_pool = []
        self.trajectory_reward = []
        
        self.memory = []
        
        iter = 0 
        
        args = {
            "n_generate_sample": 10,
            "max_iters": 10
        }
        
        args = argparse.Namespace(**args)
        
        self.memory.append(("Question", question))
        
        for iter in range(args.max_iters):
            
            action, observation = self.lookahead_decision_model(reward_threshold=self.reward_threshold)
            
            if action is not None:
                
                if observation is not None:
                    self.memory.append(("Action", action))
                    self.memory.append(("Observation", observation))
                else:
                    success, observation = self.world_model_step(action)
            
            else:
                
                need_tip, reflection_tip = self.reflection_tip(reward_threshold=self.reward_threshold, window_size=self.window_size)
                
                if need_tip:
                    self.memory.append(("Reflection", reflection_tip))
                
                
                system_message = self.prompts["system_msg"]
                input_prompt = self.make_prompt(self.prompts)
                wrapped_prompt = f"{input_prompt}\nActions and Observations:"
            
                success, samples = self.get_samples(args.n_generate_sample, wrapped_prompt)
                samples = set(list(samples))
                for sample in samples:
                    self.update_trajectory_pool(sample)
                    
        answer_prompt = self.make_prompt(self.prompts) + "\n Answer:"
        success, answer = self.llm_model.generate(self.system_message, answer_prompt)
            
        if success:
            result_lists = self.parse_integer_lists(answer)
            if len(result_lists) > 0:
                return True, result_lists[-1]
        
            
        return False, None
    
    @classmethod
    def from_config(cls, llm_model, config):
        return cls(llm_model, config["prompt_path"])