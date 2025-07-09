import pdb

from agents.base_agent import BaseAgent
from common.registry import registry
# from rouge import Rouge
import json
import random
import re
import torch
from sentence_transformers import SentenceTransformer, util


@registry.register_agent("LookAheadEvalAgent")
class LookAheadEvalAgent(   # add world modeling objective in agent 
    BaseAgent):  # the agent should receive goal, state and action, then return the next state
    def __init__(self,
                 llm_model,
                 memory_size=100,
                 # set this to a very large number if you want to keep all history till context length limit
                 examples=[],
                 instruction="",
                 init_prompt_path=None,
                 system_message="You are a helpful assistant.",
                 need_goal=False,
                 check_actions=None,
                 check_inventory=None,
                 use_parser=True,
                 ):
        super().__init__()
        self.use_parser = use_parser
        self.llm_model = llm_model
        self.memory_size = memory_size
        self.goal = None
        self.init_obs = None
        if init_prompt_path is not None:  # load from file
            self.init_prompt_dict = json.load(open(init_prompt_path, 'r'))
            self.instruction = self.init_prompt_dict["instruction"]
            self.examples = self.init_prompt_dict["examples"]
        else:

            self.instruction = instruction
            self.examples = examples

            # self.reset(goal, init_obs)
            self.init_prompt_dict = {
                "examples": examples,
                "instruction": instruction,
                "system_msg": system_message
            }

        self.max_context_length = self.llm_model.context_length
        self.need_goal = need_goal
        self.check_actions = check_actions
        self.check_inventory = check_inventory

        self.example_prompt = None

        self.n_gram = 3
        self.n_gram_pool = []
        
        self.reward = []
        
        self.use_guess_cnt = 0
        
        
        # self.guess_action = []

        
        self.similarity_metric = SimilarityMetric()
        
        if "claude" in self.llm_model.engine:
            self.split = self.llm_model.xml_split
        else:
            self.split = {"example": [""],
                          "text": [""],
                          "rule": [""],
                          "system_msg": [""],
                          "instruction": [""],
                          "goal": [""]}

    def get_example_prompt(self): #return the prompt for an interaction turn
        return self.example_prompt
    
    def log_example_prompt(self, prompt):
        self.example_prompt = prompt

    def reset(self, goal, init_obs, init_act=None):
        self.goal = goal
        self.init_obs = init_obs
        self.memory = [("Action", init_act), ('Observation', self.init_obs)] if init_act \
            else [
            ('Observation', self.init_obs)]  # list of [('State', "xxx"), ('Action', "xxx"), ...]
        self.steps = 0
        self.done = False
        self.n_gram_pool = []
        self.reward = []
        
        self.use_guess_cnt = 0
        # self.guess_action = []


    def update(self, action, state):
        self.steps += 1

        self.memory.append(("Action", action))
        self.memory.append(("Observation", state))

    def make_prompt(self, need_goal=False, check_actions="check valid actions", check_inventory="inventory", system_message='', tip=None):
        query = ""
        query += self.split["instruction"][0] + self.instruction + self.split["instruction"][-1]

        if isinstance(self.examples, str):
            self.examples = [self.examples]

        if len(self.examples) > 0:
            query += "\nHere are examples:\n" + self.split["example"][0]
            for example in self.examples:
                query += example + "\n"
            query += self.split["example"][-1]
        if need_goal:
            query += self.split["goal"][0] + "You should perform actions to accomplish the goal: " + self.goal + "\n" + \
                     self.split["goal"][-1]
        if check_actions is not None:
            query += "You should use the following commands for help when your action cannot be understood: " + check_actions + "\n"
        if check_inventory is not None:
            query += "You should use the following commands for help when your action cannot be understood: inventory\n"

        history = self.memory[-self.memory_size:]
        input_prompt = query + "\n".join([item[0] + ": " + item[1] for item in history])
        
        if tip is not None:
            input_prompt += "\n Thought: " + tip

        input_prompt += "\nActions and Observations: "

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_prompt}
        ]
        num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
        while num_of_tokens > self.max_context_length - self.llm_model.max_tokens:
            history = history[1:]
            input_prompt = query + "\n".join([item[0] + ": " + item[1] for item in history])
            if tip is not None:
                input_prompt += "\n Thought: " + tip

            input_prompt += "\nActions and Observations: "
            
            # input_prompt += "\nPlease enter your action:"
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_prompt}
            ]
            num_of_tokens = self.llm_model.num_tokens_from_messages(messages)

        return input_prompt

    def action_parser_for_special_llms(self, action):
        
        '''
        This function is used to parse the action for special llms, e.g. codellama-13b, codellama-34b, llama, lemur, vicuna, etc.
        These llms often struggle to generate the format of the action correctly, so we need to parse the action to make it executable.
        '''
        
        origin_action = action
        if 'action' in action.lower():
            action_temp = action.split('\n')
            for act in action_temp:
                if "next action" in act and ':' in act: # zzh: in Claude will return "Here is the next action to take:"
                    idx = action_temp.index(act)
                    while idx + 1 < len(action_temp):
                        if action_temp[idx + 1]:
                            action = action_temp[idx + 1]
                            break
                        idx += 1
                if act.split(':')[0].lower().endswith('with action input'): # chang: in case parse tool output
                    action = act
                    break
                if 'action' in act.lower() and ':' in act:
                    action_temp = ':'.join(act.split(':')[1:])
                    if action_temp != "":
                        action = action_temp
                        break
                if 'action' in act.lower() and 'is to' in act:
                    action_temp = act.split('is to')[1]
                    if action_temp != "":
                        action = action_temp
                        break
                        
        if action.strip() == "":
            action = origin_action.split('\n')[0]   # temperary comment this line for codellama
        action = action.strip()
        action = action.strip("'/")
        action = action.split('\n')[0]
        return action

    
    def parse_action_sequnece(self,action):
        action_sequences = action.split('\n')
        all_actions = []
        for action in action_sequences:
            try: 
                if "action:" in action.lower():
                    new_action = action.split(":")[1]
                    new_action = new_action.strip()
                    all_actions.append({"Action": new_action})
                elif ":" in action.lower():
                    type = action.split(":")[0]
                    content = action.split(":")[1]
                    type = type.strip()
                    content = content.strip()
                    all_actions[-1][type.capitalize()] = content
            except:
                continue
                
        if len(all_actions)>0: 
            first_action = all_actions[0]["Action"] 
        else:
            first_action = ""
        return all_actions, first_action
    
    def update_n_gram(self, action):
        # self.guess_action = []
        action_ngram, new_action = self.parse_action_sequnece(action)
        
        self.n_gram_pool.append(action_ngram)
        
        # self.update_reward()
        # for sequence in self.n_gram_pool:
        #     for i in range(len(sequence)-self.n_gram):
        #         n_gram_list = [sequence[i+s]["Action"] for s in  range(self.n_gram)]
                
        #         if new_action == sequence[i]["Action"]:
        #             self.guess_action = n_gram_list[1:]
    
    def update_reward(self):
        self.reward = [] # re-calculate the reward for each n-gram sequence
        history = []
        for item in self.memory:
            if item[0] == "Action":
                history.append((item[1], self.memory[self.memory.index(item)+1][1]))
            else:
                continue
        
        for i in range(len(self.n_gram_pool)):
            trajectory = self.n_gram_pool[i]
            observations = [item["Observation"] for item in trajectory if "Observation" in item]
            if len(observations) == 0:
                self.reward.append(None)
                continue
                
            sim_to_goal= float(torch.max(self.similarity_metric.get_similarity(observations, self.goal)))  # get the max similarity score of states to goal, measuring how close the state is to the goal
            all_correctness = []
            
            for item in trajectory:
                action = item["Action"]
                observation = item["Observation"] if "Observation" in item else None
                for (at, st) in history:
                    if action == at:
                        if observation is not None:
                            correctness_of_lookahead = self.similarity_metric.get_similarity([st], observation)[0]
                        else:
                            correctness_of_lookahead = 1
                        all_correctness.append(float(correctness_of_lookahead))

            correctness_of_traj = 1 if len(all_correctness) == 0 else sum(all_correctness)/len(all_correctness)
            
            self.reward.append(sim_to_goal*correctness_of_traj) # evaluate two aspects of the trajectory, the similarity to the goal and the correctness of the lookahead action
            
        return
    
    
    def lookahead_decision_model(self):
        # given the look ahead predictions, make the next action
        
        # if self.guess_action) > 0:
        #     return self.guess_action.pop()
                
        action_history = [item[1] for item in self.memory if item[0] == "Action"]
        valid_guess_actions = []
        valid_reward = []
        for id, sequence in enumerate(self.n_gram_pool):
            for i in range(len(sequence)-self.n_gram):
                n_gram_list = [sequence[i+s]["Action"] for s in  range(self.n_gram)]
                
                if action_history[-self.n_gram+1:] == n_gram_list[:-1]:
                    valid_guess_actions.append(n_gram_list[-1])
                
                    valid_reward.append(self.reward[id])
        
        # find the action with the highest reward
        if len(valid_guess_actions) > 0:
            guess_action =  valid_guess_actions[valid_reward.index(max(valid_reward))]
        
            return guess_action
        
        else:
            return None
        

    def reflection_tips(self): #determine if the model is stuck and requires self-reflection, used sparingly
        
        try:
            history = []
            for item in self.memory:
                if item[0] == "Action":
                    history.append((item[1], self.memory[self.memory.index(item)+1][1]))
                else:
                    continue
            
            action_history = [item[1] for item in self.memory if item[0] == "Action"]
            # first scenario: based on history it is not good, and deviating from the goal
            # there are repeat cycles of actions that are not helping the model to reach the goal
            if action_history.count(action_history[-1])>1 and action_history.count(action_history[-2])>1:
                
                action = "I have been repeating the same action, and it is not helping me to reach the goal. I need to try something different."
                return True, action
        except:
            pass
        
        # second scenario: based on lookahead prediction it is not good, and deviating from the goal
        
        # last n-gram is looked ahead, but the actual observations are not according to plan
        
        try:
            last_n_gram = action_history[-self.n_gram:]
            average_similarity = []
            average_reward = []
            
            for id, sequence in enumerate(self.n_gram_pool):
                for i in range(len(sequence)-self.n_gram):
                    n_gram_list = [sequence[i+s]["Action"] for s in  range(self.n_gram)]
                    if last_n_gram == n_gram_list:
                        average_reward.append(self.reward[id])
                        
                        observations = [item["Observation"] for item in sequence[i:i+self.n_gram]]
                        real_observation = [item[1] for item in history ][-self.n_gram:]
                        average_similarity.append(torch.max(self.similarity_metric.get_similarity(observations, real_observation)))
                        
            average_reward = sum(average_reward)/len(average_reward)
            average_similarity = sum(average_similarity)/len(average_similarity)
            
            if average_reward < 0.5:
                action = "I have been making actions that are not helping me to reach the goal. I need to try something different."
                return True, action
            
            if average_similarity < 0.5:
                action = "The execution of my plan is not going as expected. I need to try something different."
                return True ,action
        except:
            pass
            
        
        return False, None
        
        
        
    
    def run(self, init_prompt_dict=None):
        
        self.update_reward()

        # decide upon the best action based on simulated planning
        action = self.lookahead_decision_model()
        
        if action is not None:
            self.use_guess_cnt += 1
            return True, action, True
        
        else:
            need_tip, reflection_tip = self.reflection_tips()
            
            # if need_tip:
            #     return True, reflection_tip, True
            
            # else:        
                # note that these configs are originally provided when initialized, but you can choose to override them here with parameters
            if init_prompt_dict is not None:
                self.init_prompt_dict = init_prompt_dict
                self.instruction = init_prompt_dict['instruction']
                self.examples = init_prompt_dict['examples']
                
            system_message = self.init_prompt_dict['system_msg']
            input_prompt = self.make_prompt(need_goal=self.need_goal,
                                            check_actions=self.check_actions,
                                            check_inventory=self.check_inventory,
                                            system_message=system_message,
                                            tip = reflection_tip)
            
            self.log_example_prompt(input_prompt)

            success, action_sequence = self.llm_model.generate(system_message, input_prompt)
            
            if success:

                action_ngram, action = self.parse_action_sequnece(action_sequence)
                
                self.update_n_gram(action_sequence)
                
                if success and self.use_parser:
                    action = self.action_parser_for_special_llms(action)

            return success, action, False   

    @classmethod
    def from_config(cls, llm_model, config):
        memory_size = config.get("memory_size", 100)
        instruction = config.get("instruction", "")
        examples = config.get("examples", [])
        init_prompt_path = config.get("init_prompt_path", None)
        system_message = config.get("system_message", "You are a helpful assistant.")
        check_actions = config.get("check_actions", None)
        check_inventory = config.get("check_inventory", None)
        use_parser = config.get("use_parser", True)
        need_goal = config.get("need_goal", False)
        return cls(llm_model, memory_size, examples, instruction, init_prompt_path, system_message, 
                   need_goal, check_actions, check_inventory, use_parser)
        

class SimilarityMetric(object):
    def __init__(self,
                 model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def get_similarity(self, sequences, source_sequence):
        source_embedding = self.model.encode(source_sequence)
        sequence_embeddings = self.model.encode(sequences)
        
        similarity = util.pytorch_cos_sim(source_embedding, sequence_embeddings)
        return similarity