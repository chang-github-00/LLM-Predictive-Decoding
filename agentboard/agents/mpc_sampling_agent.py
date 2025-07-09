import pdb

from agents.base_agent import BaseAgent
from common.registry import registry
# from rouge import Rouge
import json
import random
import re
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from sentence_transformers import SentenceTransformer, util
# from InstructorEmbedding import INSTRUCTOR

@registry.register_agent("MPCSample")
class MPCSample(   # add world modeling objective in agent 
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
                 logger=None,
                 n_gram=30,
                 similarity_threshold_high=0.7,
                 similarity_threshold_low=0.5,
                 reward_threshold=0.5,
                 do_sample=True,
                 max_world_model_len=200,
                 beam_temperature=0.7,
                 select_temperature=0.1,
                 n_generate_sample=6,
                 value_type = "heuristic",
                 lookahead_length = 8,
                 ):
        super().__init__()
        self.use_parser = use_parser
        self.llm_model = llm_model
        self.memory_size = memory_size
        self.goal = None
        self.init_obs = None
        self.logger = logger
        self.lookahead_length = lookahead_length
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

        self.n_gram = n_gram
        self.trajectory_pool = []
        self.trajectory_reward = []
        
        self.use_guess_cnt = 0

        # self.guess_action = []
        
        self.style = "full" # "no_cache_lookahead" "no_cache_no_lookahead"
        
        self.task = 'pddl' # 'pddl'
        self.n_generate_sample = n_generate_sample
        self.stop = []
        self.max_tokens = max_world_model_len
        self.temperature = beam_temperature
        self.select_temperature = select_temperature
        self.do_sample = do_sample
        self.max_iters = 2
        self.generation_config = {"n": self.n_generate_sample, "stop": self.stop, "max_tokens": self.max_tokens, "temperature": self.temperature}
        
        self.value_type = value_type # "logp", "llm"
        self.similarity_threshold_high = similarity_threshold_high # determine if two observations are similar
        self.similarity_threshold_low = similarity_threshold_low# determine if two observations are similar
        self.reward_threshold = reward_threshold #0.7 # determine if the reward is good enough
        self.window_size = 1 # determine the action window size for self-evaulation
        
        if self.value_type == "heuristic":
            self.similarity_metric = SimilarityMetric() #InstructionSimilarityMetric() #SimilarityMetric()
        else:
            self.similarity_metric = None
            
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

        
        self.trajectory_pool = []
        self.trajectory_reward = []
        
        self.use_guess_cnt = 0
        
        self.reflection_tip = None


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

        input_prompt += "Predict actions and observations of next 5 steps in this format:\nAction:\nObservation:\n" # alfworld - "\nActions and Observations:"  #(stop generating if you are not certain about the next observation)"

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

            input_prompt += "Predict actions and observations of next 5 steps in this format:\nAction:\nObservation:\n" # alfworld - "\nActions and Observations:"  #(stop generating if you are not certain about the next observation)"
            
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

    
    # ---------------------------------------- components of lookahead agent  ----------------------------------------
    
    def parse_action_sequence(self,action): 
        
        if isinstance(action, dict):
            action = action["text"]
        # parse the llm generated action sequence into a trajectory list
        
        action_sequences = action.split('\n')
        all_actions = []
        for action in action_sequences:
            try:
                if "action:" in action.lower():
                    new_action = action.split(":")[1]
                    new_action = new_action.strip()
                    all_actions.append({"Action": new_action, "Verified": None, "Observation": None, "Reward": None})
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
            
        all_actions = all_actions[:self.lookahead_length]
            
        if len(all_actions)>0: 
            first_action = all_actions[0]["Action"] 
        else:
            first_action = action_sequences[0]
            all_actions.append({"Action": first_action, "Verified": None, "Observation": None, "Reward": None})
        return all_actions, first_action
    
    def update_trajectory_pool(self, action, lookahead=True, reward = None  ):
        
        # update the trajectory pool with the generated action rollouts by llm
        if lookahead:
            action_rollouts, new_action = self.parse_action_sequence(action)
            
            history_rollouts = []
            
            begin_observation = self.memory[0][1]
            
            history_rollouts.append({"Action":None, "Verified": True, "Observation": begin_observation, "Reward": None})
            
            for item in self.memory:
                if item[0] == "Action": 
                    history_rollouts.append({"Action": item[1], "Verified": None, "Observation": None, "Reward": None})
                if item[0] == "Observation":
                    history_rollouts[-1]["Observation"] = item[1]
                    history_rollouts[-1]["Verified"] = True
                    
            full_rollouts = history_rollouts + action_rollouts
            self.trajectory_pool.append(full_rollouts)
            
            self.update_trajectory_reward(reward)
            
        else:
            action_rollouts, new_action = self.parse_action_sequence(action)
            
            history_rollouts = []
            
            begin_observation = self.memory[0][1]
            
            history_rollouts.append({"Action":None, "Verified": True, "Observation": begin_observation, "Reward": None})
            
            for item in self.memory:
                if item[0] == "Action": 
                    history_rollouts.append({"Action": item[1], "Verified": None, "Observation": None, "Reward": None})
                if item[0] == "Observation":
                    history_rollouts[-1]["Observation"] = item[1]
                    history_rollouts[-1]["Verified"] = True
            
            full_rollouts = history_rollouts + action_rollouts
            self.trajectory_pool.append(full_rollouts)
            
            self.update_trajectory_reward(reward)
            
        
    def update_trajectory_reward(self, reward=None):
        
        for id in range(len(self.trajectory_pool[-1])):
            if "Reward" in self.trajectory_pool[-1][id]:
                self.trajectory_pool[-1][id]["Reward"] = reward

    def get_heuristic_value(self, new_trajectory):
        
        # calculate the reward for new action rollouts, this function is called after each action execution

        # new_trajectory = self.trajectory_pool[-1]
        
        observations = [item["Observation"] for item in new_trajectory if "Observation" in item and item["Observation"] is not None]
        
        # action_history = [item[1] for item in self.memory if item[0] == "Action"]
        
        if len(observations) < 1:
            sim_to_goal = 0
        else:
            similarity = self.similarity_metric.get_goal_similarity(observations, self.goal)
            
            # index of max similarity
            step_most_similar = int(torch.argmax(similarity)) + 1
            
            sim_to_goal= float(torch.max(similarity))
            
            # sim_to_goal = sim_to_goal * self.gamma / step_most_similar
            
        return sim_to_goal  
    
    def get_llm_value(self, new_trajectory, init_prompt_dict=None):
        
        if self.task == 'alfworld':
            llm_generation_config = {"n": 1, "max_tokens": 100, "temperature": 0, "stop": ['\n']}
        else:
            llm_generation_config = {"n": 1, "max_tokens": 200, "temperature": 0, "stop": ['\n']}
        
        system_message = "Please evaluate the given trajectory."
        
        prompt = self.make_llm_scorer_prompt(new_trajectory, need_goal=self.need_goal, check_actions=self.check_actions, check_inventory=self.check_inventory, system_message=system_message, tip=None)
        
        success = False
         
        it = 0
        
        while not success:
            success, output = self.llm_model.generate_with_config(system_message, prompt, llm_generation_config)
            
            it += 1
            
            if it > 5:
                raise ValueError("LLM model failed to generate the score.")
            
            
        print(output)
        
        if "25" in output or "unlikely" in output:
            return 0.25
        if "50" in output or "not sure" in output:
            return 0.5
        if "75" in output or "very likely" in output:
            return 0.75
        if "0" in output or "impossible" in output:
            return 0
        if "definitely" in output or "100" in output:
            return 1
        return 0
    
    def get_pddl_value(self, new_trajectory):
        
        observations = [item["Observation"] for item in new_trajectory if "Observation" in item and item["Observation"] is not None]
        constraints = [subgoal.strip() for subgoal in self.goal.split(":")[1].split(",")]
        used_constraints = []
        
        for item in self.memory:
            if "Observation" in item and item[1] is not None:
                for subgoal in constraints:
                    if subgoal.lower() in item[1].lower():
                        used_constraints.append(subgoal)
        
        satisfied = 0
        for subgoal in constraints:
            if subgoal not in used_constraints:
                for ob in observations:
                    if subgoal.lower() in ob.lower():
                        satisfied += 1
                        continue
            # xxx is xxx, if the prefix matches yet the following part does not match, satisfied - 1
            if subgoal in used_constraints:
                for ob in observations:
                    if subgoal.split("is")[0].lower() in ob.lower():
                            satisfied -= 1
                            continue
        satisified_percentage = satisfied/len(constraints)
        
        return satisified_percentage
    
    def get_logp_value(self, new_trajectory, action_sequence):
        
        def _get_start_end_token_id(original_text, text, tokens):
            if text not in original_text:
                text = text.strip()
                if text not in original_text:
                    return 0, -1
            cnt_length = [len(token) for token in tokens]
            cumulated_cnt_length = np.cumsum(cnt_length)
            index_start =  original_text.index(text)#processed action index in action
            index_end = index_start + len(text)
                
            token_start = np.argmax(cumulated_cnt_length >= index_start)
            token_end = np.argmax(cumulated_cnt_length >= index_end)
            if token_end < token_start:
                token_end = -1
            return token_start+1, token_end+1 # +1 to account for the >= sign, instead of > sign
        
        logp = action_sequence["logprobs"]
        tokens = action_sequence["tokens"]
        
        actions = [item["Action"] for item in new_trajectory if "Action" in item and item["Action"] is not None]
        
        action = actions[0]
        # count the number of action in memory
        count = 0
        past_actions = []
        for item in self.memory:
            if item[0] == "Action":
                past_actions.append(item[1])
        if len(past_actions) >0 and past_actions[-1] == action:
            return 0
                
        observations = [item["Observation"] for item in new_trajectory if "Observation" in item and item["Observation"] is not None]
        
        if len(actions) < 1:
            return 0
        if len(observations) < 1:
            return 0
        
        first_action = actions[0] 
        first_action_start, first_action_end = _get_start_end_token_id(action_sequence["text"], first_action, tokens)
        last_action = actions[-1]
        last_action_start, last_action_end = _get_start_end_token_id(action_sequence["text"], last_action, tokens)
        
        last_observation = observations[-1]
        last_observation_start, last_observation_end = _get_start_end_token_id(action_sequence["text"], last_observation, tokens)
        
        first_token_id = first_action_start
        last_token_id = max(last_action_end, last_observation_end, first_action_end)
        
        action_logprobs = logp[first_token_id: last_token_id]
        action_prob = np.exp(sum(action_logprobs)) 
        action_length = last_token_id - first_token_id
        action_prob = action_prob ** (1 / action_length)
        
        return action_prob
            
    def make_llm_scorer_prompt(self, trajectory, need_goal=False, check_actions="check valid actions", check_inventory="inventory", system_message='', tip=None):
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
        
        query += "If an action is wrong or cannot be executed, the observation would be Nothing Happens."

        history = self.memory[-self.memory_size:] # need to add memory to make the trajectory whole
        
        new_interaction = []
        for item in trajectory:
            if "Action" in item and item["Action"] is not None:
                new_interaction.append(("Action", item["Action"]))
            if "Observation" in item and item["Observation"] is not None:
                new_interaction.append(("Observation", item["Observation"]))
        
        # evluation_instruction = f'''
        #     Evaluate the possibility the current trajectory could finish the task {self.goal} in the future in format: 
        #     [Because ...](a one-phrase rationale), the current trajectory has a score [0.x] (You must give a score between: 0: impossible, 25: unlikely, 50: not sure, 75: very likely, 100: definitely).
        #     Also penalize unnecessary repeated actions or actions that deviate far from the goal.
        # '''
        
        evluation_instruction_alfworld = f'''
            Evaluate the possibility the current trajectory could finish the task {self.goal} in the future in format: 
            [Because ...](a one-phrase rationale), the current trajectory has a score [0.x] (You must give a score between: 0: impossible, 25: unlikely, 50: not sure, 75: very likely, 100: definitely).
            Also, penalize unnecessary repeated actions or actions that deviate far from the goal {self.goal}. 
            Rules: 
            The agent needs to go to a place to pick up an object.
            The agent needs to go to a place to drop off an object.
            The agent can only carry one object at a time, therefore, the agent needs to drop off all inventory before picking up another object.
            The agent needs to go to a sinkbasin to clean an object.
            The agent needs to go to a microwave to heat an object.
            The agent can not pick up a desklamp, but can use - use desklamp - to turn on the light and look at object in inventory.
            If a correct object mentioned in {self.goal} is taken, you should give a score of 0.25 for each.
            If a wrong object not mentioned in {self.goal} is taken, you should give a score of 0. If an action is repeated a few times without success, you should give a score of 0.25.
            Otherwise, you should give a score of at least 0.25.
        '''
        evaluation_instruction_pddl = f'''
            Evaluate the possibility the current trajectory could finish the task {self.goal} in the future in format: 
            [Because ...](a one-phrase rationale), the current trajectory has a score [0.x] (You must give a score between: 0: impossible, 25: unlikely, 50: not sure, 75: very likely, 100: definitely).
            Also, penalize unnecessary repeated actions or actions that deviate far from the goal {self.goal}, or if there is backward movement. 
            If a subgoal was previously satisfied, you don't need to award it again.
            '''
            # If the task involves picking up a ball, the agent should pick up one ball each arm, move to another room, and drop the ball. Then go to the next room to pick up another ball.
            # If the task involves rearranging blocks, the agent should expose the blocks at the bottom, move the blocks to the right place, and then arrange the blocks in the correct order.
            # If the task involves making a cocktail, the agent should first acquire each ingredient with clean containers, then mix the ingredients in the correct order, and finally serve the cocktail. 
            # If the task involves arranging new tyres, the agent should first open boot for tools and tyres, jackup the tyre, then loosen the nuts, and unfaste the tyre, before putting the new tyre on.

        
        if self.task == 'alfworld':
            evluation_instruction = evluation_instruction_alfworld
        else:
            evluation_instruction = evaluation_instruction_pddl
        
        input_prompt = query + "Past interactions: \n" + "\n".join([item[0] + ": " + item[1] for item in history])

        input_prompt += "\n"  #(stop generating if you are not certain about the next observation)"

        input_prompt += input_prompt + "Evaluate the following trajectory: \n" + "\n".join([item[0] + ": " + item[1] for item in new_interaction])

        input_prompt += evluation_instruction
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_prompt}
        ]
        num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
        while num_of_tokens > self.max_context_length - self.llm_model.max_tokens:
            
            history = history[1:]
            input_prompt =  query + "Past interactions: \n" + "\n".join([item[0] + ": " + item[1] for item in history])
            

            input_prompt += "\n"  #(stop generating if you are not certain about the next observation)"
            
            input_prompt += input_prompt + "Evaluate the following trajectory: \n" + "\n".join([item[0] + ": " + item[1] for item in new_interaction])

            input_prompt += evluation_instruction
            # input_prompt += "\nPlease enter your action:"
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_prompt}
            ]
            num_of_tokens = self.llm_model.num_tokens_from_messages(messages)

        return input_prompt


    def make_llm_scorer_prompt_v2(self, trajectory, need_goal=False, check_actions="check valid actions", check_inventory="inventory", system_message='', tip=None):
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

        history = self.memory[-self.memory_size:] # need to add memory to make the trajectory whole
        
        rollout = []
        action_to_evaluate = None
        for item in trajectory:
            if "Action" in item and item["Action"] is not None:
                if action_to_evaluate is None: 
                    action_to_evaluate = item["Action"]  # first action in rollout
                rollout.append(("Action", item["Action"]))
            if "Observation" in item and item["Observation"] is not None:
                rollout.append(("Observation", item["Observation"]))

        
        evluation_instruction = f'''
            Evaluate if action {action_to_evaluate} could help finish the task {self.goal} in the future, based on past interactions as well as imagined future trajectory.
            Actions that lead to successful competion in future trajectory should be awarded high scores.
            If the trajectory is not plausible, you should also give a lower score.
            Score: (choose between: E: impossible, D: unlikely, C: not sure, B: very likely, A: definitely).
            Reason:
        '''
        
        input_prompt = query + "\n Past interactions".join([item[0] + ": " + item[1] for item in history])

        input_prompt += "\n"  #(stop generating if you are not certain about the next observation)"
        
        input_prompt += f"Action: {action_to_evaluate}\n"
        
        input_prompt += "Imagined future trajectory:" 

        input_prompt += evluation_instruction
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_prompt}
        ]
        num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
        while num_of_tokens > self.max_context_length - self.llm_model.max_tokens:
            history = history[1:]
            input_prompt = query + "\n".join([item[0] + ": " + item[1] for item in history])
            

            input_prompt += "\n"  #(stop generating if you are not certain about the next observation)"

            input_prompt += evluation_instruction
            # input_prompt += "\nPlease enter your action:"
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_prompt}
            ]
            num_of_tokens = self.llm_model.num_tokens_from_messages(messages)

        return input_prompt

    def verify_trajectory(self, threshold_high=0.5, threshold_low=0.3):
        
        # after the new execution, provide verification for action rollouts

        # if an action has been executed
        
        # todo: need to merge the same n_gram actions together, also need to double check the similarity implementation
                    
        action_history = [item[1] for item in self.memory if item[0] == "Action"]
        observation_history = [item[1] for item in self.memory if item[0] == "Observation"]
        
        if len(action_history) < 1:
            return
        
        begin_observations_history = observation_history[:-1]
        end_observations_history = observation_history[1:]
        
        for action_id, action in enumerate(action_history):
            
            last_executed_action = action_history[action_id]
            last_begin_observation = begin_observations_history[action_id]
            last_end_observation = end_observations_history[action_id]

            for traj_id, trajectory in enumerate(self.trajectory_pool):
                begin_observation = trajectory[0]["Observation"]
                
                for id, item in enumerate(trajectory):
                    
                    if item["Verified"] == True: # avoid re-verification
                        continue    
                    
                    if "Action" in item and item["Action"] == last_executed_action:
                        if "Observation" in item and item["Observation"] is not None and begin_observation is not None:
                            begin_observation = trajectory[id-1]["Observation"]
                            end_observation = item["Observation"]

                            # if begin_observation is the same as last_begin_observation, and end_observation is not the same as last_end_observation, then the action is not executed as expected
                            if begin_observation is None or end_observation is None:
                                continue
                            begin_observation_similarity = float(torch.max(self.similarity_metric.get_similarity([begin_observation], [last_begin_observation])))
                            end_observation_similarity = float(torch.max(self.similarity_metric.get_similarity([end_observation], [last_end_observation])))
                            
                            if begin_observation_similarity > threshold_high and end_observation_similarity > threshold_high:
                                # this action is executed as expected, verify it as True
                                
                                if "Verified" in trajectory[id]:
                                    self.trajectory_pool[traj_id][id]["Verified"] = True
                                    
                            if begin_observation_similarity > threshold_high and end_observation_similarity < threshold_low:
                                # all the actions after the action should be verified as False
                                
                                for i in range(id, len(trajectory)):
                                    if "Verified" in trajectory[i]:
                                        self.trajectory_pool[traj_id][i]["Verified"] = False
                                        
                                break
                            
                        else:
                            continue
                       
    
    def get_valid_actions(self, action_history):
        
        all_results = []
        
        for traj_id, trajectory in enumerate(self.trajectory_pool):
            
            # trajectory = trajectory[1:] # remove the first action, which is None
            
            start = max(1, len(trajectory) - self.n_gram + 1)
            
            for id in range(start):
                
                n = min([len(trajectory) - id, self.n_gram, len(action_history)+1])
                
                n_gram_list = [trajectory[id+s]["Action"] for s in range(n)]
                n_gram_verification = [trajectory[id+s]["Verified"] for s in range(n)]
                n_gram_reward = [trajectory[id+s]["Reward"] for s in range(n)][-1]
                
                match = (action_history[-n+1:] == n_gram_list[:-1])
                verified = False in n_gram_verification
                
                if match and not verified:
                    all_results.append((n_gram_list[-1], n_gram_reward))
                    
        return all_results

    def lookahead_decision_model(self, reward_threshold=0.5):
        # given the look ahead predictions, make the next action
        
        # different from prior approach, sampling based decision model to simulate energy function
        
        # ! todo: choose the best action when there are multiple options
        
        action_history = [None] + [item[1] for item in self.memory if item[0] == "Action"]
        all_valid_action_values = self.get_valid_actions(action_history)
        
        if len(all_valid_action_values) < 1:
            
            return None
        
        all_valid_values = np.array([item[1] for item in all_valid_action_values])
        all_valid_actions = [item[0] for item in all_valid_action_values]
        
        
        if all_valid_values.max() < reward_threshold:
            
            return None
        
        if self.do_sample: 
            probs = np.exp(all_valid_values/self.select_temperature)
            probs = probs / probs.sum()
            
            all_action_prob_pairs = dict()
            
            for (action, prob) in zip(all_valid_actions, probs):
                if action not in all_action_prob_pairs:
                    all_action_prob_pairs[action] = prob
                else:
                    all_action_prob_pairs[action] += prob
            
            # print in style action:prob, action: prob...
            print("Action probabilities: ", all_action_prob_pairs)
            
            all_valid_actions = list(all_action_prob_pairs.keys())
            probs = list(all_action_prob_pairs.values())
            
            sample = torch.multinomial(torch.tensor(probs), 1).item()
        
            action = all_valid_actions[sample]
        else:
            
            action = all_valid_actions[np.argmax(all_valid_values)]
            
        return action
        

    def reflection_tips(self, reward_threshold=0.5, window_size=2): 
        
        # determine if the model is stuck and requires self-reflection, used sparingly
        
        # first scenario: there are repeat cycles of actions that are not helping the model to reach the goal
        
        reflection = ""
        try:
            action_history = [item[1] for item in self.memory if item[0] == "Action"]
            
            
            if action_history.count(action_history[-1])>1 and action_history.count(action_history[-2])>1:
                # action = f"I have been repeating the same action. I need to perform diverse exploration and try different actions. " # alfworld
                reflection += f"I have been repeating the same action {action_history[-1]}. I need to perform diverse exploration and try different actions. I can use the check valid actions command to find available actions.\n"
                
        except:
            pass
        
        if len(self.trajectory_pool) > 0 :
            
            all_actions = ",".join([trajectory[-1]["Action"] for trajectory in self.trajectory_pool if trajectory[-1]["Action"] is not None])
            
            reflection += f" I have tried actions {all_actions}, but none of them are advancing towards my goal: {self.goal}. I need to revise my approach to be more goal-driven."

        if reflection != "":
            return True, reflection
        else:
            return False, None
     
    def run(self, init_prompt_dict=None):
        
        self.trajectory_pool = []
        
        value_type = self.value_type #"heuristic", "logp", "llm" 

        for i in range(self.max_iters):
            
            # self.trajectory_pool = []
            
            if init_prompt_dict is not None:
                self.init_prompt_dict = init_prompt_dict
                self.instruction = init_prompt_dict['instruction']
                self.examples = init_prompt_dict['examples']

            system_message = self.init_prompt_dict['system_msg']
            input_prompt = self.make_prompt(need_goal=self.need_goal,
                                            check_actions=self.check_actions,
                                            check_inventory=self.check_inventory,
                                            system_message=system_message,
                                            tip = self.reflection_tip)
            
            self.log_example_prompt(input_prompt)
            
            success, action_sequence_samples = self.llm_model.generate_with_config(system_message, input_prompt, self.generation_config)
            
            if success:
                for action_sequence in action_sequence_samples:

                    action_ngram, action = self.parse_action_sequence(action_sequence)
                    
                    reward = 0
                    if value_type == "heuristic":
                        reward = self.get_heuristic_value(action_ngram)
                    elif value_type == "llm":
                        reward = self.get_llm_value(action_ngram)
                    elif value_type == "pddl":
                        reward = self.get_pddl_value(action_ngram)
                    elif value_type == "logp":
                        reward = self.get_logp_value(action_ngram, action_sequence)
                    # elif value_type == "logp":
                    #     only support some models  
                    else:
                        raise NotImplementedError
                    
                    self.update_trajectory_pool(action_sequence, lookahead=True, reward=reward)
                
            # self.verify_trajectory(threshold_high=self.similarity_threshold_high, threshold_low=self.similarity_threshold_low) #don't currently use verify

            # decide upon the best action based on simulated planning
            action = self.lookahead_decision_model(reward_threshold=self.reward_threshold)

            if action is not None:
                self.use_guess_cnt += 1
                return True, action, True
            
            else:
                need_tip, reflection_tip = self.reflection_tips(reward_threshold=self.reward_threshold, window_size=self.window_size)
        
        action = self.lookahead_decision_model(reward_threshold=0)
        action = "check valid actions" if action is None else action
        self.use_guess_cnt += 1
        return True, action, True
       
    
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
        logger = config.get("logger", None)
        
        n_gram = config.get("n_gram", None)
        similarity_threshold_low = config.get("similarity_threshold_low", None)
        similarity_threshold_high = config.get("similarity_threshold_high", None)
        reward_threshold = config.get("reward_threshold", None)
        
        do_sample = config.get("do_sample", True)
        max_world_model_len = config.get("max_world_model_len", 200)
        beam_temperature = config.get("beam_temperature", 0.7)
        select_temperature = config.get("select_temperature", 0.1)
        n_generate_sample = config.get("n_generate_sample", 6)
        
        value_type = config.get("value_type", "heuristic")
        lookahead_length = config.get("lookahead_length", 10)
        
        
        return cls(llm_model, memory_size, examples, instruction, init_prompt_path, system_message, 
                   need_goal, check_actions, check_inventory, use_parser, logger,
                   n_gram=n_gram, 
                   similarity_threshold_high=similarity_threshold_high, 
                   similarity_threshold_low=similarity_threshold_low,
                   reward_threshold=reward_threshold, 
                   do_sample=do_sample, 
                   max_world_model_len=max_world_model_len,
                   beam_temperature=beam_temperature, 
                   select_temperature=select_temperature, 
                   n_generate_sample=n_generate_sample,
                   value_type=value_type,
                   lookahead_length=lookahead_length
                   )
                   
        

class SimilarityMetric(object):
    def __init__(self,
                 model_name='/ssddata/model_hub/all-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.model.eval()

    def get_similarity(self, sequences, source_sequence):
        source_embedding = self.model.encode(source_sequence)
        sequence_embeddings = self.model.encode(sequences)
        
        similarity = util.pytorch_cos_sim(source_embedding, sequence_embeddings)
        return similarity
    
    def get_goal_similarity(self, sequences, source_sequence):
        source_embedding = self.model.encode(source_sequence)
        sequence_embeddings = self.model.encode(sequences)
        
        similarity = util.pytorch_cos_sim(source_embedding, sequence_embeddings)
        return similarity
    
    
class InstructionSimilarityMetric(object):
    def __init__(self,
                 model_name='/root/huggingface/instructor-large'):
        self.model = INSTRUCTOR(model_name)

    def get_goal_similarity(self, sequences, source_sequence):
        source_embedding = self.model.encode(["Represent the goal state for an agent", source_sequence])
        sequences = [["Represent the state for an agent"] + [s] for s in sequences]
        sequence_embeddings = self.model.encode(sequences)
        
        similarity = util.pytorch_cos_sim(source_embedding, sequence_embeddings)
        return similarity
    
    def get_similarity(self, sequences, source_sequence):
        source_embedding = self.model.encode(["Represent the state for agent"] + source_sequence)
        sequences = [["Represent the state for an agent"] + [s] for s in sequences]
        sequence_embeddings = self.model.encode(sequences)
        
        similarity = util.pytorch_cos_sim(source_embedding, sequence_embeddings)
        return similarity