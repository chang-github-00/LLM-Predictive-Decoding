import pdb

from common.registry import registry
# from rouge import Rouge
import json
import torch
import random
import re
import io
import argparse
import numpy as np
from utils.logging.token_logger import token_count, count_flag

@registry.register_algorithm("MPC_Sample")
class MPC_Sample:  # the algorithm should be stateless, and generates a whole plan / code / chain of actions at once.
    def __init__(self,
                 llm_model,
                 task="gsm8k",
                 prompt_path=None,
                 lookahead_thought_length=3,
                 lookahead_token_length=None,    # the length of the lookahead token sequence, default use thought length as evaluation chunk
                 reward_threshold=1.0,
                 beam_size=8,
                 beam_temperature=0.7,
                 select_temperature=0.1,
                 n_generate_sample=8,
                 value_type = "logp",
                 do_sample=True,
                 use_memory=True,
                 ):
        
        self.llm_model = llm_model
        
        if prompt_path is not None:
            self.prompts = json.load(open(prompt_path, 'r'))
        else:
            self.prompts = {}
        
        self.task = task
        
        self.problem_size = 25 if self.task == "gsm8k" else 30
        
        self.n_gram = self.problem_size
        
        self.reward_threshold = reward_threshold 
        self.lookahead_decision_length = lookahead_thought_length
        self.lookahead_token_length = lookahead_token_length
        
        self.do_sample = do_sample
        self.select_temperature = select_temperature
        self.beam_temperature = beam_temperature
        self.select_temperature = select_temperature
        self.beam_size = beam_size
        self.n_generate_sample = n_generate_sample
        self.value_type = value_type
        self.use_memory =use_memory
        
        
        
    def make_prompt(self, prompt, question, memory=None):
        if memory is None:
            memory = self.memory
        else:
            memory = memory
            
        if self.task == "gsm8k":
            with io.StringIO() as f:
                f.write(prompt)
                f.write("\n\n\n\n\n")
                # f.write(f'Q: {self.example}\n\n# solution in Python:\n\n\ndef solution():\n    """{self.example}"""\n')
                f.write(f'Solve this problem following previous examples:\nQ: {question}\n# solution in Python:\n\n')
                model_input = f.getvalue()
            
            with io.StringIO() as f:    
                f.write("def solution():\n")
                for a in memory:
                    if a is not None:
                        f.write(f"{a}")
                answer_prefix = f.getvalue()
                
            return model_input, answer_prefix
        if self.task == "math": 
            with io.StringIO() as f:
                f.write(prompt)
                f.write("\n\n\n\n\n")
                f.write(f'Solve this problem following previous examples:\n\nQ: {question}\n')
                model_input = f.getvalue()

            with io.StringIO() as f:    
                f.write("solution in Python:\n```\n")
                for a in memory:
                    if a is not None:
                        f.write(f"{a}")
                answer_prefix = f.getvalue()
                
            return model_input, answer_prefix
        if self.task in ["humaneval", "mbpp"]:
            with io.StringIO() as f:
                f.write(question)
                model_input = f.getvalue()

            with io.StringIO() as f:    
                f.write(prompt)
                for a in memory:
                    if a is not None:
                        f.write(f"{a}")
                answer_prefix = f.getvalue()
                
            return model_input, answer_prefix
    def update_trajectory_pool(self, outputs, reward=None, id=None, memory=None):
        
        # update the trajectory pool with the generated action rollouts by llm
        
        # if we have the id, means this is for parallel generation, therefore we need to update only the corresponding state for that question
        
        action_rollouts = outputs["action_chain"]
        
        if memory is None:
            memory = self.memory[id] if id is not None else self.memory
        
        history_rollouts = [a for a in memory if a is not None]
        
        item = []
        
        item.append({"Action": None, "Verified": False, "Reward": reward})
        
        for action in history_rollouts:
            item.append({"Action": action, "Verified": True, "Reward": reward})
            
        for action in action_rollouts:
            item.append({"Action": action, "Verified": None, "Reward": reward})
        
        if id is not None:
            self.trajectory_pool[id].append(item)
        else:
            self.trajectory_pool.append(item)
        
    def parse_action_sequence(self, action_output, parse_prefix="", id=None, memory=None): 
        
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
        def _clean_action(action):
            action = action.replace("`\n", "")
            return action + "\n"
        
        prefix = parse_prefix
        
        if type(action_output) == str: # no logprob information
            
            if memory is None:
                memory = self.memory[id] if id is not None else self.memory
            
            all_prefix = [prefix] + [a for a in memory if a is not None]
            
            if "mistral" not in self.llm_model.engine.lower(): # mistral don't know when to stop and easily generate more than one prefix...
                # actually this code is not quite useful for llama3 anyway, perhaps could remove it.
                for prefix in all_prefix:
                    if prefix in action: # added, in case there is repeat of prompt inside the generation
                        action = action.split(prefix)[1]
            action = action.lstrip('\n')
            
            if action == "":
                return None, None
            
            # Here is the start of the action chain:
            if '\n' in action:
                all_actions = action.split('\n')
            else:
                all_actions = [action]
            
            first_action = all_actions[0] + '\n'
            action_chain = [a + '\n' for a in all_actions][: self.lookahead_decision_length] # only keep the first n actions
            
            return {"action": first_action, "action_chain": action_chain}, first_action
        
        elif type(action_output) == dict: # need logprob information
            
            action_text_output = action_output["text"]
            action_logprobs = action_output["logprobs"]
            action_tokens = action_output["tokens"]
            
            if memory is None:
                memory = self.memory[id] if id is not None else self.memory
            
            all_prefix = [prefix] + [a for a in memory if a is not None]
            
            token_start, token_end = 0, -1
            action = action_text_output
            
            if "mistral" not in self.llm_model.engine.lower(): # mistral don't know when to stop and easily generate more than one prefix...
                # actually this code is not quite useful for llama3 anyway, perhaps could remove it.
                for prefix in all_prefix:
                    if prefix in action: # added, in case there is repeat of prompt inside the generation
                        action = action.split(prefix)[1]
                    
             # remove all '\n' in the beginning
            action = action.lstrip('\n')
            
            if action == "":
                return None, None
            
            if self.lookahead_token_length is not None: # limit the length of the lookahead token sequence as a chunk for lookahead
                
                token_start, token_end = _get_start_end_token_id(action_text_output, action, action_tokens)
                
                token_end = min(token_end, token_start + self.lookahead_token_length)
                
                action = "".join(action_tokens[token_start:token_end])
                
            
                # Here is the start of the action chain:
                if '\n' in action:
                    all_actions = action.split('\n')
                else:
                    all_actions = [action]
                
                first_action = all_actions[0] + '\n'
                action_chain = [a + '\n' for a in all_actions]
                
                action_logprobs = action_logprobs[token_start:token_end]
                
                action_prob = np.exp(sum(action_logprobs)) 
                if token_end - token_start > 0:
                    action_length = token_end - token_start
                else:
                    action_length = len(action_tokens)
                    
                if count_flag:
                    token_count.add_generation_tokens(action_length)
                    
                action_prob = action_prob ** (1 / action_length) # normalize by the length of the action
                
                action_chain = [a for a in action_chain if a.strip()!=""]
            
                return {"action": first_action, "action_chain": action_chain, "action_prob": action_prob}, first_action
            
            else: # limit the number of thoughts in the lookahead
                
                # Here is the start of the action chain:
                if '\n' in action:
                    all_actions = action.split('\n')
                else:
                    all_actions = [action]
                
                first_action = all_actions[0] + '\n'
                action_chain = [_clean_action(a) for a in all_actions][: self.lookahead_decision_length] # only keep the first n actions
                
                token_start, token_end = _get_start_end_token_id(action_text_output, "".join(action_chain), action_tokens)
                
                action_logprobs = action_logprobs[token_start:token_end]
                
                action_prob = np.exp(sum(action_logprobs))
                if token_end - token_start > 0:
                    action_length = token_end - token_start
                else:
                    action_length = len(action_tokens)
                
                if count_flag:
                    token_count.add_generation_tokens(action_length)
                    
                action_prob = action_prob ** (1 / action_length)  # normalize by the length of the action
                action_chain = [a for a in action_chain if a.strip()!=""]
                
                return {"action": first_action, "action_chain": action_chain, "action_prob": action_prob}, first_action
        
        else:
            raise NotImplementedError
    
    def get_valid_actions(self, action_history, id=None ):
        
        def is_valid_python(code):
            with io.StringIO() as f:
                f.write("def solution():\n")
                # iterate through the state
                for a  in self.memory:
                    if a is not None:
                        f.write(f"{a}\n")
                f.write(code+"\n")
                full_code = f.getvalue()
            try:
                # Try to compile the string of code.
                # If the code compiles without raising a SyntaxError, it is valid Python code.
                compile(code, "<string>", "exec")
                return True
            except SyntaxError:
                try: 
                    compile(full_code, "<string>", "exec")
                    return True
                except SyntaxError:
                    pass
                return False
        
        all_results = []
        
        if id is not None:
            trajectory_pool = self.trajectory_pool[id]
        else:
            trajectory_pool = self.trajectory_pool
        
        for traj_id, trajectory in enumerate(trajectory_pool):
            
            # trajectory = trajectory[1:] # remove the first action, which is None
            
            start = max(1, len(trajectory) - self.n_gram + 1)
            
            for id in range(start):
                
                n = min([len(trajectory) - id, self.n_gram, len(action_history)+1])
                
                n_gram_list = [trajectory[id+s]["Action"] for s in range(n)]
                # n_gram_verification = [trajectory[id+s]["Verified"] for s in range(n)]
                n_gram_reward = [trajectory[id+s]["Reward"] for s in range(n)][-1]
                
                match = (action_history[-n+1:] == n_gram_list[:-1])
                # verified = False in n_gram_verification
                
                if match:
                    all_results.append((n_gram_list[-1], n_gram_reward))
                    
        # for coding tasks, screen out the actions that are not valid
        if self.task == "math": #don't allow generating ``` in code as it will end the code and cannot fix return
            all_results = [item for item in all_results if "```" not in item[0]]
        return all_results
    
      
    def lookahead_decision_model(self, reward_threshold=1.0):
        # given the look ahead predictions, make the next action
        
        # ! todo: choose the best action when there are multiple options
        
        action_history = [None] + [action for action in self.memory if action is not None]
        
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
            # print("Action probabilities: ", all_action_prob_pairs)
            
            all_valid_actions = list(all_action_prob_pairs.keys())
            probs = list(all_action_prob_pairs.values())
            
            sample = torch.multinomial(torch.tensor(probs), 1).item()
        
            action = all_valid_actions[sample]
            
        else:
            
            action = all_valid_actions[np.argmax(all_valid_values)]
            
            
        return action
    
    def parallel_lookahead_decision_model(self, all_ind, reward_threshold=1.0):
        # given the look ahead predictions, make the next action
        
        # ! todo: choose the best action when there are multiple options
        all_decided_actions = []
        for ind in all_ind:
            action_history = [None] + [action for action in self.memory[ind] if action is not None]
        
            all_valid_action_values = self.get_valid_actions(action_history, id=ind)
        
            if len(all_valid_action_values) < 1:
            
                all_decided_actions.append(None)
                continue
        
            all_valid_values = np.array([item[1] for item in all_valid_action_values])
            all_valid_actions = [item[0] for item in all_valid_action_values]
        
        
            if all_valid_values.max() < reward_threshold:
            
                all_decided_actions.append(None)
                continue
        
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
                # print("Action probabilities: ", all_action_prob_pairs)
            
                all_valid_actions = list(all_action_prob_pairs.keys())
                probs = list(all_action_prob_pairs.values())
            
                sample = torch.multinomial(torch.tensor(probs), 1).item()
        
                action = all_valid_actions[sample]
            
            else:
            
                action = all_valid_actions[np.argmax(all_valid_values)]
                
            all_decided_actions.append(action)
            
            
        return all_decided_actions

    def search_lookahead_decision_model(self, all_ind, all_nodes, reward_threshold=1.0, expand_width=2):
        # given the look ahead predictions, make the next action
        
        # ! todo: choose the best action when there are multiple options
        all_decided_actions = []
        for ind in all_ind:
            action_history = [None] + [action for action in self.memory[ind] if action is not None]
        
            all_valid_action_values = self.get_valid_actions(action_history, id=ind)
        
            if len(all_valid_action_values) < 1:
            
                all_decided_actions.append(None)
                continue
        
            all_valid_values = np.array([item[1] for item in all_valid_action_values])
            all_valid_actions = [item[0] for item in all_valid_action_values]
        
        
            if all_valid_values.max() < reward_threshold:
            
                all_decided_actions.append(None)
                continue
        
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
                # print("Action probabilities: ", all_action_prob_pairs)
            
                all_valid_actions = list(all_action_prob_pairs.keys())
                probs = list(all_action_prob_pairs.values())
            
                sample = torch.multinomial(torch.tensor(probs), 1).item()
        
                action = all_valid_actions[sample]
            
            else:
            
                action = all_valid_actions[np.argmax(all_valid_values)]
                
            all_decided_actions.append(action)
            
            
        return all_decided_actions

    
    def reflection_tips(self, reward_threshold=0.5, window_size=2): 
        
        # determine if the model is stuck and requires self-reflection, used sparingly
                
        reflection = ""
        
        if len(self.trajectory_pool) > 0 :
            
            all_actions = ",".join(list(set([trajectory[-1]["Action"] for trajectory in self.trajectory_pool if trajectory[-1]["Action"] is not None])))
            
            question = self.prompts["question"]
            reflection += f"I have generated {all_actions}, but none of them are correct. I need to revise them to solve the problem {question}."
            
            # if reflection is multiline, follow the format of python comment
            indent = "    "
            reflection = indent + "# " + reflection.replace("\n", "\n# ")

        if reflection != "":
            return True, reflection
        else:
            return False, None

    def run(self, question, prompts=None, **kwargs):
        
        if "end_suffix" in kwargs:
            end_suffix = kwargs["end_suffix"]
        else:
            end_suffix = None
            
        if prompts is not None:
            self.prompts = prompts
        self.prompts["question"] = question
        
        self.trajectory_pool = []
        
        args = {
            "n_generate_sample":self.beam_size,
            "max_iters": self.problem_size,
            "max_tokens": 500,# if self.lookahead_token_length is None else self.lookahead_token_length,
            "temperature": self.beam_temperature,
            "top_p": 1.0,
            "stop": [],            
            "logprobs": (self.value_type == "logp"),
            "value_type": self.value_type
        }
        
        args = argparse.Namespace(**args)
        
        
        generation_config = {"n": args.n_generate_sample, 
                            "stop": args.stop, 
                            "top_p": args.top_p,
                            "max_tokens": args.max_tokens, 
                            "temperature": args.temperature,
                            "do_sample": True,
                            "logprobs": args.logprobs}
        
        all_iter = 0
        iter = 0
        
        
        self.memory = [None]*self.problem_size
        
        reflection_tips = ""

        while iter < args.max_iters:
            if not self.use_memory:
                self.trajectory_pool = [] # don't keep memory, but re-init the trajectory pool every time
            
            input_prompt, answer_prefix = self.make_prompt(self.prompts["prompt"], question)
            system_message = self.prompts["system_msg"]
            success, action_sequence_samples = self.llm_model.generate_with_config(system_message, input_prompt, generation_config, answer_prefix=answer_prefix)
            
            if success:
                for action_sequence in action_sequence_samples:
                    if self.task == "gsm8k":
                        parse_prefix = "def solution():\n"
                    if self.task == "math":
                        parse_prefix = "Null"
                    if self.task in ["humaneval", "mbpp"]:
                        parse_prefix = self.prompts["prompt"]
                    processed_output, action = self.parse_action_sequence(action_sequence, parse_prefix=parse_prefix)

                    if action is None: continue
                    reward = 0
                    if args.value_type == "logp":
                        reward = processed_output["action_prob"]
                    else:
                        raise NotImplementedError
                    
                    self.update_trajectory_pool(processed_output, reward=reward)
            else:
                print("Failed to generate action sequence.")
                return False, None
                
            reward_threshold = self.reward_threshold if reflection_tips == "" else 0  # if reflection is needed, lower the threshold so that the model won't get stuck
            action = self.lookahead_decision_model(reward_threshold=self.reward_threshold)
            
            if action is not None:
                
                self.memory[iter] = action
                
                iter += 1
                
                reflection_tips = self.reflection_tips(reward_threshold=self.reward_threshold)
                
                if iter > args.max_iters:
                    break
                
                if end_suffix is not None and end_suffix in action.strip():
                    break
            else:
                reflection_tips = self.reflection_tips(reward_threshold=self.reward_threshold)
                if reflection_tips[0]:
                    self.memory[iter] = reflection_tips[1]
                    break
                      
        if success:
            with io.StringIO() as f:
                if self.task == "gsm8k":
                    f.write("def solution():\n")
                # iterate through the state
                for a  in self.memory:
                    if a is not None:
                        f.write(f"{a}\n")

                full_output = f.getvalue()

            return True, full_output
            
        return False, None
    
    def parallel_run(self, questions, prompts=None, **kwargs): # code for original vllm
        
        if self.n_generate_sample == 1:
            return self.parallel_run_single(questions, prompts=prompts, **kwargs)
        elif self.n_generate_sample > 1:
            return self.parallel_run_multiple(questions, prompts=prompts, **kwargs)
        else:
            raise ValueError("n_generate_sample must be greater than 0.")
    
    def parallel_run_single(self, questions, prompts=None, **kwargs): # code for original vllm
        
        args = {
            "n_generate_sample":self.beam_size,
            "max_iters": self.problem_size,
            "max_tokens": 500,# if self.lookahead_token_length is None else self.lookahead_token_length,
            "temperature": self.beam_temperature,
            "top_p": 1.0,
            "stop": [],            
            "logprobs": (self.value_type == "logp"),
            "value_type": self.value_type
        }
        
        args = argparse.Namespace(**args)
        
        
        generation_config = {"n": args.n_generate_sample, 
                            "stop": args.stop, 
                            "top_p": args.top_p,
                            "max_tokens": args.max_tokens, 
                            "temperature": args.temperature,
                            "do_sample": True,
                            "logprobs": args.logprobs}
        
        if "end_suffix" in kwargs:
            end_suffix = kwargs["end_suffix"]
        else:
            end_suffix = None
            
        if prompts is not None:
            self.prompts = prompts
        
        self.trajectory_pool = {}
        self.memory = {}
        
        for id, question in enumerate(questions):
            self.trajectory_pool[id] = []
            self.memory[id] = [None]*self.problem_size

        all_iter = 0
        iter = 0

        while iter < args.max_iters:
            all_input_prompts = []
            all_answer_prefixes = []
            all_system_messages = []
            all_index = []
            for id in range(len(questions)):
                if not self.use_memory:
                    self.trajectory_pool[id] = []

                ended = False
                for action in self.memory[id]:
                    if action is not None and end_suffix is not None and end_suffix in action.strip().lower(): 
                        ended = True    
                        break
                    
                if not ended: # as not all samples have the same number of steps, we only inference those that have not finished
                    if self.task in ["gsm8k", "math"]:
                        input_prompt, answer_prefix = self.make_prompt(self.prompts["prompt"], questions[id], memory=self.memory[id])
                    elif self.task in ["humaneval", "mbpp"]: # coding tasks have different prompts for each question
                        input_prompt, answer_prefix = self.make_prompt(self.prompts["prompt"][id], questions[id], memory=self.memory[id])
                    else:
                        raise NotImplementedError
                    all_input_prompts.append(input_prompt)
                    all_answer_prefixes.append(answer_prefix)
                    all_system_messages.append(self.prompts["system_msg"])
                    all_index.append(id)
            
            if len(all_index) == 0: # all questions have finished
                break
            
            success, action_sequence_samples = self.llm_model.parallel_generate_with_config(all_system_messages, all_input_prompts, generation_config, answer_prefixes=all_answer_prefixes)
            
            if success:
                for id, action_sequence_sample in zip(all_index, action_sequence_samples):
                    
                    if action_sequence_sample is None: # no action sequence generated when length exceeds max_tokens
                        continue
                    
                    for action_sequence in action_sequence_sample:
                        if self.task == "gsm8k":
                            parse_prefix = "def solution():\n"
                        if self.task == "math":
                            parse_prefix = "Null"
                        if self.task in ["humaneval", "mbpp"]:
                            parse_prefix = self.prompts["prompt"][id]
                        processed_output, action = self.parse_action_sequence(action_sequence, parse_prefix=parse_prefix, id=id)

                        if action is None: continue
                        reward = 0
                        if args.value_type == "logp":
                            reward = processed_output["action_prob"]
                        else:
                            raise NotImplementedError
                        
                        self.update_trajectory_pool(processed_output, reward=reward, id=id)    
            else:
                print("Failed to generate action sequence.")
                return False, None
                
            # reward_threshold = self.reward_threshold if reflection_tips == "" else 0  # if reflection is needed, lower the threshold so that the model won't get stuck
            actions = self.parallel_lookahead_decision_model(reward_threshold=self.reward_threshold, all_ind=all_index)
            
            for id, action in zip(all_index, actions):
                if action is not None:
                    step_id = self.memory[id].index(None)
                    self.memory[id][step_id] = action
            
            iter += 1
            if iter > args.max_iters:
                break
                
             
        if success:
            all_outputs = []
            for id in range(len(questions)):
                with io.StringIO() as f:
                    if self.task == "gsm8k":
                        f.write("def solution():\n")
                    elif self.task == "math":
                        pass
                    elif self.task in ["humaneval", "mbpp"]:
                        f.write(self.prompts["prompt"][id])
                    else:
                        raise NotImplementedError
                    # iterate through the state
                    for a  in self.memory[id]:
                        if a is not None:
                            f.write(f"{a}\n")

                    full_output = f.getvalue()
                all_outputs.append(full_output)
            
            return True, all_outputs
            
        return False, None
    
    
    def parallel_run_multiple(self, questions, prompts=None, **kwargs): # code for original vllm
        
        args = {
            "n_generate_sample":self.beam_size,
            "max_iters": self.problem_size,
            "max_tokens": 500,# if self.lookahead_token_length is None else self.lookahead_token_length,
            "temperature": self.beam_temperature,
            "top_p": 1.0,
            "stop": [],            
            "logprobs": (self.value_type == "logp"),
            "value_type": self.value_type
        }
        
        args = argparse.Namespace(**args)
        
        
        generation_config = {"n": args.n_generate_sample, 
                            "stop": args.stop, 
                            "top_p": args.top_p,
                            "max_tokens": args.max_tokens, 
                            "temperature": args.temperature,
                            "do_sample": True,
                            "logprobs": args.logprobs}
        
        if "end_suffix" in kwargs:
            end_suffix = kwargs["end_suffix"]
        else:
            end_suffix = None
            
        if prompts is not None:
            self.prompts = prompts
        
        self.trajectory_pool = {}
        self.memory = {}
        
        # make the questions repeat n_generate_sample times
        questions = [q  for q in questions for _ in range(self.n_generate_sample)]
        if type(self.prompts["prompt"]) == list:
            self.prompts["prompt"] = [p for p in self.prompts["prompt"] for _ in range(self.n_generate_sample)]
            
        for id, question in enumerate(questions):
            self.trajectory_pool[id] = []
            self.memory[id] = [None]*self.problem_size

        all_iter = 0
        iter = 0

        while iter < args.max_iters:
            all_input_prompts = []
            all_answer_prefixes = []
            all_system_messages = []
            all_index = []
            for id in range(len(questions)):
                if not self.use_memory:
                    self.trajectory_pool[id] = []

                ended = False
                for action in self.memory[id]:
                    if action is not None and end_suffix is not None and action.strip().startswith(end_suffix): 
                        ended = True    
                        break
                    
                if not ended: # as not all samples have the same number of steps, we only inference those that have not finished
                    if self.task in ["gsm8k", "math"]:
                        input_prompt, answer_prefix = self.make_prompt(self.prompts["prompt"], questions[id], memory=self.memory[id])
                    elif self.task in ["humaneval", "mbpp"]: # coding tasks have different prompts for each question
                        input_prompt, answer_prefix = self.make_prompt(self.prompts["prompt"][id], questions[id], memory=self.memory[id])
                    else:
                        raise NotImplementedError
                    all_input_prompts.append(input_prompt)
                    all_answer_prefixes.append(answer_prefix)
                    all_system_messages.append(self.prompts["system_msg"])
                    all_index.append(id)
            
            if len(all_index) == 0: # all questions have finished
                break
            
            success, action_sequence_samples = self.llm_model.parallel_generate_with_config(all_system_messages, all_input_prompts, generation_config, answer_prefixes=all_answer_prefixes)
            
            if success:
                for id, action_sequence_sample in zip(all_index, action_sequence_samples):
                    
                    if action_sequence_sample is None: # no action sequence generated when length exceeds max_tokens
                        continue
                    
                    for action_sequence in action_sequence_sample:
                        if self.task == "gsm8k":
                            parse_prefix = "def solution():\n"
                        if self.task == "math":
                            parse_prefix = "Null"
                        if self.task in ["humaneval", "mbpp"]:
                            parse_prefix = self.prompts["prompt"][id]
                        processed_output, action = self.parse_action_sequence(action_sequence, parse_prefix=parse_prefix, id=id)

                        if action is None: continue
                        reward = 0
                        if args.value_type == "logp":
                            reward = processed_output["action_prob"]
                        else:
                            raise NotImplementedError
                        
                        self.update_trajectory_pool(processed_output, reward=reward, id=id)    
            else:
                print("Failed to generate action sequence.")
                return False, None
                
            # reward_threshold = self.reward_threshold if reflection_tips == "" else 0  # if reflection is needed, lower the threshold so that the model won't get stuck
            actions = self.parallel_lookahead_decision_model(reward_threshold=self.reward_threshold, all_ind=all_index)
            
            for id, action in zip(all_index, actions):
                if action is not None:
                    step_id = self.memory[id].index(None)
                    self.memory[id][step_id] = action
            
            iter += 1
            if iter > args.max_iters:
                break
                
             
        if success:
            all_outputs = []
            for id in range(len(questions)):
                with io.StringIO() as f:
                    if self.task == "gsm8k":
                        f.write("def solution():\n")
                    elif self.task == "math":
                        pass
                    elif self.task in ["humaneval", "mbpp"]:
                        f.write(self.prompts["prompt"][id])
                    else:
                        raise NotImplementedError
                    # iterate through the state
                    for a  in self.memory[id]:
                        if a is not None:
                            f.write(f"{a}\n")

                    full_output = f.getvalue()
                all_outputs.append(full_output)
            
            # merge the outputs for each question
            all_outputs = [all_outputs[i:i+self.n_generate_sample] for i in range(0, len(all_outputs), self.n_generate_sample)]
            return True, all_outputs
            
        return False, None
    
    @classmethod
    def from_config(cls, llm_model, config):
        return cls(llm_model, 
                   task=config.get("task", "gsm8k"),
                   prompt_path=config.get("prompt_path", None),
                   lookahead_thought_length=config.get("lookahead_thought_length", 3),
                   lookahead_token_length=config.get("lookahead_token_length", None),
                   reward_threshold=config.get("reward_threshold", 1.0),
                   beam_size=config.get("beam_size", 8),
                   beam_temperature=config.get("beam_temperature", 0.7),
                   select_temperature=config.get("select_temperature", 0.1),
                   n_generate_sample=config.get("n_generate_sample", 8),
                   value_type=config.get("value_type", "logp"),
                   do_sample=config.get("do_sample", True),
                   use_memory=config.get("use_memory", True)
                   )

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        
    def add_child(self, child):
        self.children.append(child)
    
    
    def get_state(self):
        return self.state
    
    def get_parent(self):
        return self.parent
    
    def get_children(self):
        return self.children
    
    def get_trajectory(self):
        trajectory = []
        node = self
        while node is not None:
            trajectory.append(node.get_state())
            node = node.get_parent()
        trajectory.reverse()
        return trajectory
    

class SearchTree:
    def __init__(self, root):
        self.root = root
        self.nodes = []
        self.nodes.append(root)
        
    def add_node(self, node):
        self.nodes.append(node)
    
    
    def get_all_trajectories_from_root(self):
        trajectories = []
        trajectories_nodes = []
        
        # search for all paths from root to leaf
        for node in self.nodes:
            trajectory = []
            trajectory_nodes = []
            while node is not None:
                trajectory.append(node.get_state())
                trajectory_nodes.append(node)
                node = node.get_parent()
            trajectory.reverse()
            trajectory_nodes.reverse()
            trajectories.append(trajectory)
            trajectories_nodes.append(trajectory_nodes)
        return trajectories, trajectories_nodes