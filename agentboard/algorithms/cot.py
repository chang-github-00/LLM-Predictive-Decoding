import pdb

from common.registry import registry
# from rouge import Rouge
import json
import random
import re
import io
import argparse
import torch
from utils.logging.token_logger import token_count, count_flag

@registry.register_algorithm("COT")
class COT:  # the agent should receive goal, state and action, then return the next state
    def __init__(self,
                 llm_model,
                 prompt_path
                 ):
        
        self.llm_model = llm_model
        
        self.prompts = json.load(open(prompt_path, 'r'))
    
    def make_prompt(self, question, prompt):
        query = ""
        query += prompt["instruction"] + '\n'
        query += "\nHere is an example:\n" + prompt["examples"][0] + '\n'
        
        input_prompt = query + question
        input_prompt = input_prompt 

        return input_prompt

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
        
        system_message = self.prompts["system_msg"]
        input_prompt, answer_prefix = self.make_prompt(question, self.prompts)
        success, results = self.llm_model.generate( system_message, input_prompt, answer_prefix=answer_prefix)
        if success:
            success, answer = self.llm_model.generate( system_message, f"{input_prompt}\nThought:{results}\noutput=")
        
        if success:
            result_lists = self.parse_integer_lists(answer)
            if len(result_lists) > 0:
                return True, result_lists[-1]
        return False, None

    @classmethod
    def from_config(cls, llm_model, config):
        return cls(llm_model, config["prompt_path"])



@registry.register_algorithm("COT_Reward")
class COT_Reward :  # the algorithm should be stateless, and generates a whole plan / code / chain of actions at once.
    def __init__(self,
                 llm_model,
                 reward_model=None,
                 prompt_path=None,
                 beam_temperature=0.7,
                 n_generate_sample=8,
                 do_sample=True,
                 beam_search=False,
                 task="gsm8k",
                 value_type = "reward",
                 result_type = "rank", # "vote"
                 ):
        
        self.llm_model = llm_model
        
        if prompt_path is not None:
            self.prompts = json.load(open(prompt_path, 'r'))
        else:
            self.prompts = {}
        
        self.task = task
        
        self.do_sample = do_sample
        self.beam_temperature = beam_temperature
        self.n_generate_sample = n_generate_sample
        self.beam_search = beam_search
        
        self.value_type = value_type
        if self.value_type == "reward":
            self.reward_model = reward_model
        else:
            self.reward_model = None
        self.result_type = result_type
        if self.n_generate_sample == 1:
            self.reward_model = None
        
    def make_prompt(self, prompt, question):
        if self.task == "gsm8k":
            with io.StringIO() as f:
                f.write(prompt)
                f.write("\n\n\n\n\n")
                # f.write(f'Q: {self.example}\n\n# solution in Python:\n\n\ndef solution():\n    """{self.example}"""\n')
                f.write(f'Solve this problem following previous examples:\nQ: {question}\n')
                model_input = f.getvalue()
            
            with io.StringIO() as f:    
                f.write("A: ")
                answer_prefix = f.getvalue()
            return model_input, answer_prefix
        else:
            raise NotImplementedError
        
    def get_reward_parallel(self, action_sequences, questions):
        # get the reward for the action sequence
        assert self.reward_model is not None, "Reward model is not provided."
        
        good_token = '+'
        bad_token = '-'
        step_tag = 'ки'
        
        # construct the input for the reward model in the following format:
        # output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки """ # 18 is right
        
        full_prompts = []
        
        for action_sequences, question in zip(action_sequences, questions):
            full_prompt = []
            index = 0
            for action in action_sequences:
                if action is None:
                    continue
                if "answer" in action:
                    full_prompt.append(f"{action}ки\n")
                if action is not None:
                    full_prompt.append(f"Step {index+1}: {action} ки\n")
                    index += 1
            full_prompt = "".join(full_prompt)
            full_prompt = f"{question} {full_prompt}"
            full_prompts.append(full_prompt)
        
        all_logprobs,  all_tokens = self.reward_model.encode(full_prompts)
        
        if count_flag:
            for tokens in all_tokens:
                token_count.add_reward_tokens(len(tokens), 1)
        
        
        all_rewards = []
        for (logprobs, tokens) in zip(all_logprobs, all_tokens):
            tag_token_index = [i+1 for i, token in enumerate(tokens) if step_tag in token]
            results = []
            for token_index in tag_token_index:
                logprob = logprobs[token_index]
                good_score = 0.0
                bad_score = 0.0
                for key, value in logprob.items():
                    if good_token in key and good_score == 0.0:
                        good_score = value
                    if bad_token in key and bad_score == 0.0:
                        bad_score = value
                normalized_good_score = torch.softmax(torch.tensor([good_score, bad_score]), dim=0)[0].item()
                results.append(normalized_good_score)
            all_rewards.append(results[-1])
        return all_rewards
    
    
    
    def parse_action_sequence(self, action_output, parse_prefix="", id=None, memory=None): 
        if self.value_type == "reward":
            action_output = action_output.split(".")
            action_output = [action for action in action_output if action.strip() != ""]
            action_output = [action+". " for action in action_output]
            if len(action_output) == 0:
                return None
            return action_output
        else:
            raise NotImplementedError
    
    def parse_answer(self, answer):
        match = re.search(r'answer is (\d+)', answer)
        if match is not None:
            return eval(match.group(1))
        else:
            return None
    
    def return_best_answer(self, results, rewards):
        if self.result_type == "vote":
            rewards = [reward / sum(rewards) for reward in rewards]
            
            scores = dict()
            results_answer = dict()
            for (result, reward) in zip(results, rewards):
                answer = self.parse_answer(result)
                if answer is None:
                    continue
                if answer not in scores:
                    scores[answer] = 0
                if answer not in results_answer:
                    results_answer[answer] = []
                scores[answer] += reward
                results_answer[answer].append(result)
            
            if len(scores) == 0:
                return results
            
            best_answer = max(scores, key=scores.get)
            all_outputs = results_answer[best_answer]
            
            return all_outputs
        
        elif self.result_type == "rank":
            if len(rewards) == 0:
                return results
            best_index = rewards.index(max(rewards))
            all_outputs = [results[best_index]]
            
            return all_outputs
        else:
            raise NotImplementedError
        
    def record_num_tokens(self, system_messages, input_prompts, answer_prefixes):
        if not count_flag:
            return
        
        for system_message, input_prompt, answer_prefix in zip(system_messages, input_prompts, answer_prefixes):
            text = system_message + input_prompt + answer_prefix
            token = self.llm_model.tokenizer.encode(text)
            token_count.add_prompt_tokens(len(token), self.n_generate_sample)
            
    def record_generated_num_tokens(self, results):
        if not count_flag:
            return
        
        for res in results:
            for result in res:
                token = self.llm_model.tokenizer.encode(result)
                token_count.add_generation_tokens(len(token))
            
        
    def run(self, question, prompts=None, **kwargs):
        
        if prompts is not None:
            self.prompts = prompts
        self.prompts["question"] = question
        
        self.trajectory_pool = []
        
        args = {
            "n_generate_sample":self.n_generate_sample,
            "max_tokens": 500,
            "temperature": self.beam_temperature,
            "top_p": 1.0,
            "stop": [],            
            "logprobs": 0,
            "use_beam_search": self.beam_search
        }
        
        args = argparse.Namespace(**args)
        
        
        generation_config = {"n": args.n_generate_sample, 
                            "stop": args.stop, 
                            "top_p": args.top_p,
                            "max_tokens": args.max_tokens, 
                            "temperature": args.temperature,
                            "do_sample": True,
                            "logprobs": args.logprobs,
                            "use_beam_search": args.use_beam_search
                            }
        
        if self.task == "humaneval":
            input_prompt = question
            answer_prefix = self.prompts["prompt"]
        else:
            input_prompt = self.make_prompt(self.prompts["prompt"], question)
            answer_prefix = None
        
        system_message = self.prompts["system_msg"]
        success, results = self.llm_model.generate_with_config(system_message, input_prompt, generation_config, answer_prefix)
        
        if not success:
            
            return False, None
        
        if self.reward_model is None:
            if args.n_generate_sample == 1: 
                all_outputs = [results]
            else:
                all_outputs = results
        
        if self.reward_model is not None:
            result_sequences = [self.parse_action_sequence(result) for result in results]
            rewards = self.get_reward_parallel([result_sequences], [question])
            
            all_outputs = self.return_best_answer(results, rewards)
            
        return True, all_outputs
            
    def parallel_run(self, questions, prompts=None, **kwargs):
        
        args = {
            "n_generate_sample":self.n_generate_sample,
            "max_tokens": 500,
            "temperature": self.beam_temperature,
            "top_p": 1.0,
            "stop": [],            
            "logprobs": 0,
            "use_beam_search": self.beam_search
        }
        
        args = argparse.Namespace(**args)
        
        
        generation_config = {"n": args.n_generate_sample, 
                            "stop": args.stop, 
                            "top_p": args.top_p,
                            "max_tokens": args.max_tokens, 
                            "temperature": args.temperature,
                            "do_sample": True,
                            "logprobs": args.logprobs,
                            "use_beam_search": args.use_beam_search
                            }
        
        
        if prompts is not None:
            self.prompts = prompts
        
        all_prompts = []
        all_answer_prefixes = []
        
        for i in range(len(questions)):
            self.prompts["question"] = questions[i]
            input_prompt, answer_prefix = self.make_prompt(self.prompts["prompt"], questions[i])
            all_prompts.append(input_prompt)
            all_answer_prefixes.append(answer_prefix)

        all_system_messages = [self.prompts["system_msg"]] * len(all_prompts)
        
        success, results = self.llm_model.parallel_generate_with_config(all_system_messages, all_prompts, generation_config, all_answer_prefixes)
        
        self.record_num_tokens(all_system_messages, all_prompts, all_answer_prefixes)
        self.record_generated_num_tokens(results)
        
        if not success:
            
            return False, None
        
        if self.reward_model is None:
            if args.n_generate_sample == 1: 
                all_outputs = [[result] for result in results]
            else:
                all_outputs = results
        
        if self.reward_model is not None:
            all_outputs = []
            
            record_result_id = []
            all_result_sequences = []
            all_reward = []
                
            for i in range(len(questions)):
                for result in results[i]:
                    result_sequences = self.parse_action_sequence(result)
                    if result_sequences is None:
                        continue
                    all_result_sequences.append(result_sequences)
                    record_result_id.append(i)
                    
            all_questions_for_rm = [questions[i] for i in record_result_id]
            all_reward = self.get_reward_parallel(all_result_sequences, all_questions_for_rm)

            # group the reward by the question
            for i in range(len(questions)):
                rewards = [all_reward[j] for j in range(len(all_reward)) if record_result_id[j] == i]
                result_sequences = [all_result_sequences[j] for j in range(len(all_result_sequences)) if record_result_id[j] == i]
                result_strings = ["".join(result_sequence) for result_sequence in result_sequences]
                
                output = self.return_best_answer(result_strings, rewards)
                all_outputs.append(output)
            assert len(all_outputs) == len(questions)
        
        return True, all_outputs
    
    @classmethod
    def from_config(cls, llm_model, config, reward_model=None):
        return cls(llm_model, 
                   reward_model=reward_model,
                   prompt_path=config.get("prompt_path", None),
                   beam_temperature=config.get("beam_temperature", 0.7),
                   n_generate_sample=config.get("n_generate_sample", 8),
                   do_sample=config.get("do_sample", True),  
                   task=config.get("task", "gsm8k"),
                   beam_search=config.get("beam_search", False),
                   value_type=config.get("value_type", "reward"),
                   result_type=config.get("result_type", "rank")
                   )
        