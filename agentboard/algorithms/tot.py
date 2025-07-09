import pdb

from common.registry import registry
# from rouge import Rouge
import json
import random
import re
import logging
import argparse

logging.basicConfig(filename='tot_50it.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

@registry.register_algorithm("TOT")
class TOT:  # the agent should receive goal, state and action, then return the next state
    def __init__(self,
                 llm_model,
                 prompt_path
                 ):
        
        self.llm_model = llm_model
        self.world_model = llm_model
        self.prompts = json.load(open(prompt_path, 'r'))
        self.system_message = self.prompts["system_msg"]
        
        self.task = "dp" if "dp" in prompt_path else "pf"
    
    def make_prompt(self, node):
        query = ""
        query += self.prompts["instruction"] + '\n'
        query += "\nHere is an example:\n" + self.prompts["examples"][0] + '\n'
        
        input_prompt = query + 'Question:' + node.question
        
        trajectory = []
        new_segment = []
        while node:
            new_segment = []
            if node.state['action']:
                new_segment.append(f"Action: {node.state['action']}")
            if node.state['observation'] and node.depth != 0:  # Exclude the observation from the root node
                new_segment.append(f"Observation: {node.state['observation']}")
            trajectory.append('\n'.join(new_segment))
            node = node.parent
        input_prompt = input_prompt + '\n'.join(reversed(trajectory))
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
        
        args = {
            "n_generate_sample": 10,
            "depth_limit": 6,
            "iterations": 50
        }
        
        #dict to object
        
        args = argparse.Namespace(**args)
        all_terminals = self.dfs_search(args, question, args.iterations, args.depth_limit)
        
        
        if len(all_terminals) > 0:
            
            sorted_terminals = sorted(all_terminals, key=lambda x: x.reward, reverse=True)
            
            best_node = sorted_terminals[0]
            
            answer_prompt = self.make_prompt(best_node) + "\n output="
            success, answer = self.llm_model.generate(self.system_message, answer_prompt)
            
            if success:
                result_lists = self.parse_integer_lists(answer)
                if len(result_lists) > 0:
                    return True, result_lists[-1]
        
        return False, None

    def expand_node(self, node, args, depth_limit):
        if node.depth >= depth_limit:
            logging.info("Depth limit reached")
            print("Depth limit reached")
            node.is_terminal = True
            return
        new_nodes = self.generate_new_states(node, args)
        node.children.extend(new_nodes)
        
        
    def get_samples(self, n_generate_sample, prompt, stop=None):
        if stop is not None:
            config = {"n": n_generate_sample, "stop": stop, "max_tokens": 20}
        else:
            config = {"n": n_generate_sample, "max_tokens": 20}
            
        success, samples = self.llm_model.generate_with_config(self.system_message, prompt, config)
        return success, samples
    
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

    
    def world_model_step(self, node, action):
        prompt = self.make_prompt(node) + '\n Action:' + action + f'\n Predict the outcome observation of this action as well as generate a reward for this new state ({self.prompts["evaluation_prompt"]}). Observation:\n Reward:\n Is done: True/False\n'
        config = {"stop": '', "max_tokens": 100}
        success, output = self.llm_model.generate_with_config(self.system_message, prompt, config)
        output = output.split("\n")
        reward = -50
        observation = ""
        done = False    
        for line in output:
            if "Observation" in line:
                observation = ":".join(line.split(":")[1:]).strip()
            if "Reward" in line:
                numbers = self.find_numbers(line)
                if len(numbers)>0:
                    reward = numbers[0]
            elif "reward is" in line:
                numbers = self.find_numbers(line.split("reward is")[1])
                if len(numbers)>0:
                    reward = numbers[0]
            else:
                continue
            if "done" in line.lower():
                if "true" in line.lower():
                    done = True
        if self.task == "dp":
            done = done or ("None" not in observation)
        return success, observation, reward, done

    def generate_new_states(self, node, args):
        prompt = self.make_prompt(node) + '\n Action:'
        success, sampled_actions = self.get_samples(args.n_generate_sample, prompt, stop="Observation")
        logging.info(f"SAMPLED ACTION: {sampled_actions}")
        
        unique_states = {}  # Store unique states here
        
        sampled_actions = list(set(sampled_actions))
        for action in sampled_actions:
            new_state = node.state.copy()  # Make a copy of the parent node's state

            # reformat action here
            action_line = action.strip()

            # Use thought and action to form a unique key
            unique_key = f"{action_line}"
            
            if unique_key in unique_states:
                continue  # Skip if this state already exists
            
            if action_line:
                action_type = action_line.split('[')[0] if '[' in action_line else action_line
                action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""
                # obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")
                success, obs, r, done = self.world_model_step(node, action_line)

                # Update the new state dictionary
                new_state['action'] = action_line
                new_state['observation'] = obs

                new_node = Node(state=new_state, question=node.question, parent=node)
                new_node.is_terminal = r == 1 or done
                new_node.reward = r
                
                unique_states[unique_key] = new_node  # Add this state to unique_states
                logging.info(f"NEW NODE: {new_node}")
        return list(unique_states.values())  # Return unique nodes as a list
    
    def collect_all_nodes(self, node):
        """Recursively collect all nodes starting from the given node."""
        nodes = [node]
        for child in node.children:
            nodes.extend(self.collect_all_nodes(child))
        return nodes


    def dfs_search(self, args, question, iterations, depth_limit):
        root = Node(state=None, question=question)
        all_nodes = []
        failed_trajectories = []
        stack = [root] 
        it = 0
        
        all_terminals = []
        
        visited = 0
        
        exhausted = []
        
        while stack and it < iterations:
            node = stack.pop()
            logging.info(f"DFS at node depth {node.depth}...")
            
            if node.is_terminal and node.reward == 1:
                logging.info(f"Terminal node with reward {node.reward} found at depth {node.depth}")
                all_terminals.append(node)
                continue
                
            if node.depth >= depth_limit:
                logging.info("Depth limit reached")
                it += 1
                all_terminals.append(node)
                logging.info(f"Terminal node with reward {node.reward} found at depth {node.depth}")
                continue  # go to next iteration
            self.expand_node(node, args, depth_limit)
            stack.extend(reversed(node.children))  # adding all child nodes to stack for DFS

            all_nodes = [(node, node.reward) for node in self.collect_all_nodes(root)]
            logging.info(f"State of all_nodes after iteration: {all_nodes}")
            it += 1
        # If we reach here, no solution was found
        # if len(all_terminals) > 0: all_terminals += exhausted
        return all_terminals
    
    
    @classmethod
    def from_config(cls, llm_model, config):
        return cls(llm_model, config["prompt_path"])
    
class Node:
    def __init__(self, state, question, parent=None):
        self.state = {'action': '', 'observation': ''} if state is None else state
        self.parent = parent
        self.question = question
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.exhausted = False # If all children are terminal

    def uct(self):
        if self.visits == 0:
            #return float('inf')
            return self.value * 2
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def uct_with_depth(self, C1=1, C2=1):
        if self.visits == 0:
            return self.value
        exploitation_term = self.value / self.visits
        exploration_term = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        depth_term = self.depth
        return exploitation_term + C1 * exploration_term + C2 * depth_term

    def __str__(self):
        return f"Node(depth={self.depth}, reward={self.reward:.2f}, is_terminal={self.is_terminal}, action={self.state['action']}, observation={self.state['observation']})"
    
    def to_dict(self):
        return {
            'state': self.state,
            'question': self.question,
            'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children],
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'reward': self.reward,
        }
 