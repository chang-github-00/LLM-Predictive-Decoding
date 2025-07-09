class count_tokens:
    def __init__(self):
        self.prompt_count = 0
        self.generation_count = 0
        self.reward_count = 0
        self.instance_count = 0
        self.prompt_tokens = []
        self.reward_prompt_tokens = []

    def add_prompt_tokens(self, num, num_return_sequences):
        self.prompt_count += num_return_sequences
        self.prompt_tokens.append(num)
    
    def add_generation_tokens(self, num):
        self.generation_count += num
        
    def add_reward_tokens(self, num, num_return_sequences):
        self.reward_count += num_return_sequences
        self.reward_prompt_tokens.append(num)
    
    def add_instance(self, num):
        self.instance_count += num
    
    def print(self):
        # average tokens per instance
        if self.instance_count == 0:
            return
        # print("Prompt times: ", self.prompt_count/self.instance_count)
        # print("Average prompt length: ", sum(self.prompt_tokens)/len(self.prompt_tokens))
        print("Generation tokens per instance: ", self.generation_count/self.instance_count)
        print("Reward prompt times: ", self.reward_count/self.instance_count)
        # print("Average reward prompt length: ", sum(self.reward_prompt_tokens)/len(self.reward_prompt_tokens))
    
    def get_count(self):
        stats = dict()
        if len(self.prompt_tokens) == 0:
            self.prompt_tokens.append(0)
        if len(self.reward_prompt_tokens) == 0:
            self.reward_prompt_tokens.append(0)
        stats['avg_prompt_tokens'] = sum(self.prompt_tokens)/len(self.prompt_tokens)
        stats['avg_prompt_times'] = self.prompt_count/self.instance_count
        stats['avg_generation_tokens'] = self.generation_count/self.instance_count
        stats['avg_reward_tokens'] = sum(self.reward_prompt_tokens)/len(self.reward_prompt_tokens)
        stats['avg_reward_times'] = self.reward_count/self.instance_count
        
        # keep only 1 decimal
        stats['avg_prompt_tokens'] = round(stats['avg_prompt_tokens'], 1)
        stats['avg_prompt_times'] = round(stats['avg_prompt_times'], 1)
        stats['avg_generation_tokens'] = round(stats['avg_generation_tokens'], 1)
        stats['avg_reward_tokens'] = round(stats['avg_reward_tokens'], 1)
        stats['avg_reward_times'] = round(stats['avg_reward_times'], 1)
        return stats

count_flag = True
token_count = count_tokens()