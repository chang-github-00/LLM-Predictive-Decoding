from vllm import LLM, SamplingParams
import pdb
import sys
sys.path.append('.')
from common.registry import registry
from prompts.prompt_template import prompt_templates



@registry.register_llm("vllm")
class VLLM:
    def __init__(self,
                 model='',
                 temperature=0,
                 max_tokens=100,
                 top_p=1.0,
                 context_length=4096,
                 stop='\n',
                 ngpu=4,
                 d_type='bfloat16'
                 ):
        self.engine = model
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.context_length = context_length
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            max_tokens=max_tokens
        )
        if self.context_length > 8192:
            self.llm = LLM(model=self.model, dtype=d_type, tensor_parallel_size=ngpu, gpu_memory_utilization=0.9, max_num_batched_tokens=8192, max_model_len=8192)
        else:
            self.llm = LLM(model=self.model, dtype=d_type, tensor_parallel_size=ngpu, gpu_memory_utilization=0.9, max_num_batched_tokens=self.context_length)
        self.tokenizer = self.llm.get_tokenizer()
        
    def make_prompt(self, system_message, prompt):
        full_prompt = None
        system_message += "Generate your next step of action after Action. Action must not be empty. e.g. Action: put down cup. \n"
        if "codellama-13b" in self.model.lower():
            full_prompt = prompt_templates["codellama-13b"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
        elif "codellama-34b" in self.model.lower():
            full_prompt = prompt_templates["codellama-34b"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
            
        elif "llama-2" in self.model.lower():
            full_prompt = prompt_templates["llama2"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
        elif "llama-3" in self.model.lower():
            full_prompt = prompt_templates["llama3"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
            
        elif 'lemur' in self.model.lower():
            full_prompt = prompt_templates["lemur"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
            
        elif 'vicuna' in self.model.lower():
            full_prompt = prompt_templates["vicuna"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
            
        elif 'deepseek' in self.model.lower():
            full_prompt = prompt_templates["deepseek"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
        
        elif 'mistral' in self.model.lower():
            full_prompt = prompt_templates["mistral"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
            
        else:
            raise NotImplementedError
        
        return full_prompt
    
    def make_prompt_new(self, system_message, prompt, answer_prefix=None):
        if 'llama-3' in self.model.lower():
            full_prompt=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
            ]
            
            if answer_prefix is not None:
                full_prompt.append({"role": "assistant", "content": ""})
            # full_prompt = self.make_prompt(system_message, prompt)
            full_prompt = self.tokenizer.apply_chat_template(full_prompt, tokenize=False)
            if answer_prefix is not None:
                
                full_prompt = full_prompt.rstrip("<|eot_id|>")
                full_prompt += answer_prefix
            return full_prompt
        elif 'mistral' in self.model.lower():
            prompt_template = '''<s>[INST] {system_message}\n\n{prompt}[/INST] '''
            full_prompt = prompt_template.format(system_message=system_message, prompt=prompt)
            if answer_prefix is not None:
                full_prompt += answer_prefix
            return full_prompt
        elif 'codellama' in self.model.lower():
            prompt_template = '''<s>[INST] {system_message}\n\n{prompt}[/INST] \n[BEGIN]\n'''
            full_prompt = prompt_template.format(system_message=system_message, prompt=prompt)
            if answer_prefix is not None:
                full_prompt += answer_prefix
            return full_prompt
        elif 'deepseek-coder' in self.model.lower():
            prompt_template = '''{system_message}\n\n{prompt} ### Response: '''
            full_prompt = prompt_template.format(system_message=system_message, prompt=prompt)
            if answer_prefix is not None:
                full_prompt += answer_prefix
            return full_prompt
        else:
            raise NotImplementedError
            
    def generate(self, system_message, prompt, answer_prefix=None):
        full_prompt=[
                # {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
        ]
        
        if answer_prefix is not None:
            full_prompt.append({"role": "assistant", "content": ""})
        # full_prompt = self.make_prompt(system_message, prompt)
        full_prompt = self.tokenizer.apply_chat_template(full_prompt, tokenize=False)
        if answer_prefix is not None:
            full_prompt = full_prompt.rstrip("<|eot_id|>")
            full_prompt += answer_prefix
        assert full_prompt is not None
        
        outputs = self.llm.generate([full_prompt], self.sampling_params)
        outputs = outputs[0].outputs[0].text
        
        if 'vicuna' in self.model.lower():
            # Note: vicuna tends to generate get\_search\_movie with Action Input: {"movie\_name": "Crouching Tiger, Hidden Dragon"} when using tools
            outputs = outputs.replace('\_', '_')

        return True, outputs 
    
    
    def generate_with_config(self, system_message, prompt, config, answer_prefix=None):
        
        
        stop = config.get("stop", self.stop)
        temperature = config.get("temperature", self.temperature)
        max_tokens = config.get("max_tokens", self.max_tokens)
        n = config.get("n", 1)
        logprobs = config.get("logprobs", 5)
        do_sample = config.get("do_sample", True)
        top_p = config.get("top_p", 1)
        
        if "codellama" in self.model.lower():
            stop = "[END]"
            
        samplingparams = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            max_tokens=max_tokens,
            logprobs=logprobs,
            n=n,
        )
        
        full_prompt= self.make_prompt_new(system_message, prompt, answer_prefix)
            
        response = self.llm.generate([full_prompt], samplingparams)
        
        if response[0] is None:
            return False, None
        
        if logprobs == 0:
            if n == 1:
                return True, response[0].outputs[0].text
            else:
                return True, [choice.text for choice in response[0].outputs]
            
        else:
            outputs = []
        
            for i in range(n):
                choice = response[0].outputs[i]
                item = {}
                item["text"] = choice.text
                raw_log_prob = choice.logprobs
                item["logprobs"] = []
                item["tokens"] = []
                for token in raw_log_prob:
                    for key in token:
                        item["logprobs"].append(token[key].logprob)
                        item["tokens"].append(token[key].decoded_token)
                        break
                outputs.append(item)

            return True, outputs 
    
    def parallel_generate_with_config(self, system_messages, prompts, config, answer_prefixes=None):
        
        
        stop = config.get("stop", self.stop)
        temperature = config.get("temperature", self.temperature)
        max_tokens = config.get("max_tokens", self.max_tokens)
        n = config.get("n", 1)
        logprobs = config.get("logprobs", 5)
        do_sample = config.get("do_sample", True)
        top_p = config.get("top_p", 1)
        
        if "codellama" in self.model.lower():
            stop = ["[END]", "[BEGIN]", "[/"]
        
        use_beam_search = config.get("use_beam_search", False)
        # use_beam_search = True 
        
        if use_beam_search:
            if n > 1:
                best_of = n
            else:
                best_of = 8
            samplingparams = SamplingParams(
                temperature=0,
                top_p=top_p,
                stop=stop,
                max_tokens=max_tokens,
                logprobs=logprobs,
                n=n,
                best_of=best_of,
                use_beam_search=True,
            )
        else:
            samplingparams = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                max_tokens=max_tokens,
                logprobs=logprobs,
                n=n,
            )
            
        full_prompts = []
        for i in range(len(prompts)):
            prompt = prompts[i]
            system_message = system_messages[i]
            if answer_prefixes is not None:
                answer_prefix = answer_prefixes[i]
            else:
                answer_prefix = None
            
            full_prompt= self.make_prompt_new(system_message, prompt, answer_prefix)
            
            full_prompts.append(full_prompt)
            
        response = self.llm.generate(full_prompts, samplingparams)
        
        if logprobs == 0:
            if n == 1:
                return True, [res.outputs[0].text for res in response]
            else:
                return True, [[choice.text for choice in res.outputs] for res in response]
            
        else:
            all_outputs = []
            for res in response:
                if res is None:
                    all_outputs.append(None)
                    continue
                if len(res.outputs) < n:
                    all_outputs.append(None)
                    continue
                outputs = []
            
                for i in range(n):
                    choice = res.outputs[i]
                    item = {}
                    item["text"] = choice.text
                    raw_log_prob = choice.logprobs
                    item["logprobs"] = []
                    item["tokens"] = []
                    for token in raw_log_prob:
                        for key in token:
                            item["logprobs"].append(token[key].logprob)
                            item["tokens"].append(token[key].decoded_token)
                            break
                    outputs.append(item)
                all_outputs.append(outputs)
            return True, all_outputs
    
    def encode(self, system_messages, prompts, answer_prefixes=None):
        full_prompts = []
        for i in range(len(prompts)):
            prompt = prompts[i]
            system_message = system_messages[i]
            if answer_prefixes is not None:
                answer_prefix = answer_prefixes[i]
            else:
                answer_prefix = None
            
            full_prompt=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
            ]
            
            samplingparams = SamplingParams(
                temperature=0,
                # max_tokens=1,
                prompt_logprobs=1,
            )
            
            if answer_prefix is not None:
                full_prompt.append({"role": "assistant", "content": ""})
            # full_prompt = self.make_prompt(system_message, prompt)
            full_prompt = self.tokenizer.apply_chat_template(full_prompt, tokenize=False)
            if answer_prefix is not None:
                full_prompt = full_prompt.rstrip("<|eot_id|>")
                full_prompt += answer_prefix
            assert full_prompt is not None
            
            full_prompts.append(full_prompt)
        
        outputs = self.llm.generate(full_prompts, samplingparams)
        
        # get tokens and logprobs
        all_logprobs = []
        
        for i in range(len(outputs)):
            tokens = [self.tokenizer.decode(prompt_token_id) for prompt_token_id in outputs[i].prompt_token_ids][1:]
            logprobs = []
            token_id = 1
            for logprob in outputs[i].prompt_logprobs:
                if logprob is not None:
                    token = outputs[i].prompt_token_ids[token_id]
                    logprobs.append(logprob[token].logprob)
                    token_id += 1
            item = {'tokens': tokens, 'logprobs': logprobs}
            all_logprobs.append(item)
        return True, all_logprobs
        
    def get_input(self, system_message, prompt, answer_prefix=None):
        full_prompt=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
        ]
        
        if answer_prefix is not None:
            full_prompt.append({"role": "assistant", "content": answer_prefix})
        # full_prompt = self.make_prompt(system_message, prompt)
        full_prompt = self.tokenizer.apply_chat_template(full_prompt, tokenize=False)
        full_prompt = full_prompt.rstrip("<|eot_id|>")
        assert full_prompt is not None
        
        return full_prompt

    def num_tokens_from_messages(self, messages):
        prompt = messages[1]["content"]
        system_message = messages[0]["content"]
        full_prompt = self.make_prompt(system_message, prompt)
        tokens = self.tokenizer(full_prompt)
        num_tokens = len(tokens["input_ids"])
        #print(num_tokens)
        return num_tokens

    @classmethod
    def from_config(cls, config):

        engine = config.get("engine", "gpt-35-turbo")
        temperature = config.get("temperature", 0)
        max_tokens = config.get("max_tokens", 100)
        top_p = config.get("top_p", 1)
        stop = config.get("stop", ["\n"])
        context_length = config.get("context_length", 4096)
        ngpu = config.get("ngpu", 4)
        dtype = config.get("dtype", 'bfloat16')
        return cls(model=engine,
                   temperature=temperature,
                   max_tokens=max_tokens,
                   top_p=top_p,
                   context_length=context_length,
                   stop=stop,
                   ngpu=ngpu,
                   d_type=dtype)