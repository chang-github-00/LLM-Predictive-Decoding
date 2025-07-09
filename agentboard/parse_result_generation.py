import re
import json
import io
import datasets
from utils.math.math_utils import parse_question, parse_ground_truth, math_equal, call_with_timeout



# all_ids = [int(exp["exp_id"]) for exp in result]
# for id in range(1, 2000):
#     if id not in all_ids:
#         print(id)

def parse_answer(file_path):
    pattern = re.compile(
        r"\[EXP\] (?P<id>\d+): \[success_rate\]: (?P<success_rate>\w+), "
        r"\[answer\]: (?P<answer>[^\n]+), \[output\]: (?P<output>.*?)"
        r"Executed result: (?P<executed_result>[^\n]*)",
        re.DOTALL
    )
    data = open(file_path, "r").read()
    matches = pattern.finditer(data)
    
    result = []
    for match in matches:
        exp_data = {
            "exp_id": int(match.group("id")),
            "success_rate": match.group("success_rate") == "True",
            "answer": match.group("answer"),
            "output": match.group("output").strip(),
            "executed_result": match.group("executed_result")
        }
        result.append(exp_data)
    
    if len(result) == 0:
        pattern= re.compile(
            r"\[EXP\] (?P<id>\d+): \[success_rate\]: (?P<success_rate>\w+), "
            r"\[output\]: (?P<output>.*?)"
            r"Executed result: (?P<executed_result>[^\n]*)",
            re.DOTALL
        )
        
        matches = pattern.finditer(data)
        result = []
        for match in matches:
            exp_data = {
                "exp_id": int(match.group("id")),
                "success_rate": match.group("success_rate") == "True",
                "output": match.group("output").strip(),
                "executed_result": match.group("executed_result")
            }
            result.append(exp_data)
            
    return result


def load_dataset(task, path='/root/huggingface/gsm8k'):
    if task == "gsm8k":
        full_dataset = datasets.load_dataset(path, 'main', split='test')
        dataset = [{"question": a["question"], "answer": a["answer"]} for a in full_dataset]
        return dataset
    
    if task == "math":
        examples = []
        with open(path, "r") as f: 
            for line in f:
                js = json.loads(line)
                examples.append(js)
        
        dataset = []
        for example in examples:
            idx = example['idx']
            example['question'] = parse_question(example, "math")
            gt_cot, gt_ans = parse_ground_truth(example, "math")
            example = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'answer': gt_ans}
            dataset.append(example)  

        return dataset



task = "gsm8k"
data_path = "/root/huggingface/gsm8k"

file_path = "/root/Agent-Decoding/results/run_parallel_self_consistency_gsm8k_8_29_beam_search_best_1/gsm8k_gsm8k.txt"
# file_path = "/root/Agent-Decoding/results/run_parallel_mpc_sample_math_8_21_small_temp/math_test.txt"
output_path = "/root/Agent-Decoding/results/run_anlaysis_important_results/gsm8k_beamsearch_output.jsonl"

file_path = "/root/Agent-Decoding/results/run_self_consistency_gsm8k_llama3_8_1_n_1_pal_prompt/gsm8k_gsm8k.txt"
output_path = "/root/Agent-Decoding/results/run_anlaysis_important_results/gsm8k_autoregressive_output.jsonl"

file_path = "/root/Agent-Decoding/results/run_parallel_mpc_sample_gsm8k_8_21_smaller_temp_n_8/gsm8k_gsm8k.txt"
output_path = "/root/Agent-Decoding/results/run_anlaysis_important_results/gsm8k_mpc_output.jsonl"


task = "math"
data_path = "data/math/test.json"

file_path = "/root/Agent-Decoding/results/run_parallel_mpc_sample_math_8_21_small_temp_continue/math_test.txt"
output_path = "/root/Agent-Decoding/results/run_anlaysis_important_results/math_mpc_output.jsonl"

file_path = "/root/Agent-Decoding/results/run_parallel_self_consistency_math_8_29_beam_search_best_1/math_test.txt"
output_path = "/root/Agent-Decoding/results/run_anlaysis_important_results/math_beamsearch_output.jsonl"

file_path = "/root/Agent-Decoding/results/run_parallel_self_consistency_math_8_21/math_test.txt"
output_path = "/root/Agent-Decoding/results/run_anlaysis_important_results/math_autoregressive_output.jsonl"


dataset = load_dataset(task, path=data_path)
results = parse_answer(file_path)

output_f = open(output_path, "w")

for item in results:
    idx = item["exp_id"]
    question = dataset[idx-1]['question']
    answer = dataset[idx-1]['answer']
    output = item['output']
    executed_result = item['executed_result']
    success_rate = item['success_rate']
    
    # first clean output, so there are no more than one \n in a row
    output = re.sub(r"\n+", "\n", output)
    output = output.strip()
    
    if task == "gsm8k":
        # GSM8k
        from prompts.Reasoning.gsm8k_prompt import code_prompt, evaluate_prompt, pal_prompt, pal_prompt_1
        prompt = pal_prompt + f"\n\n\n\n\nSolve this problem following previous examples:\nQ: {question}\n# solution in Python:\n\n"
        system_msg = "You will write python program to solve math problems. You will only write code blocks."
        answer_prefix = "def solution():\n"
    elif task == "math":
        from prompts.Reasoning.math_prompt import math_deepseekpal_prompt 
        prompt = math_deepseekpal_prompt + f"\n\n\n\n\nSolve this problem following previous examples:\n\nQ: {question}\n"
        system_msg = "You will write python program to solve math problems. You will only write imports and code blocks ."
        answer_prefix = "solution in Python:\n```\n"
    
    # prompt = prompt.format(question=question)
    if task == "gsm8k":
        if answer_prefix in output:
            output = output.split(answer_prefix)[1]
        with io.StringIO() as f:
            f.write(answer_prefix)
            f.write(output)
            output = f.getvalue()
    elif task == "math":
        with io.StringIO() as f:
            f.write(answer_prefix)
            f.write(output)
            f.write("\n```\n")
            output = f.getvalue()
            
    item["output"] = output
    item["prompt"] = prompt
    item["answer_prefix"] = answer_prefix
    item["system_msg"] = system_msg
    
    if "answer" not in item:
        item["answer"] = answer
    output_f.write(json.dumps(item) + "\n")