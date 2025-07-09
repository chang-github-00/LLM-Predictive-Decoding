

PROJECT_DIR=...
echo "Running from $PROJECT_DIR"


# pal self-consistency run experiment 
# math parallel run pal, set n generate samples to 1

python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs/math/self_consistency_math_llama3.yaml --tasks math --algorithm Self_Consistency --model llama-3 --data_path $PROJECT_DIR/data/math/test.json
python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs/gsm8k/self_consistency_gsm8k_llama3.yaml --tasks gsm8k --algorithm Self_Consistency --model llama-3 --data_path $PROJECT_DIR/data/gsm8k


#predictive decoding run experiment
#batch size means parallel generation of # samples with vllm
python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs/math/mpc_sample_math_llama3.yaml --tasks math --algorithm MPC_Sample --model llama-3 --data_path $PROJECT_DIR/data/math/test.json   --batch_size 500
python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs/gsm8k/mpc_sample_gsm8k_llama3.yaml --tasks gsm8k --algorithm MPC_Sample --model llama-3 --data_path $PROJECT_DIR/data/gsm8k --batch_size 500



# reward model + predictive decoding

# first serve reward model in a separate terminal
vllm serve /root/huggingface/math-shepherd-mistral-7b-prm
# then run the following command, could possibly run more than one in parallel.
python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/gsm8k/mpc_reward_gsm8k_llama3.yaml --tasks gsm8k --algorithm MPC_Sample_Reward --model llama-3 --data_path $PROJECT_DIR/data/gsm8k --batch_size 2000 --reward_model math-shepherd



# agent experiments

# Act. React
python agentboard/eval_main.py --cfg-path eval_configs/alf-world/act_alfworld_gpt35.yaml --tasks alfworld --model gpt-35-turbo  --max_num_steps 20
python agentboard/eval_main.py --cfg-path eval_configs/alf-world/react_alfworld_gpt35.yaml --tasks alfworld --model gpt-35-turbo  --max_num_steps 20

# predictive decoding
python agentboard/eval_main.py --cfg-path eval_configs/alf-world/mpc_sample_alfworld_gpt35.yaml --tasks alfworld --model gpt-35-turbo  --max_num_steps 20