export CUDA_VISIBLE_DEVICES=2
export OPENAI_API_BASE="http://localhost:8000/v1"

# run rank scaling law experiments
python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/rank/cot_rank_4.yaml --tasks gsm8k --model llama-3 --reward_model math-shepherd --algorithm COT_Reward --batch_size 50 --data_path  /root/huggingface/gsm8k --log_path results/scaling_law_experiment_cot_rank_4_error_bar --reward_model math-shepherd
# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/rank/cot_rank_8.yaml --tasks gsm8k --model llama-3 --reward_model math-shepherd --algorithm COT_Reward --batch_size 50 --data_path  /root/huggingface/gsm8k --log_path results/scaling_law_experiment_cot_rank_8_error_bar --reward_model math-shepherd
# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/rank/cot_rank_16.yaml --tasks gsm8k --model llama-3 --reward_model math-shepherd --algorithm COT_Reward --batch_size 20 --data_path  /root/huggingface/gsm8k --log_path results/scaling_law_experiment_cot_rank_16_error_bar --reward_model math-shepherd



# run vote scaling law experiments
python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/vote/cot_vote_2.yaml --tasks gsm8k --model llama-3 --reward_model math-shepherd --algorithm COT_Reward --batch_size 100 --data_path  /root/huggingface/gsm8k --log_path results/scaling_law_experiment_cot_vote_2_error_bar --reward_model math-shepherd
# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/vote/cot_vote_4.yaml --tasks gsm8k --model llama-3 --reward_model math-shepherd --algorithm COT_Reward --batch_size 50 --data_path  /root/huggingface/gsm8k --log_path results/scaling_law_experiment_cot_vote_4_error_bar --reward_model math-shepherd
# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/vote/cot_vote_8.yaml --tasks gsm8k --model llama-3 --reward_model math-shepherd --algorithm COT_Reward --batch_size 50 --data_path  /root/huggingface/gsm8k --log_path results/scaling_law_experiment_cot_vote_8_error_bar --reward_model math-shepherd
# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/vote/cot_vote_16.yaml --tasks gsm8k --model llama-3 --reward_model math-shepherd --algorithm COT_Reward --batch_size 20 --data_path  /root/huggingface/gsm8k --log_path results/scaling_law_experiment_cot_vote_16_error_bar --reward_model math-shepherd
# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/vote/cot_vote_32.yaml --tasks gsm8k --model llama-3 --reward_model math-shepherd --algorithm COT_Reward --batch_size 20 --data_path  /root/huggingface/gsm8k --log_path results/scaling_law_experiment_cot_vote_32_error_bar --reward_model math-shepherd
# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/vote/cot_vote_64.yaml --tasks gsm8k --model llama-3 --reward_model math-shepherd --algorithm COT_Reward --batch_size 10 --data_path  /root/huggingface/gsm8k --log_path results/scaling_law_experiment_cot_vote_64_error_bar --reward_model math-shepherd

# run mpc scaling law experiments
python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/mpc/mpc_t_2.yaml --tasks gsm8k --algorithm MPC_Sample_Reward --model llama-3 --data_path /root/huggingface/gsm8k --log_path results/scaling_law_experiment_mpc_k_2_t_2_error_bar --batch_size 200 --reward_model math-shepherd
# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/mpc/mpc_t_4.yaml --tasks gsm8k --algorithm MPC_Sample_Reward --model llama-3 --data_path /root/huggingface/gsm8k --log_path results/scaling_law_experiment_mpc_k_2_t_4_error_bar --batch_size 200 --reward_model math-shepherd
# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/mpc/mpc_k_2.yaml --tasks gsm8k --algorithm MPC_Sample_Reward --model llama-3 --data_path /root/huggingface/gsm8k --log_path results/scaling_law_experiment_mpc_k_2_t_6_error_bar --batch_size 200 --reward_model math-shepherd
# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/mpc/mpc_k_4.yaml --tasks gsm8k --algorithm MPC_Sample_Reward --model llama-3 --data_path /root/huggingface/gsm8k --log_path results/scaling_law_experiment_mpc_k_4_t_6_error_bar --batch_size 200 --reward_model math-shepherd
