# Non-Myopic Generation of Language Models for Reasoning and Planning (ICLR 2025)
![Figure: Non-Myopic Generation Overview](/assets/main_fig.jpeg)

[Non-Myopic Generation of Language Models for Reasoning and Planning📝](https://openreview.net/pdf?id=OoNazl6T7D)

## Callback on Motivation



**See our latest follow-up work**:

- [phi-decoding: Adaptive Foresight Sampling for Balanced Inference-Time Exploration and Exploitation](https://arxiv.org/pdf/2503.13288) (ACL 2025)

- [Genius: A Generalizable and Purely Unsupervised Self-Training Framework For Advanced Reasoning](https://arxiv.org/pdf/2504.08672) (ACL 2025)

Also see [phi-decoding](https://github.com/xufangzhi/phi-Decoding/blob/main/baselines/Baseline-PD-aime.py) repo for implementation of predictive decoding on AIME and latest models (Qwen, Deepseek). 




## Quick Start 
## Download dataset
under path `data/`

## Parallel decoding + lookahead
use `lade_results_all_tasks.yaml`, following [lade](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)

## Lookahead + direct selection of hard match n-gram action
use `agent_lade_results_all_tasks.yaml`

```
git clone https://github.com/chang-github-00/vllm
python -m pip install vllm
cd Agent-Decoding
INSTALL_WEBARENA=false bash ./setup.sh
```

