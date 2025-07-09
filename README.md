# Decoding Strategy for Agents


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

