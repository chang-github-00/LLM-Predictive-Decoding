# from .openai_gpt import OPENAI_GPT
# from .azure_gpt import OPENAI_GPT_AZURE
# # from .claude import CLAUDE
# from .vllm import VLLM
from common.registry import registry
# from .huggingface import HgModels
# # from .lookahead_lm import LookAhead_LM

# __all__ = [
#     "OPENAI_GPT",
#     "OPENAI_GPT_AZURE",
#     "VLLM",
#     "CLAUDE",
#     "HgModels"
# ]





def load_llm(name, config):
    if name == "hg": from .huggingface import HgModels
    if name == "gpt": from .openai_gpt import OPENAI_GPT
    if name == "gpt_azure": from .azure_gpt import OPENAI_GPT_AZURE
    if name == "vllm": from .vllm import VLLM
    if name == "claude": from .claude import CLAUDE
    if name == "lade": from .lookahead_lm import LookAhead_LM
    if name == "openai_vllm": from .openai_vllm import OPENAI_VLLM

        
    llm = registry.get_llm_class(name).from_config(config)
    
    return llm
    
