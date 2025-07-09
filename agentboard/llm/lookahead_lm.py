import sys
import os
import torch
import warnings
import math
import psutil

sys.path.append('.')
from common.registry import registry
from prompts.prompt_template import prompt_templates
from tqdm import tqdm
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaForCausalLM
from fastchat.model.model_adapter import Llama2Adapter, raise_warning_for_incompatible_cpu_offloading_configuration

sys.path.append(os.path.join(os.getcwd(), 'agentboard', 'llm')) 
import lade

@registry.register_llm("lade")
class LookAhead_LM:
    def __init__(self,
                 model='',
                 temperature=0,
                 max_tokens=100,
                 top_p=1.0,
                 context_length=4096,
                 stop='\n',
                 ngpu=4,
                 d_type='float16',
                 use_lade=1,
                 level=3,
                 window=10,
                 guess=10,
                 use_flash=0,
                 max_gpu_memory="24.0GB"
                 ):
        
        self.use_lade = use_lade

        if bool(self.use_lade):
            lade.augment_all()
            lade.config_lade(LEVEL=level, WINDOW_SIZE=window, GUESS_SET_SIZE=guess, DEBUG=1, USE_FLASH=use_flash, DIST_WORKERS=0)
            print("lade activated config: ",  lade.decoding.CONFIG_MAP)
        

        self.engine = model
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.context_length = context_length
        # self.sampling_params = SamplingParams(
        #     temperature=temperature,
        #     top_p=top_p,
        #     stop=stop,
        #     max_tokens=max_tokens
        # )

        if d_type == "float16": 
            d_type = torch.float16
        else:
            raise NotImplementedError
        
        llm, tokenizer = self.load_model(
            model,
            use_flash=use_flash,
            device=f"cuda:{lade.get_device()}",
            num_gpus=ngpu,
            max_gpu_memory=max_gpu_memory,
            dtype=d_type,
            load_8bit=False,
            cpu_offloading=True,
            debug=True,
        )
        
        self.llm = llm
        self.tokenizer = tokenizer
    
    def load_model(
        self,
        model_path: str,
        device: str = "cuda",
        device_map: str= "",
        num_gpus: int = 1,
        max_gpu_memory: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        revision: str = "main",
        debug: bool = False,
        use_flash:bool = False
    ):
        """Load a model from Hugging Face."""
        # get model adapter
        adapter = Llama2Adapter()
        # Handle device mapping
        cpu_offloading = raise_warning_for_incompatible_cpu_offloading_configuration(
            device, load_8bit, cpu_offloading
        )
        if device == "cpu":
            kwargs = {"torch_dtype": torch.float32}
            if CPU_ISA in ["avx512_bf16", "amx"]:
                try:
                    import intel_extension_for_pytorch as ipex

                    kwargs = {"torch_dtype": torch.bfloat16}
                except ImportError:
                    warnings.warn(
                        "Intel Extension for PyTorch is not installed, it can be installed to accelerate cpu inference"
                    )
        elif device.startswith("cuda"):
            kwargs = {"torch_dtype": torch.float16}
            if num_gpus != 1:
                kwargs["device_map"] = "auto"
                if max_gpu_memory is None:
                    kwargs[
                        "device_map"
                    ] = "sequential"  # This is important for not the same VRAM sizes
                    available_gpu_memory = get_gpu_memory(num_gpus)
                    kwargs["max_memory"] = {
                        i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                        for i in range(num_gpus)
                    }
                else:
                    kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}

        if cpu_offloading:
            # raises an error on incompatible platforms
            from transformers import BitsAndBytesConfig

            if "max_memory" in kwargs:
                kwargs["max_memory"]["cpu"] = (
                    str(math.floor(psutil.virtual_memory().available / 2**20)) + "Mib"
                )
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit_fp32_cpu_offload=cpu_offloading
            )
            kwargs["load_in_8bit"] = load_8bit
        elif load_8bit:
            if num_gpus != 1:
                warnings.warn(
                    "8-bit quantization is not supported for multi-gpu inference."
                )
            else:
                model, tokenizer = adapter.load_compress_model(
                    model_path=model_path,
                    device=device,
                    torch_dtype=kwargs["torch_dtype"],
                    revision=revision,
                )
                if debug:
                    print(model)
                return model, tokenizer
        kwargs["revision"] = revision

        if dtype is not None:  # Overwrite dtype if it is provided in the arguments.
            kwargs["torch_dtype"] = dtype
        if use_flash:
            kwargs["use_flash_attention_2"] = use_flash
        if len(device_map) > 0:
            kwargs["device_map"] = device_map
        # Load model
        model, tokenizer = adapter.load_model(model_path, kwargs)
        
        if len(device_map) > 0:
            return model, tokenizer

        if (
            device == "cpu"
            and kwargs["torch_dtype"] is torch.bfloat16
            and CPU_ISA is not None
        ):
            model = ipex.optimize(model, dtype=kwargs["torch_dtype"])

        if (device.startswith("cuda") and num_gpus == 1 and not cpu_offloading) or device in (
            "mps",
            "xpu",
            "npu",
        ):
            model.to(device)

        if device == "xpu":
            model = torch.xpu.optimize(model, dtype=kwargs["torch_dtype"], inplace=True)

        if debug:
            print(model)

        return model, tokenizer

    
    def make_prompt(self, system_message, prompt):
        # only support model using llama architecture
        full_prompt = None
        system_message += "Generate your next step of action after Action. Action must not be empty. e.g. Action: put down cup. \n"
        if "llama" in self.model.lower():
            full_prompt = prompt_templates["llama"].format(system_prompt=system_message, prompt=prompt)
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
        
        else:
            raise NotImplementedError
        
        return full_prompt

    def generate(self, system_message, prompt):
        full_prompt = self.make_prompt(system_message, prompt)
        assert full_prompt is not None
        
        input_ids = self.tokenizer([full_prompt]).input_ids
        
        if self.temperature> 1e-4:
            do_sample = True
        else:
            do_sample = False   
        
        output_ids, past_tokens = self.llm.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample=do_sample,
                        temperature=self.temperature,
                        max_new_tokens=self.max_tokens
                    )
        
        output_ids = output_ids.squeeze()
        
        prompt_length = len(input_ids[0])
        
        output_ids = output_ids[prompt_length:-1] # remove eos
        
        outputs = self.tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )

        if self.stop is not None:
            outputs = outputs.split(self.stop)[0]
        
        if 'vicuna' in self.model.lower():
            # Note: vicuna tends to generate get\_search\_movie with Action Input: {"movie\_name": "Crouching Tiger, Hidden Dragon"} when using tools
            outputs = outputs.replace('\_', '_')

        return True, outputs 

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
        dtype = config.get("dtype", 'float16')
        
        use_lade = config.get("use_lade", 1)
        level = config.get("level", 3)
        window = config.get("window", 10)
        guess = config.get("guess", 10)
        max_gpu_memory = config.get("max_gpu_memory", "24.0GB")
        
        return cls(model=engine,
                   temperature=temperature,
                   max_tokens=max_tokens,
                   top_p=top_p,
                   context_length=context_length,
                   stop=stop,
                   ngpu=ngpu,
                   d_type=dtype,
                   use_lade=use_lade,
                   level=level,
                   window=window,
                   guess=guess,
                   use_flash=0,
                   max_gpu_memory=max_gpu_memory)


