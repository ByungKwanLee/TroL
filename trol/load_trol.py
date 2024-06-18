import torch
import warnings
from config import *
from peft import LoraConfig
from transformers import BitsAndBytesConfig

warnings.filterwarnings(action='ignore')

def setting_trol_config(trol, tok_trol, image_special_token):
    trol.config.image_token_index = tok_trol.convert_tokens_to_ids(image_special_token)
    trol.config.ignore_index = -100
    trol.config.pad_token_id = tok_trol.eos_token_id
    trol.config.eos_token_id = tok_trol.eos_token_id


def load_trol(link):

    """
    model selection
    """
    if link == 'TroL-1.8B':
        from .arch_internlm2.modeling_trol import TroLForCausalLM
        from .arch_internlm2.tokenization_internlm2 import InternLM2Tokenizer as TroLTokenizer
        bits = 4
        path = TROL_1_8B
        image_special_token = "<image>"
        bit_quant_skip = ["vit", "vision_proj", "ffn", "output", "trol_gating"]

    elif link == 'TroL-3.8B':
        from trol.arch_phi3.modeling_trol import TroLForCausalLM 
        from transformers import LlamaTokenizerFast as TroLTokenizer
        bits = 8
        path = TROL_3_8B
        image_special_token = "<IMG_CONTEXT>"
        bit_quant_skip = ["vision_model", "vision_proj", "lm_head", "trol_gating"]

    elif link == 'TroL-7B':
        from .arch_internlm2.modeling_trol import TroLForCausalLM
        from .arch_internlm2.tokenization_internlm2 import InternLM2Tokenizer as TroLTokenizer
        bits = 4
        path = TROL_7B
        image_special_token = "<image>"
        bit_quant_skip = ["vit", "vision_proj", "ffn", "output", "trol_gating"]
    else:
        raise Exception("Unsupported Link")

    # huggingface model configuration
    huggingface_config = {}

    # Bit quantization
    if bits in [4, 8]:
        huggingface_config.update(dict(
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            ignore_mismatched_sizes=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                llm_int8_skip_modules=bit_quant_skip,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        ))
    else:
        huggingface_config.update(dict(
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            ignore_mismatched_sizes=True,
        ))

    # Loading tokenizer & Loading backbone model (error -> then delete flash attention)
    tok_trol = TroLTokenizer.from_pretrained(path, padding_side='left')
    try:
        trol = TroLForCausalLM.from_pretrained(path, **huggingface_config)
    except:
        del huggingface_config["attn_implementation"]
        trol = TroLForCausalLM.from_pretrained(path, **huggingface_config)
        trol.config.llm_config.use_cache = False

    # setting config
    setting_trol_config(trol, tok_trol, image_special_token)
    
    # trol gating load
    from huggingface_hub import hf_hub_download
    try:
        trol.model.trol_gating.load_state_dict(torch.load(hf_hub_download(repo_id=path, filename="trol_gating.pt")))
    except:
        trol.language_model.model.trol_gating.load_state_dict(torch.load(hf_hub_download(repo_id=path, filename="trol_gating.pt")))
    return trol, tok_trol