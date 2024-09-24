# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importingx transformers.
import sys
sys.path.append("/workspace/Layout_Auto_Generation_MLLM")  
from setproctitle import setproctitle

from miridih_llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from miridih_llava.eval.eval_llava import eval
if __name__ == "__main__":
    setproctitle('MIRIDIH-JJY')
    eval()
