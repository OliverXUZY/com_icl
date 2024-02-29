import torch
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer

model = "meta-llama/Llama-2-7b-chat-hf"

model = LlamaForCausalLM.from_pretrained(model)

print(model)

