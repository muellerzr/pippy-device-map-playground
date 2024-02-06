import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import prepare_pippy

# sdpa implementation which is the default torch>2.1.2 fails with the tracing + attention mask kwarg
# with attn_implementation="eager" mode, the forward is very slow for some reason
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", 
    low_cpu_mem_usage=True,
    attn_implementation="sdpa"
)
model.eval()

# Input configs
# Create example inputs for the model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
prompts = ("I would like to", 'I really like to', 'The weather is') # bs = 3
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompts, return_tensors="pt", padding=True)

# Create a pipeline stage from the model
# Using `auto` is equivalent to letting `device_map="auto"` figure
# out device mapping and will also split the model according to the
# number of total GPUs available if it fits on one GPU
model = prepare_pippy(model, split_points="auto", example_args=inputs)

# currently we don't support `model.generate` 
# output = model.generate(**inputs, max_new_tokens=1)

with torch.no_grad():
    output = model(**inputs)

# First `n` values in output are the model outputs
# which will be located on the last device
if output is not None:
    next_token_logits = output[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    print(tokenizer.batch_decode(next_token))