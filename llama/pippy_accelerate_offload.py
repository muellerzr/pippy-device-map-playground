# Attempting to use `pippy` with `bert` and `accelerate`'s `infer_device_map`
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState
from accelerate.inference import prepare_pippy
import inspect
state = PartialState()

# Create a blank model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", low_cpu_mem_usage=True)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
prompts = ("I would like to", 'I really like to', 'The weather is') # bs = 3
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompts, return_tensors="pt", padding=True)

model = prepare_pippy(model, example_args=(inputs['input_ids'],))

if state.is_main_process:
    def get_number_of_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    for i, sm in enumerate(model.pippy_stage.pipe.split_gm.children()):
        print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

inputs = inputs.to(0)

# doesn't work yet 
# output = model.generate(**inputs, max_new_tokens=1)

with torch.no_grad():
    output = model(inputs['input_ids'])

if output is not None:
    next_token_logits = output[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    print(tokenizer.batch_decode(next_token))