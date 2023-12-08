# Attempting to use `pippy` with `bert` and `accelerate`'s `infer_device_map`
import math
import time
import torch
from accelerate import infer_auto_device_map, PartialState
from accelerate.utils import get_balanced_memory
from transformers import AutoModelForCausalLM, AutoTokenizer

from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineStage import PipelineStage

# Generate a distributed environment
state = PartialState()

# Create a blank model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", low_cpu_mem_usage=True)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
prompts = ("I would like to",'I really like to') # bs = 2
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(state.device)

# mimic device_map = "auto" 
max_memory = get_balanced_memory(model,dtype=model.dtype)
device_map = infer_auto_device_map(model,
                                   dtype=model.dtype, 
                                   max_memory=max_memory, 
                                   no_split_module_classes=model._no_split_modules, 
                                   clean_result=False)

split_point_list = []
prev_device=None
for layer, device in device_map.items(): 
    if prev_device is None:
        prev_device=device
        continue
    if prev_device!=device:
        split_point_list.append(layer)
        prev_device=device

# Create split points for the model based on the device map
for split_point in split_point_list:
    annotate_split_points(model, {split_point: PipeSplitWrapper.SplitPoint.BEGINNING})

# Input configs
# Create example inputs for the model

# Move model to `device` and set to evaluation
model.to(state.device)
model.eval()


# Create a pipeline stage from the model
bert_pipe = Pipe.from_tracing(
    model,
    num_chunks=state.num_processes,
    example_args=(inputs['input_ids'],),
)

if state.is_main_process:
    def get_number_of_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    for i, sm in enumerate(bert_pipe.split_gm.children()):
        print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

# Verify we created two stages
# Create schedule runtime
stage = PipelineStage(
    bert_pipe,
    state.local_process_index,
    device=state.device,
)

if state.is_local_main_process:
    args = inputs['input_ids']
else:
    args = None
# Run
with torch.no_grad():
    output = stage(args)
    if output is not None:
        next_token_logits = output[0][:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        print(tokenizer.batch_decode(next_token))