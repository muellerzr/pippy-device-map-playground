# Attempting to use `pippy` with `bert` and `accelerate`'s `infer_device_map`
import math
import time
import torch
from accelerate import infer_auto_device_map, PartialState, dispatch_model
from accelerate.utils import calculate_maximum_sizes, convert_bytes
from accelerate.hooks import remove_hook_from_submodules
from transformers import BertForMaskedLM, BertConfig

from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineStage import PipelineStage

# Generate a distributed environment
state = PartialState()

# Create a blank model
config = BertConfig()
model = BertForMaskedLM(config)

model_size, shared = calculate_maximum_sizes(model)

# Split in half for two devices
memory = (model_size + shared[0]) / 2
memory = convert_bytes(memory)
value, ending = memory.split(" ")

# Add a chunk to deal with err:
# cannot access free variable 'chunk_args_list' where it is not associated with a value in enclosing scope
memory = math.ceil(float(value)) * 1.1
memory = f"{memory} {ending}"
device_map = infer_auto_device_map(
    model,
    max_memory={0: memory, 1: memory},
    clean_result=False,
    no_split_module_classes=model._no_split_modules,
)

split_point = next(k for k, v in device_map.items() if v == 1)

# Move model to `device` and set to evaluation without using Hooks, causes issue with tracing
model.eval()

# Input configs
# Create example inputs for the model
input = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(2, 512),  # bs x seq_len
    device="cpu",
    dtype=torch.int64,
    requires_grad=False,
)

example_inputs = {"input_ids": input}
input_ids = example_inputs["input_ids"]

# input_ids = input_ids.to(state.device)
model = dispatch_model(model, device_map)
remove_hook_from_submodules(model)

# Create split points for the model based on the device map
annotate_split_points(model, {split_point: PipeSplitWrapper.SplitPoint.BEGINNING})

bert_pipe = Pipe.from_tracing(
    model,
    num_chunks=state.num_processes,
    example_args=(input_ids.cuda(),),
)



if state.is_main_process:

    def get_number_of_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    for i, sm in enumerate(bert_pipe.split_gm.children()):
        print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")



stage = PipelineStage(
    bert_pipe,
    state.local_process_index,
    device=state.device,
)

if state.is_local_main_process:
    args = input_ids
else:
    args = None

# Run

# Take an average of 5 times
# Measure first batch
torch.cuda.synchronize()
start_time = time.time()
with torch.no_grad():
    output = stage(args)
torch.cuda.synchronize()
end_time = time.time()
first_batch = end_time - start_time

# Now that CUDA is init, measure after
torch.cuda.synchronize()
start_time = time.time()
for i in range(5):
    with torch.no_grad():
        output = stage(args)
torch.cuda.synchronize()
end_time = time.time()

# First `n` values in output are the model outputs
if output is not None:
    output = torch.stack(tuple(output[0]))
    print(f"Time of first pass: {first_batch}")
    print(f"Average time per batch: {(end_time - start_time)/5}")
