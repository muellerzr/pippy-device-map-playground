# Attempting to use `pippy` with `bert` and `accelerate`'s `infer_device_map`
import math
import torch
from accelerate import infer_auto_device_map, PartialState
from accelerate.utils import calculate_maximum_sizes, convert_bytes
from transformers import T5ForConditionalGeneration, T5Config

from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineStage import PipelineStage

# Generate a distributed environment
state = PartialState()

# Create a blank model
config = T5Config()
model = T5ForConditionalGeneration(config)

model_size, shared = calculate_maximum_sizes(model)
# Returns 242026496, (65798144, ['shared'])

# Split in half for two devices
memory = (model_size + shared[0]) / 2
memory = convert_bytes(memory)
# Returns 115.41 MB
value, ending = memory.split(' ')

# Add a chunk to deal with err:
# cannot access free variable 'chunk_args_list' where it is not associated with a value in enclosing scope
memory = math.ceil(float(value)) * 1.1
memory = f'{memory} {ending}'
device_map = infer_auto_device_map(
    model, 
    max_memory={0: memory, 1: memory}, 
    no_split_module_classes=["T5Block"], 
    clean_result=False,
)

# Should be `decoder.block0`
split_point = next(k for k, v in device_map.items() if v == 1)

# Create split points for the model based on the device map
annotate_split_points(model, {split_point: PipeSplitWrapper.SplitPoint.BEGINNING})

# Create example inputs for the model
input = torch.randint(
    low=0,
    high=config.vocab_size,
    size = (2, 1024), # bs x seq_len
    device=state.device,
    dtype=torch.int64,
    requires_grad=False,
)

example_inputs = {"input_ids": input, "decoder_input_ids": input}

# Move model to `device` and set to evaluation
model.to(state.device)
model.eval()

# Create a pipeline stage from the model
t5_pipe = Pipe.from_tracing(
    model, 
    num_chunks=state.num_processes,
    example_args=(),
    example_kwargs=example_inputs,
)

if state.is_main_process:
    def get_number_of_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    for i, sm in enumerate(t5_pipe.split_gm.children()):
        print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

# Verify we created two stages
# Create schedule runtime
stage = PipelineStage(
    t5_pipe,
    state.local_process_index,
    device=state.device,
)

if state.is_local_main_process:
    args = (example_inputs["input_ids"].contiguous(), example_inputs["decoder_input_ids"].contiguous())
else:
    args = ()

# Run
import torch
import time
# Take an average of 5 times
times = []
for _ in range(5):
    start_time = time.time()
    with torch.no_grad():
        output = stage(*args)
    end_time = time.time()
    times.append(end_time - start_time)

# First `n` values in output are the model outputs
if output is not None:
    output = torch.stack(tuple(output[0]))
    print(f'Time of first pass: {times[0]}')
    print(f'Total elapsed time: {sum(times[1:]) / len(times[1:])}')