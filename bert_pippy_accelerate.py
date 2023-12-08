# Attempting to use `pippy` with `bert` and `accelerate`'s `infer_device_map`
from accelerate import infer_auto_device_map, PartialState
from transformers import T5ForConditionalGeneration, T5Config

from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineStage import PipelineStage

from hf_utils import generate_inputs_for_model

# Generate a distributed environment
state = PartialState()

# Create a blank model
config = T5Config()
model = T5ForConditionalGeneration(config)

# Create a device map which roughly splits the model in half on each device
device_map = infer_auto_device_map(model, max_memory={0: "0.2GB", 1: "0.2GB"}, no_split_module_classes=["T5Block"])
# Split points occur at device boundaries, such as `encoder.block.5`
split_point = next(k for k, v in device_map.items() if v == 1)

# Create split points for the model based on the device map
annotate_split_points(model, {split_point: PipeSplitWrapper.SplitPoint.BEGINNING})

# Create example inputs for the model
example_inputs = generate_inputs_for_model(
    T5ForConditionalGeneration, model, "T5ForConditionalGeneration", 2, state.device
)

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
start_time = time.time()
with torch.no_grad():
    output = stage(*args)
end_time = time.time()

# First `n` values in output are the model outputs
if output is not None:
    output = torch.stack(tuple(output[0]))
    print(f'Total elapsed time: {end_time - start_time}')