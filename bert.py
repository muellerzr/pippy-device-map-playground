# Attempting to use `pippy` with `bert` and `accelerate`'s `infer_device_map`
import time
import torch
from accelerate import PartialState, prepare_pippy
from transformers import BertForMaskedLM, BertConfig

# Generate a distributed environment
state = PartialState()

# Create a blank model
config = BertConfig()
model = BertForMaskedLM(config)

# Input configs
# Create example inputs for the model
input = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(2, 512),  # bs x seq_len
    device=state.device,
    dtype=torch.int64,
    requires_grad=False,
)

# Move model to `device` and set to evaluation
model.to(state.device)
model.eval()

# Create a pipeline stage from the model
model = prepare_pippy(model, split_points="auto", example_args=(input,))

# Take an average of 5 times
# Measure first batch
torch.cuda.synchronize()
start_time = time.time()
with torch.no_grad():
    output = model(input)
torch.cuda.synchronize()
end_time = time.time()
first_batch = end_time - start_time

# Now that CUDA is init, measure after
torch.cuda.synchronize()
start_time = time.time()
for i in range(5):
    with torch.no_grad():
        output = model(input)
torch.cuda.synchronize()
end_time = time.time()

# First `n` values in output are the model outputs
if output is not None:
    output = torch.stack(tuple(output[0]))
    print(f"Time of first pass: {first_batch}")
    print(f"Average time per batch: {(end_time - start_time)/5}")
