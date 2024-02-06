import time
import torch
from accelerate import prepare_pippy
from accelerate.utils import set_seed
from transformers import AutoModelForSeq2SeqLM

# Set the random seed to have reproducable outputs
set_seed(42)

# Create an example model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
model.eval()

# Input configs
# Create example inputs for the model
input = torch.randint(
    low=0,
    high=model.config.vocab_size,
    size=(2, 1024),  # bs x seq_len
    device="cpu",
    dtype=torch.int64,
    requires_grad=False,
)

example_inputs = {"input_ids": input, "decoder_input_ids": input}

# Create a pipeline stage from the model
# Using `auto` is equivalent to letting `device_map="auto"` figure
# out device mapping and will also split the model according to the
# number of total GPUs available if it fits on one GPU
model = prepare_pippy(
    model, 
    no_split_module_classes=["T5Block"],
    example_kwargs=example_inputs,
)

# The model expects a tuple during real inference
# with the data on the first device
args = (
    example_inputs["input_ids"].to("cuda:0"),
    example_inputs["decoder_input_ids"].to("cuda:0")
)

# Warm up (NCCL init, etc)
with torch.no_grad():
    output = model(*args)

# Measure first batch
torch.cuda.synchronize()
start_time = time.time()
with torch.no_grad():
    output = model(*args)
torch.cuda.synchronize()
end_time = time.time()
first_batch = end_time - start_time

# Now that CUDA is init, measure after
torch.cuda.synchronize()
start_time = time.time()
for i in range(5):
    with torch.no_grad():
        output = model(*args)
torch.cuda.synchronize()
end_time = time.time()

# First `n` values in output are the model outputs
# which will be located on the last device
if output is not None:
    output = torch.stack(tuple(output[0]))
    print(f"Time of first pass: {first_batch}")
    print(f"Average time per batch: {(end_time - start_time)/5}")
