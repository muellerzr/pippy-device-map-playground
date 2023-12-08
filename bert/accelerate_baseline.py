# Calculates time it takes for device_map to run sequentially through the batch and model
# To compare with `PiPPy`, it will run on two batches individually

from transformers import BertForMaskedLM, BertConfig
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import calculate_maximum_sizes, convert_bytes
import math
import time
import torch

config = BertConfig()

model = BertForMaskedLM(config)

model_size, shared = calculate_maximum_sizes(model)
# Returns 242026496, (65798144, ['shared'])

# Split in half for two devices
memory = (model_size + shared[0]) / 2
memory = convert_bytes(memory)
# Returns 115.41 MB
value, ending = memory.split(" ")

# Add a chunk to deal with err:
# cannot access free variable 'chunk_args_list' where it is not associated with a value in enclosing scope
memory = math.ceil(float(value)) * 1.1
memory = f"{memory} {ending}"
device_map = infer_auto_device_map(
    model,
    max_memory={0: memory, 1: memory},
    no_split_module_classes=model._no_split_modules,
)

model = dispatch_model(model, device_map)

model.eval()

example_inputs = []

# Create example inputs for the model
input = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(1, 512),  # bs x seq_len
    device="cuda",
    dtype=torch.int64,
    requires_grad=False,
)

example_inputs = {"input_ids": input}

example_inputs = [example_inputs, example_inputs]

times = []
for _ in range(5):
    start_time = time.time()
    for input in example_inputs:
        with torch.no_grad():
            output = model(**input)
    end_time = time.time()
    times.append(end_time - start_time)
print(f"Time of first pass: {times[0]}")
print(f"Time taken for sequential run: {sum(times[1:]) / len(times[1:])}")