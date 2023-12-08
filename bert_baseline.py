# Calculates time it takes for device_map to run sequentially through the batch and model
# To compare with `PiPPy`, it will run on two batches individually

from transformers import T5ForConditionalGeneration, T5Config
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from hf_utils import generate_inputs_for_model

config = T5Config()

model = T5ForConditionalGeneration(config)
state_dict = model.state_dict()

device_map = infer_auto_device_map(model, max_memory={0: "0.2GB", 1: "0.2GB"}, no_split_module_classes=["T5Block"])

model = dispatch_model(model, device_map)

model.eval()

example_inputs = [
        generate_inputs_for_model(
        T5ForConditionalGeneration, model, "T5ForConditionalGeneration", 1, "cuda:0"
    ),
    generate_inputs_for_model(
        T5ForConditionalGeneration, model, "T5ForConditionalGeneration", 1, "cuda:0"
    ),
    
]


import time
import torch

start_time = time.time()
for input in example_inputs:
    with torch.no_grad():
        output = model(**input)
end_time = time.time()
print(f"Time taken for sequential run: {end_time - start_time}")