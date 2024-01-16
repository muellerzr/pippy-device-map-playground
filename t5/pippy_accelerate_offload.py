# Attempting to use `pippy` with `bert` and `accelerate`'s `infer_device_map`
import time
import torch
from accelerate.inference import prepare_pippy
from accelerate.utils import set_seed
from transformers import T5ForConditionalGeneration, T5Config

set_seed(42)

config = T5Config()
model = T5ForConditionalGeneration(config)
model.eval()

# Create example inputs for the model
input = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(2, 1024),  # bs x seq_len
    device="cpu",
    dtype=torch.int64,
    requires_grad=False,
)

example_inputs = {"input_ids": input, "decoder_input_ids": input}

start_time = time.time()
model = prepare_pippy(model, example_kwargs=example_inputs)
end_time = time.time()
tracing = end_time - start_time

args = (
    example_inputs["input_ids"].to("cuda:0"),
    example_inputs["decoder_input_ids"].to("cuda:0")
)

torch.cuda.synchronize()
start_time = time.time()
for i in range(5):
    with torch.no_grad():
        output = model(*args)
torch.cuda.synchronize()
end_time = time.time()
if output is not None:
    print(f'Time to trace model on GPU: {tracing}')
    print(f'Average time per batch: {(end_time - start_time)/5}')
