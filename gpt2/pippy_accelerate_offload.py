# Attempting to use `pippy` with `bert` and `accelerate`'s `infer_device_map`
import time
import torch
from accelerate import PartialState
from accelerate.inference import prepare_pippy
from accelerate.utils import set_seed
from transformers import GPT2ForSequenceClassification, GPT2Config

set_seed(42)

config = GPT2Config()
model = GPT2ForSequenceClassification(config)
model.eval()

input = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(2, 1024),  # bs x seq_len
    device="cpu",
    dtype=torch.int64,
    requires_grad=False,
)

example_inputs = {"input_ids": input}

start_time = time.time()
model = prepare_pippy(model, example_args=(input,))
end_time = time.time()
tracing = end_time - start_time

args = (input.to("cuda:0"))
state = PartialState()
with torch.no_grad():
    output = model(args)

torch.cuda.synchronize()
start_time = time.time()
for i in range(5):
    with torch.no_grad():
        output = model(args)
torch.cuda.synchronize()
end_time = time.time()
if output is not None:
    print(f'Time to trace model on GPU: {tracing}')
    print(f'Average time per batch: {(end_time - start_time)/5}')
