# Attempting to use `pippy` with `bert` and `accelerate`'s `infer_device_map`
import math
import torch
from accelerate import infer_auto_device_map, PartialState
from accelerate.utils import calculate_maximum_sizes, convert_bytes, set_seed
from transformers import BertForMaskedLM, BertConfig

from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineStage import PipelineStage

set_seed(42)

# Generate a distributed environment
state = PartialState()

# Create a blank model
config = BertConfig()
# with init_empty_weights():
# Try loading on CPU
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

# Create split points for the model based on the device map
annotate_split_points(model, {split_point: PipeSplitWrapper.SplitPoint.BEGINNING})

# Trace on the `cpu` device then dispatch the weights of the model how we need to
bert_pipe = Pipe.from_tracing(model, state.num_processes, example_args=(input_ids,))
state_dict = torch.load("model.pt")

old_named_params = set(state_dict.keys())
new_named_params = set(bert_pipe.state_dict().keys())

new_device_map = infer_auto_device_map(
    bert_pipe,
    max_memory={0: memory, 1: memory},
    clean_result=False,
)

new_state_dict = {}
not_found = []
for new_name in new_named_params:
    old_name = bert_pipe.remap_qualname(new_name)
    if ".mod." in old_name:
        old_name = old_name.replace(".mod.", ".")
    if old_name in old_named_params:
        new_state_dict[new_name] = state_dict[old_name]
    # else:
        # Possibly need to move 
        # split_gm.submod_0.moved_L__self___bert_embeddings_token_type_ids | cpu
        # split_gm.submod_0.moved_L__self___bert_embeddings_position_ids | cpu
        # To the GPU since they are stuck on the CPU
new_state_dict["split_gm.submod_0.moved_L__self___bert_embeddings_token_type_ids"] = bert_pipe.state_dict()["split_gm.submod_0.moved_L__self___bert_embeddings_token_type_ids"]
new_state_dict["split_gm.submod_0.moved_L__self___bert_embeddings_position_ids"] = bert_pipe.state_dict()["split_gm.submod_0.moved_L__self___bert_embeddings_position_ids"]

for k,v in new_state_dict.items():
    if ".submod_0" in k:
        new_state_dict[k] = v.to("cuda:0")
    elif ".submod_1" in k:
        new_state_dict[k] = v.to("cuda:1")

bert_pipe.load_state_dict(new_state_dict, strict=False, assign=True)
# issue is at self.L__self___bert_encoder_layer_0_attention_self_dropout(softmax), 
# it's still doing it as a CPU op and not present in `new_state_dict` or `bert_pipe.state_dict()`
   

stage = PipelineStage(
    bert_pipe,
    state.local_process_index,
    device=state.device,
)

input = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(2, 512),  # bs x seq_len
    device="cuda:0",
    dtype=torch.int64,
    requires_grad=False,
)

if state.is_local_main_process:
    args = input
else:
    args = None

with torch.no_grad():
    output = stage(args)
if output is not None:
    print(output)
