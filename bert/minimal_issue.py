# Attempting to use `pippy` with `bert` and `accelerate`'s `infer_device_map`
import torch
from accelerate import PartialState
from transformers import BertForMaskedLM, BertConfig

from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineStage import PipelineStage

# Generate a distributed environment, is the same as if you did `torch.distributed.init_process_group`
state = PartialState()

def add_split_points(bert, nranks):
    layers_per_rank = bert.config.num_hidden_layers // nranks
    for i in range(1, nranks):
        annotate_split_points(
            bert,
            {
                f"bert.encoder.layer.{i * layers_per_rank}": PipeSplitWrapper.SplitPoint.BEGINNING
            },
        )

# Create a blank model
config = BertConfig()
# Ensure that you've saved one version beforehand
# torch.save(BertForMaskedLM(config).state_dict(), "model.pt")
with torch.device("meta"): 
    model = BertForMaskedLM(config)
model.eval()

input = torch.randint(
    low=0, high=config.vocab_size, size=(2, 512), device="meta", dtype=torch.int64, requires_grad=False,
)

example_inputs = {"input_ids": input}
input_ids = example_inputs["input_ids"]

add_split_points(model, state.num_processes)

bert_pipe = Pipe.from_tracing(model, num_chunks=state.num_processes, example_args=(input_ids,))
stage = PipelineStage(bert_pipe, state.local_process_index, device=state.device)
state_dict = torch.load("model.pt")

old_named_params = zip(*list(model.named_parameters()))
old_names = list(old_named_params)[0]

for new_name, _ in bert_pipe.named_parameters():
    old_name = bert_pipe.remap_qualname(new_name)
    print(new_name)
    assert old_name in old_names

# for _, stage_mod in bert_pipe.split_gm.named_children():
#     for new_name, _ in stage_mod.named_parameters():
#         old_name = stage_mod.remap_qualname(new_name)
#         print
# stage.load_state_dict(state_dict, strict=True, assign=True)
# input = torch.randint(low=0, high=config.vocab_size, size=(2, 512), device="cuda", dtype=torch.int64, requires_grad=False)

# if state.is_local_main_process:
#     args = input
# else:
#     args = None

# with torch.no_grad():
#     output = stage(args)
