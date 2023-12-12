import torch
from accelerate.utils import set_seed
from accelerate.inference import InferenceHandler
from transformers import T5ForConditionalGeneration, T5Config

set_seed(42)
config = T5Config()
model = T5ForConditionalGeneration(config)
model.eval()

handler = InferenceHandler(parallel_mode="pipeline_parallel")
handler.device_map = handler.generate_device_map(model, "pipeline_parallel", 2)
model = handler.prepare(model)

input = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(2, 1024),  # bs x seq_len
    device=handler.device,
    dtype=torch.int64,
    requires_grad=False,
)

input = {"input_ids": input, "decoder_input_ids": input}

with torch.no_grad():
    # output = handler(model, **input)
    from accelerate.inference import Pipe, PipelineStage

    pipeline = Pipe.from_tracing(
        model, num_chunks=2, example_args=(), example_kwargs=input
    )
    scheduler = PipelineStage(pipeline, handler.state.local_process_index, device=handler.device)
    output = scheduler(*input.values())
# output = model._original_forward(**input)

if output is not None:
    print(output[0])