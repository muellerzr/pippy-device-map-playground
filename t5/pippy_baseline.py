# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 2 pippy_t5.py


import argparse
import os
import time

import torch
import torch.distributed as dist

from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineStage import PipelineStage

from transformers import T5ForConditionalGeneration, T5Config


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_split_points(t5, nranks):
    # Number of encoder layers: t5.config.num_layers
    # Number of decoder layers: t5.config.num_decoder_layers
    # 6 encoder layers, 6 decoder layers, 12 layers in total
    total_layers = t5.config.num_layers + t5.config.num_decoder_layers
    layers_per_rank = (total_layers + nranks - 1) // nranks
    print(f"Layers per rank = {layers_per_rank}")
    nstages = 1
    # Split encoder
    for i in range(1, t5.config.num_layers // layers_per_rank):
        annotate_split_points(
            t5,
            {
                f"encoder.block.{i * layers_per_rank}": PipeSplitWrapper.SplitPoint.BEGINNING
            },
        )
        nstages += 1
    # Split at the boundary of encoder and decoder
    annotate_split_points(
        t5, {"decoder.embed_tokens": PipeSplitWrapper.SplitPoint.BEGINNING}
    )
    nstages += 1
    # Split decoder
    for i in range(1, t5.config.num_decoder_layers // layers_per_rank):
        annotate_split_points(
            t5,
            {
                f"decoder.block.{i * layers_per_rank}": PipeSplitWrapper.SplitPoint.BEGINNING
            },
        )
        nstages += 1
    assert nstages == nranks, f"nstages = {nstages} nranks = {nranks}"


def run(args):
    # Model configs
    config = T5Config()
    print("Using device:", args.device)

    # Create model
    model_class = T5ForConditionalGeneration
    t5 = model_class(config)
    t5.to(args.device)
    t5.eval()
    if args.rank == 0:
        print(t5.config)
        print(f"Total number of params = {get_number_of_params(t5) // 10 ** 6}M")
        print(t5)

    # Input configs
    # Create example inputs for the model
    input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 1024),  # bs x seq_len
        device=args.device,
        dtype=torch.int64,
        requires_grad=False,
    )

    example_inputs = {"input_ids": input, "decoder_input_ids": input}

    # Annotate split points
    add_split_points(t5, args.world_size)

    # Create pipeline
    t5_pipe = Pipe.from_tracing(
        t5,
        num_chunks=args.chunks,
        example_args=(),
        example_kwargs=example_inputs,
    )
    assert len(list(t5_pipe.split_gm.children())) == args.world_size
    if args.rank == 0:
        for i, sm in enumerate(t5_pipe.split_gm.children()):
            print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

    # Create schedule runtime
    stage = PipelineStage(
        t5_pipe,
        args.rank,
        device=args.device,
    )

    # Run
    # Measure first batch
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        if args.rank == 0:
            stage(example_inputs["input_ids"], example_inputs["decoder_input_ids"])
        elif args.rank == args.world_size - 1:
            out = stage()
        else:
            stage()
    torch.cuda.synchronize()
    end_time = time.time()
    first_batch = end_time - start_time

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(5):
        with torch.no_grad():
            if args.rank == 0:
                stage(example_inputs["input_ids"], example_inputs["decoder_input_ids"])
            elif args.rank == args.world_size - 1:
                out = stage()
            else:
                stage()
    torch.cuda.synchronize()
    end_time = time.time()

    if args.rank == args.world_size - 1:
        print(f"Time of first pass: {first_batch}")
        print(f"Average time per batch: {(end_time - start_time)/5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 2))
    )
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument(
        "--master_addr", type=str, default=os.getenv("MASTER_ADDR", "localhost")
    )
    parser.add_argument(
        "--master_port", type=str, default=os.getenv("MASTER_PORT", "29500")
    )
    parser.add_argument("--schedule", type=str, default="FillDrain")
    parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=1)

    args = parser.parse_args()

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")

    # Init process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run(args)