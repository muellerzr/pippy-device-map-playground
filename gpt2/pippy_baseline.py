# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 4 pippy_bert.py

import argparse
import os
import time
import torch
import torch.distributed as dist

from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineStage import PipelineStage

from transformers import GPT2ForSequenceClassification, GPT2Config


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_split_points(gpt2, nranks):
    decoders_per_rank = (gpt2.config.n_layer + nranks - 1) // nranks
    print(f"decoders_per_rank = {decoders_per_rank}")
    nstages = 1
    for i in range(1, gpt2.config.n_layer // decoders_per_rank):
        annotate_split_points(
            gpt2,
            {
                f"transformer.h.{i * decoders_per_rank}": PipeSplitWrapper.SplitPoint.BEGINNING
            },
        )
        nstages += 1
    assert nstages == nranks, f"nstages = {nstages} nranks = {nranks}"


def run(args):
    # Model configs
    config = GPT2Config()
    print("Using device:", args.device)

    # Create model
    model_class = GPT2ForSequenceClassification
    bert = model_class(config)
    bert.to(args.device)
    bert.eval()
    if args.rank == 0:
        print(bert.config)
        print(f"Total number of params = {get_number_of_params(bert) // 10 ** 6}M")
        print(bert)

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

    example_inputs = {"input_ids": input}
    input_ids = example_inputs["input_ids"]

    # Annotate split points
    add_split_points(bert, args.world_size)

    # Create pipeline
    bert_pipe = Pipe.from_tracing(
        bert,
        num_chunks=args.chunks,
        example_args=(input_ids,),
    )
    nstages = len(list(bert_pipe.split_gm.children()))
    assert nstages == args.world_size, f"nstages = {nstages} nranks = {args.world_size}"
    if args.rank == 0:
        for i, sm in enumerate(bert_pipe.split_gm.children()):
            print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

    # Create schedule runtime
    stage = PipelineStage(
        bert_pipe,
        args.rank,
        device=args.device,
    )

    # Run
    # Measure first batch
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        if args.rank == 0:
            stage(example_inputs["input_ids"])
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
                stage(example_inputs["input_ids"])
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
