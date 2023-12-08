# PiPPy device map playground

## Install

Requires nightly `torch` build and `pippy` to be installed:

```
conda install pytorch pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
```

```
pip install git+https://github.com/pytorch/PiPPy accelerate transformers
```

## Results

Tests were ran with a batch size of `n_gpus`. In this case they were ran on 
two 4090's so a batch size of 2 (or 1 per each GPU when split).

When `Accelerate` was used, a device map was generated that could roughly split
the model evenly between each GPU

### Bert (`bert-base-cased`)

| Time Elapsed (s) | Accelerate/Sequential | PiPPy Example | PiPPy + Accelerate (automated) |
|---|---|---|---|
| First batch | 0.2478 | 0.1966 | 0.1683 |
| Avg for the rest | 0.0108 | 0.00336 | 0.00218 |


### T5

| Time Elapsed (s) | Accelerate/Sequential | PiPPy Example | PiPPy + Accelerate (automated) |
|---|---|---|---|
| First batch | 0.2986 | 0.2313 | 0.202 |
| Avg for the rest | 0.03167 | 0.02434 | 0.0132 |
