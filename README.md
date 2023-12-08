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

### Bert

| Time Elapsed (s) | Accelerate/Sequential | PiPPy Example | PiPPy + Accelerate (automated) |
|---|---|---|---|
| First batch | 0.2478 | 0.1732 | 0.1743 |
| Avg for five following | 0.0108 | 0.0022 | 0.0064 |

### GPT2

| Time Elapsed (s) | Accelerate/Sequential | PiPPy Example | PiPPy + Accelerate (automated) |
|---|---|---|---|
| First batch | 0.2745 | 0.2146 | 0.2216 |
| Avg for five following | 0.0341 | 0.0117 | 0.0136 |

### T5

| Time Elapsed (s) | Accelerate/Sequential | PiPPy Example | PiPPy + Accelerate (automated) |
|---|---|---|---|
| First batch | 0.2986 | 0.1961 | 0.2608 |
| Avg for five following | 0.03167 | 0.01056 | 0.0167 |
