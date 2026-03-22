# BitNet Inference Kernel

This repository provides a highly efficient GEMV kernel implementation for the BitNet model, optimized for W2A8 inference — 2-bit weights and 8-bit activations. It is tailored for use with the [BitNet-b1.58-2B-4T](https://arxiv.org/abs/2504.12285) model.

## Features

- Support for W2A8 (2-bit weight × 8-bit activation) GEMV computation  
- Custom CUDA kernels with low-latency execution  
- Optimizations for memory access, decoding, and compute throughput  

## Usage

Installation and kernel performance tests:

```bash
# (Recommended) Create a new conda environment
conda create --name bitnet-gpu "python<3.13"
conda activate bitnet-gpu

# Install dependencies
pip install -r requirements.txt

# Build the kernel
cd bitnet_kernels
bash compile.sh
cd ..

# Run performance tests
python test.py
```

End-to-end inference:

```bash
# IMPORTANT: run these commands from the gpu/ directory
# e.g. /path/to/BitNet/gpu

# Download and convert the BitNet-b1.58-2B model
mkdir checkpoints
hf download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir ./checkpoints/bitnet-b1.58-2B-4T-bf16
python ./convert_safetensors.py --safetensors_file ./checkpoints/bitnet-b1.58-2B-4T-bf16/model.safetensors --output checkpoints/model_state.pt --model_name 2B
python ./convert_checkpoint.py --input ./checkpoints/model_state.pt
rm ./checkpoints/model_state.pt

# Inference
python3 ./generate.py ./checkpoints/ --interactive --chat_format
```

If you previously ran `hf download` from `gpu/bitnet_kernels`, your model may be at `./bitnet_kernels/checkpoints/...`.
In that case, run:

```bash
mv bitnet_kernels/checkpoints/ ./
python ./convert_safetensors.py --safetensors_file ./checkpoints/bitnet-b1.58-2B-4T-bf16/model.safetensors --output checkpoints/model_state.pt --model_name 2B
python ./convert_checkpoint.py --input ./checkpoints/model_state.pt
rm ./checkpoints/model_state.pt
```


## Troubleshooting (Linux / Ubuntu)

### `nvcc: command not found`

If CUDA is installed but `nvcc` is not on your `PATH`, set CUDA env vars before building:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

Then build from [bitnet_kernels](bitnet_kernels):

```bash
cd bitnet_kernels
bash compile.sh
```

### Ubuntu 25.10 / GCC 15 host compiler issue

On newer Ubuntu versions, system `g++` may be 15.x, while CUDA 12.9 typically supports GCC <= 14.

If you see an error similar to:

```text
unsupported GNU version! gcc versions later than 14 are not supported
```

Install `g++-14` and rebuild:

```bash
sudo apt install g++-14
cd bitnet_kernels
bash compile.sh
```

The script [bitnet_kernels/compile.sh](bitnet_kernels/compile.sh) now:

- auto-detects `nvcc` and CUDA locations,
- prefers CUDA 12.9 by default,
- checks host compiler compatibility,
- prints actionable guidance when GCC is too new.

If you see errors like:

```text
bits/mathcalls.h(...): error: exception specification is incompatible with that of previous function "cospi" / "sinpi" / "rsqrt"
```

this is a known CUDA + newer glibc header interaction. The updated script already applies compatibility feature-macro flags for this case.

Optional overrides:

- `BITNET_CUDA_HOME=/path/to/cuda bash compile.sh` to force a CUDA version.
- `BITNET_ALLOW_UNSUPPORTED_COMPILER=1 bash compile.sh` to try unsupported compiler mode (at your own risk).

## Optimizations

### Weight Permutation

The weight matrix is divided into 16×32 blocks to optimize memory access patterns.  

Within each block, values are stored contiguously in memory and permuted to facilitate efficient access and processing.  

See `convert_checkpoint.py` for details.

### Fast Decoding

Every 16 two-bit values are packed into a single 32-bit integer using the following interleaving pattern:  
```
[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
```

This layout is designed to accelerate decoding by enabling efficient extraction of 4 values at a time into `int8`.

### `dp4a` Instruction

We use the `dp4a` instruction to accelerate low-precision dot product operations.  

This instruction performs a dot product between two 4-element vectors (each stored in a 32-bit word as 8-bit integers) and accumulates the result into a 32-bit integer.  

It significantly improves GEMV throughput when processing quantized weights and activations.


## Performance

### Kernel Benchmarks

Tested on NVIDIA A100 40GB GPU, our custom W2A8 kernel shows significant speedups over standard BF16 implementations:

| Shape (N×K)         | W2A8 Latency (us) | BF16 Latency (us) | Speedup Ratio        |
|---------------------|-------------------|-------------------|----------------------|
| 2560 × 2560         | 13.32             | 18.32             |   1.38               |
| 3840 × 2560         | 14.90             | 18.87             |   1.27               |
| 13824 × 2560        | 18.75             | 59.51             |   3.17               |
| 2560 × 6912         | 14.49             | 37.78             |   2.61               |
| 3200 × 3200         | 14.61             | 19.08             |   1.31               |
| 4800 × 3200         | 13.09             | 21.84             |   1.67               |
| 3200 × 10240        | 19.64             | 60.79             |   3.10               |
| 20480 × 3200        | 30.99             | 112.39            |   3.63               |

### End-to-End Generation Latency

Compared to a similarly-sized BF16 model (Gemma-2-2B using vLLM), BitNet-b1.58-2B with our kernel achieves consistent speedups across workloads:

| Input Length | Output Length | BF16 Latency (ms) | W2A8 Latency (ms) | Speedup Ratio |
| --- | --- | --- | --- | --- |
| 64 | 16 | 187.64 | 57.40 | 3.27 |
| 64 | 32 | 353.50 | 112.22 | 3.15 |
| 64 | 64 | 683.23 | 221.08 | 3.09 |
| 256 | 16 | 183.14 | 61.24 | 2.99 |
| 256 | 32 | 353.14 | 115.47 | 3.06 |
| 256 | 64 | 684.24 | 224.16 | 3.05 |
| 512 | 16 | 208.99 | 68.06 | 3.07 |
| 512 | 32 | 354.33 | 122.72 | 2.89 |
| 512 | 64 | 709.65 | 231.82 | 3.06 |

*Note: Comparison uses equivalent-sized models (2B parameters) on NVIDIA A100 40GB GPU.*