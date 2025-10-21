# MixedGemm for MicroMix

**MixedGemm** is a mixed-precision GEMM with quantize and reorder kernel performed on Blackwell GPUs (RTX5090).

## Building Kernels

1. Prepare environment
```
sudo apt-get update
sudo apt-get install python3-dev

sudo apt update
sudo apt install cmake

conda install pybind11
```


2. Make
```
bash make.sh
```

## Run Kernels Benchmark

```sh
cd mgemm
./build/bench_mxf4f6f8 M N K Iter TypeAxTypeB --validate

```
Example

```sh
./build/bench_mxf4f6f8 32 4096 4096 100 8x4 --validate
```

Output:
```
Validation Enabled.
Running MXFP6 (E3M2) x MXFP4 (E2M1)

GEMM VAL PASS!

Iter Runs =  100
BM=  128, BN=  128, BK=  128, Pipeline Stage = 2
M =   32, N = 4096, K = 4096, Time =   0.19270399 ms, AVG Performance =     5.5720 TFLOPs
```
