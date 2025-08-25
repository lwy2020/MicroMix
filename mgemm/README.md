# MixedGemm for MicroMix

**MixedGemm** is a mixed-precision GEMM with quantize and reorder kernel performed on Blackwell GPUs(RTX5090).

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
