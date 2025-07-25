# MicroMix
MicroMix is a mixed-precision quantization method using MXFP8/MXFP6/MXFP4.

![](/figures/main.png)

## 1. Installation
```bash
conda create -n micromix python=3.10 -y
conda activate micromix
```
Please make sure that [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) is in your environment.
```bash
git clone --recurse-submodules https://github.com/lwy0000/MircoMix.git
cd MicroMix
pip install -r requirements.txt
```

## 2. Usage

### 2.1 Preprocessing
Reorder_indices, p6_num, p8_num are needed for quantization:
```bash
python reorder_indices.py --model /PATH/TO/YOUR/MODEL/ --samples 32 --seqlen 2048 --act_sort_metric mean
```
Results are saved in ./saved/
### 2.2 Building Kernels
Please refer to `kernels/mgemm/README.md`
```bash
cd mgemm/
```
### 2.3 Accuracy Evaluation
```bash
bash run_micromix.sh /PATH/TO/YOUR/MODEL/
```
If you want to use the MicroMix kernel but not our algorithm, you can directly set p4_num, p6_num, p8_num (line 41-43 in /model/qLinearLayer.py) as the number you want 😄

## 3. Efficiency Evaluation
Since [FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/main) is integrated into our decoderlayer implementation, please install FlashInfer:
```bash
git clone --recurse-submodules https://github.com/flashinfer-ai/flashinfer.git
cd flashinfer
python -m pip install -v .
```
DecoderLayer efficiency:
```bash
python benchmarks/benchmark_layer_micromix.py --model 'llama-3.1-8b' --batch_size 32 --prefill_seq_len 2048
```
TensorRT efficiency:
```bash
pip install tensorrt
python benchmark/trt-fp8-prefill-llama.py
```

## Acknowledagement
Our code is built on the following repos, thank you for your contributions to community 👍:
- [Atom](https://github.com/efeslab/Atom.git)
- [QuaRot](https://github.com/spcl/QuaRot)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/main)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
