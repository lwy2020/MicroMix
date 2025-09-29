# MicroMix
MicroMix is a mixed-precision quantization method using MXFP8/MXFP6/MXFP4. 

[[paper]](https://arxiv.org/abs/2508.02343)

![](/figures/main.png)

## 1. Installation
```bash
conda create -n micromix python=3.10 -y
conda activate micromix
```
Please make sure that [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) is in your environment.
```bash
git clone --recurse-submodules https://github.com/lwy2020/MicroMix.git
cd MicroMix
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 2. Usage

### 2.1 Preprocessing
Reorder_indices, p6_num, p8_num are needed for quantization:
```bash
python reorder_indices.py --model /PATH/TO/YOUR/MODEL/ --samples 32 --seqlen 2048 --act_sort_metric mean
```
Results are saved in saved/
### 2.2 Building Kernels
Please refer to `mgemm/README.md`
```bash
cd mgemm/
```
### 2.3 Zero-shot, Few-shot Accuracy and Perplexity Evaluation
```bash
bash test.sh /PATH/TO/YOUR/MODEL/
```

### 2.4 Coder Evaluation
```bash
bash eval_plus/test.sh Qwen/Qwen2.5-Coder-32B-Instruct  '32B'
```

If you want to use the MicroMix kernel but not our algorithm, you can directly set p4_num, p6_num, p8_num (line 41-43 in /model/qLinearLayer.py) as the numbers you want 😄

## 3. Efficiency Evaluation
MicroMix efficiency:
```bash
python benchmarks/benchmark_e2e_micromix.py --model 'llama-3.1-8b' --batch_size 8 --prefill_seq_len 2048
```
FP16 efficiency:
```bash
pip install transformers==4.56.2
python benchmarks/benchmark_e2e_fp16.py --model /PATH/TO/YOUR_MODEL --batch_size 8 --prefill_seq_len 2048
```
INT8 efficiency:
```bash
pip install bitsandbytes==0.47.0
python benchmarks/benchmark_e2e_int8.py --model /PATH/TO/YOUR_MODEL --batch_size 12 --prefill_seq_len 2048
```
## Citation
If you found this work helpful, please consider citing:
```bibtex
@misc{liu2025micromix,
    title={MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models},
    author={Wenyuan Liu and Haoqian Meng and Yilun Luo and Peng Zhang and Xindian Ma},
    year={2025},
    eprint={2508.02343},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Acknowledagement
Our code is built on the following repos, thank you for your contributions to community 👍:
- [Atom](https://github.com/efeslab/Atom.git)
- [QuaRot](https://github.com/spcl/QuaRot)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/main)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
