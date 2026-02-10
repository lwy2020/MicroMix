#!/bin/bash
# path to your model 
MODEL=${1}
dir=$(pwd)
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICE="0"

# zero-shot
python ${dir}/model/main.py ${MODEL} \
    --batch_size "auto:1"\
    --tasks piqa,arc_challenge,boolq,winogrande,lambada_openai \
    --lm_eval_num_fewshot 0 \
    --lm_eval_limit -1\
    --do_quant\

# 5-shot mmlu
python ${dir}/model/main.py ${MODEL}\
    --tasks mmlu\
    --batch_size "6"\
    --lm_eval_num_fewshot 5\
    --lm_eval_limit -1\
    --do_quant\

# # wikitext2 ppl
python ${dir}/model/main.py ${MODEL}\
    --lm_eval_limit -1\
    --eval_ppl\
    --do_quant\
