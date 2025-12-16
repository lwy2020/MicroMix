#!/bin/bash
# path to your model 
MODEL=${1}
dir=$(pwd)
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICE="0"

# zero-shot
# python ${dir}/model/main.py ${MODEL} \
#         --do_quant\
#         --tasks piqa,arc_challenge,boolq,winogrande,lambada_openai \
#         --lm_eval_num_fewshot 0 \
#         --lm_eval_limit -1\

python ${dir}/model/main.py ${MODEL} \
        --do_quant\
        --tasks humaneval,mbpp \
        --lm_eval_limit -1\
        --lm_eval_num_fewshot 3\


# # 5-shot mmlu
# python ${dir}/model/main.py ${MODEL}\
#         --act_sort_metric mean\
#         --tasks mmlu\
#         --lm_eval_num_fewshot 5\
#         --lm_eval_limit -1\

# # wikitext2 ppl
# python ${dir}/model/main.py ${MODEL}\
#         --act_sort_metric mean\
#         --lm_eval_limit -1\
#         --eval_ppl\

