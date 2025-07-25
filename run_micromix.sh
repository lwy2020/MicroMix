#!/bin/bash
# path to your model 
MODEL=${1}

dir=$(pwd)
export CUDA_VISIBLE_DEVICE="0"

# zero-shot
python ${dir}/model/main.py ${MODEL} \
        --act_sort_metric mean \
        --tasks piqa,arc_challenge,boolq,winogrande,lambada_openai \
        --lm_eval_num_fewshot 0 \
        --lm_eval_limit -1\
        --multigpu\

#5-shot mmlu
python ${dir}/model/main.py ${MODEL}\
        --act_sort_metric mean\
        --tasks mmlu\
        --lm_eval_num_fewshot 5\
        --lm_eval_limit -1\
        --multigpu\

# # wikitext2 ppl
python ${dir}/model/main.py ${MODEL}\
        --act_sort_metric mean\
        --lm_eval_limit -1\
        --eval_ppl\
        --multigpu\
