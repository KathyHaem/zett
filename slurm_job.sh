#!/bin/bash

#SBATCH -J zett-test
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH -p gpu-ms,gpu-troja
#SBATCH --gres=gpu:4
#SBATCH --mem=128G


# because Flax weights are not merged in the main branch, we need to specify the revision of a PR containing Flax weights
python3 scripts/transfer.py \
    --target_model=mistralai/Mistral-7B-v0.1 \
    --revision=refs/pr/95 \
    --tokenizer_name=/lnet/work/people/limisiewicz/entangled-in-scripts/tokenizers/sp-bpe/ar-tr-zh-el-es-en-sw-hi-mr-ur-ta-te-th-ru-bg-he-ka-vi-fr-de/alpha-0.25_N-120000/ \
    --output=mistral-bpe-20-nolc-test \
    --model_class=AutoModelForCausalLM \
    --checkpoint_path=zett-hypernetwork-multilingual-Mistral-7B-v0.1 \
    --save_pt \
    # --lang_code=hi

# tokenizers tried:
# --tokenizer_name=/lnet/work/people/limisiewicz/entangled-in-scripts/tokenizers/sp-morfgram/ar-tr-zh-el-es-en-sw-hi-mr-ur-ta-te-th-ru-bg-he-ka-vi-fr-de/ \
#     --tokenizer_name=google/gemma-2b \
