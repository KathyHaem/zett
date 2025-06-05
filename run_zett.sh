#!/bin/bash

set -x

LANGS=("ar" "bg" "de" "el" "en" "es" "fr" "ru" "ur")

for LANG in "${LANGS[@]}"; do
  python3 data/prepare.py --include_langs="${LANG}"
  bash scripts/make_tokenizers.sh "${LANG}"
done

for LANG in "${LANGS[@]}"; do
  # because Flax weights are not merged in the main branch, we need to specify the revision of a PR containing Flax weights
  python3 scripts/transfer.py \
      --target_model=mistralai/Mistral-7B-v0.1 \
      --revision=refs/pr/95 \
      --tokenizer_name="/mounts/work/haemmerl/persist/tokenizers/${LANG}/" \
      --output=mistral-50kSPM-"${LANG}" \
      --model_class=AutoModelForCausalLM \
      --checkpoint_path=zett-hypernetwork-multilingual-Mistral-7B-v0.1 \
      --save_pt \
      --lang_code="${LANG}"
done

# tokenizers tried:
# --tokenizer_name=/lnet/work/people/limisiewicz/entangled-in-scripts/tokenizers/sp-bpe/ar-tr-zh-el-es-en-sw-hi-mr-ur-ta-te-th-ru-bg-he-ka-vi-fr-de/alpha-0.25_N-120000/
# --tokenizer_name=/lnet/work/people/limisiewicz/entangled-in-scripts/tokenizers/sp-morfgram/ar-tr-zh-el-es-en-sw-hi-mr-ur-ta-te-th-ru-bg-he-ka-vi-fr-de/ \
#     --tokenizer_name=google/gemma-2b \
