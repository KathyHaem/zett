#!/bin/bash

set -x

#LANGS=("cs" "uk" "ja" "he")

#for LANG in "${LANGS[@]}"; do
#  python3 data/prepare.py --include_langs="${LANG}"
  # bash scripts/make_tokenizers.sh "${LANG}"
#done

LANG_PAIRS=("ru-en")

for PAIR in "${LANG_PAIRS[@]}"; do

  python3 scripts/make_spm.py --dataset_path "/mounts/work/haemmerl/persist/train/" \
      --output "/mounts/work/haemmerl/persist/tokenizers/${PAIR}" --langs "${PAIR}" --bilingual
  # because Flax weights are not merged in the main branch, we need to specify the revision of a PR containing Flax weights
  python3 scripts/transfer.py \
      --target_model=mistralai/Mistral-7B-v0.1 \
      --revision=refs/pr/95 \
      --tokenizer_name="/mounts/work/haemmerl/persist/tokenizers/${PAIR}/" \
      --output=mistral-50kSPM-"${PAIR}" \
      --model_class=AutoModelForCausalLM \
      --checkpoint_path=zett-hypernetwork-multilingual-Mistral-7B-v0.1 \
      --save_pt \
      --lang_code="${PAIR}" \
      --bilingual
done
