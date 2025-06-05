#!/bin/bash

read_languages_from_file() {
    local file=$1
    local languages=()
    while IFS= read -r line; do
        languages+=("$line")
    done < "$file"
    echo "${languages[@]}"
}

if [[ $1 == *.txt ]]; then
    # If the argument is a .txt file, read the languages from the file
    languages=$(read_languages_from_file "$1")
else
    # If the argument is not a .txt file, treat it as the list of languages
    languages=$1
fi

mkdir -p ../persist/train/
mkdir -p ../persist/valid/
mkdir -p ../persist/large_tokenizers/

for lang in $languages
do
    gsutil -m cp gs://trc-transfer-data/hypertoken/datasets/multilingual/train/$lang.parquet ../persist/train/
    gsutil -m cp gs://trc-transfer-data/hypertoken/datasets/multilingual/valid/$lang.parquet ../persist/valid/
    gsutil -m cp -r gs://trc-transfer-data/hypertoken/datasets/multilingual/large_tokenizers/$lang ../persist/large_tokenizers/
done
