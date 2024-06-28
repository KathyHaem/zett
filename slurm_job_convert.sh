#!/bin/bash

#SBATCH -J convert-checkpoint
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --mem=64G

python3 scripts/convert_checkpoint_to_pt.py --checkpoint_path mistral-gemma-test
