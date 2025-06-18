#!/bin/bash
# jupyter nbconvert --to script eurosat_contrastive_stratified.ipynb
jupyter nbconvert --to script eurosat_contrastive.ipynb --output eurosat_contrastive-bsrep32
tmux new-session -As jobeurosatcontrastive-aug-reploss-2-bsrep32
conda activate myenv && python eurosat_contrastive-bsrep32.py > runs/jobeurosat_contrastive-bsrep32.log 2>&1