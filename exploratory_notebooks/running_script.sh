#!/bin/bash
# jupyter nbconvert --to script eurosat_contrastive_stratified.ipynb
jupyter nbconvert --to script eurosat_contrastive.ipynb --output eurosat_contrastive-bs256
tmux new-session -As jobeurosatcontrastive-aug-reploss-2-bs256
conda activate myenv && python eurosat_contrastive-bs256.py > runs/jobeurosat_contrastive-bs256.log 2>&1