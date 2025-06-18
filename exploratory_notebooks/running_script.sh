#!/bin/bash
# jupyter nbconvert --to script eurosat_contrastive_stratified.ipynb
conda  activate myenv
cd /users/c/carvalhj/projects/eurosat_preprocessing/exploratory_notebooks
jupyter nbconvert --to script eurosat_contrastive.ipynb --output eurosat_contrastive-bs256_ep100
tmux new-session -As jobeurosatcontrastive-aug-reploss-2-bs256_ep100
conda activate myenv && python eurosat_contrastive-bs256_ep100.py > runs/jobeurosat_contrastive-bs256_ep100.log 2>&1