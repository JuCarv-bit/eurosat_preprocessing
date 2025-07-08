#!/bin/bash
# jupyter nbconvert --to script eurosat_contrastive_stratified.ipynb
conda  activate myenv
cd /users/c/carvalhj/projects/eurosat_preprocessing/exploratory_notebooks
tmux new-session -As jobeurosatcontrastive-aug-reploss-2-bs256_ep60-lrg-remov-gausblur
conda activate myenv &&  jupyter nbconvert --to script eurosat_contrastive.ipynb --output eurosat_contrastive-bs256_ep60-lrg-remov-gausblur
python eurosat_contrastive-bs256_ep60-lrg-remov-gausblur.py > runs/jobeurosat_contrastive-bs256_ep60-lrg-remov-gausblur.log 2>&1