#!/bin/bash
tmux new-session -As jobeurosatcontrastive-aug-reploss-2-bs256_ep60-lrg-remov-gausblur
cd /users/c/carvalhj/projects/eurosat_preprocessing/exploratory_notebooks
conda activate myenv &&  jupyter nbconvert --to script eurosat_contrastive.ipynb --output eurosat_contrastive-bs256_ep60-lrg-remov-gausblur
python eurosat_contrastive-bs256_ep60-lrg-remov-gausblur.py > runs/jobeurosat_contrastive-bs256_ep60-lrg-remov-gausblur.log 2>&1