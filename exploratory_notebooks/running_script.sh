#!/bin/bash
tmux new-session -As jobeurosatcontrastive-yaware_v2
# cd /users/c/carvalhj/projects/eurosat_preprocessing/exploratory_notebooks
cd  exploratory_notebooks/yaware
conda activate yaware_eurosat 
jupyter nbconvert --to script yaware_newarchsimclr.ipynb --output eurosat_contrastiveyaware_v2
mkdir -p runs
python eurosat_contrastiveyaware_v2.py > runs/jobeurosat_contrastiveyaware_v2.log 2>&1