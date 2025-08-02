#!/bin/bash
tmux new-session -As train-yaware200
# cd /users/c/carvalhj/projects/eurosat_preprocessing/exploratory_notebooks
cd  exploratory_notebooks/notebooks
conda activate yaware_eurosat 
jupyter nbconvert --to script yaware_newarchsimclr_original_loss.ipynb --output yaware_newarchsimclr_original_200
mkdir -p runs
python yaware_newarchsimclr_original_200.py > runs/yaware_newarchsimclr_original_200.log 2>&1