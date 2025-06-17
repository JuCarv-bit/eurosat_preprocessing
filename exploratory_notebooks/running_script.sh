#!/bin/bash
jupyter nbconvert --to script eurosat_contrastive_stratified.ipynb
tmux new -d -s jobeurosatcontrastive-aug-reploss-2
tmux attach -t jobeurosatcontrastive-aug-reploss-2
conda activate myenv && python eurosat_contrastive_stratified.py > runs/jobeurosatcontrastive-aug-reploss-2.log 2>&1