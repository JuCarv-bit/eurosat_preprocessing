tmux new -d -s jobeurosatcontrastive 
mkdir -p runs
bash -lc 'conda activate myenv && python eurosat_contrastive_stratified.py > runs/jobeurosatcontrastive.log 2>&1'