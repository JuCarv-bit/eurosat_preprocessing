mkdir -p /share/homes/carvalhj/projects/eurosat_preprocessing/models/random_weights
# --output-dir "/share/homes/carvalhj/projects/eurosat_preprocessing/models/random_weights" \

python /share/homes/carvalhj/projects/eurosat_preprocessing/exploratory_notebooks/notebooks/main_evaluation.py \
  --weights "/share/homes/carvalhj/projects/eurosat_preprocessing/models/2025-08-22_15-59-59/simclr_seed42_bs256_temp0.2_Tepochs2_lr0.000375_epoch_002.pth" \
  --model resnet18 \
  --dataset "eurosat" \
  --n-classes 10 \
  --feature-dim 512 \
  --batch-size 256 \
  --k 5 \
  --l2norm