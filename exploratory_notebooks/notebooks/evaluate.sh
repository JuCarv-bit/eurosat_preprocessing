mkdir -p /share/homes/carvalhj/projects/eurosat_preprocessing/models/random_weights
# --output-dir "/share/homes/carvalhj/projects/eurosat_preprocessing/models/random_weights" \

python /share/homes/carvalhj/projects/eurosat_preprocessing/exploratory_notebooks/notebooks/main_evaluate.py \
  --weights "/share/homes/carvalhj/projects/eurosat_preprocessing/models/2025-08-22_01-06-21/simclr_seed42_bs256_temp0.2_Tepochs200_lr0.000375_epoch_200.pth" \
  --model resnet18 \
  --dataset "eurosat" \
  --n-classes 10 \
  --feature-dim 512 \
  --batch-size 256 \
  --k 5 \
  --l2norm