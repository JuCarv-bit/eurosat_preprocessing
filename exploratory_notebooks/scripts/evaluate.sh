mkdir -p /share/homes/carvalhj/projects/eurosat_preprocessing/models/random_weights
# --output-dir "/share/homes/carvalhj/projects/eurosat_preprocessing/models/random_weights" \
  # --weights "/share/homes/carvalhj/projects/eurosat_preprocessing/models/2025-08-22_17-07-16/simclr_seed42_bs256_temp0.2_Tepochs200_lr0.000375_epoch_200.pth" \

# define vars for each model weight  path, starting with loc-aware
loc_aware_weights="/share/homes/carvalhj/projects/eurosat_preprocessing/models/2025-08-22_01-07-36/simclr_seed42_bs256_temp0.2_Tepochs200_lr0.000375_epoch_200.pth"
simclr_weights="/share/homes/carvalhj/projects/eurosat_preprocessing/models/2025-08-22_01-06-21/simclr_seed42_bs256_temp0.2_Tepochs200_lr0.000375_epoch_200.pth"
old_yaware_lat_weights="/share/homes/carvalhj/projects/eurosat_preprocessing/models/2025-08-22_17-07-16/simclr_seed42_bs256_temp0.2_Tepochs200_lr0.000375_epoch_200.pth"
old_yaware_lon_weights="/share/homes/carvalhj/projects/eurosat_preprocessing/models/2025-08-23_00-18-42/simclr_seed42_bs256_temp0.2_Tepochs200_lr0.000375_epoch_200.pth"


# Random approach: without weights and with output-dir
python /share/homes/carvalhj/projects/eurosat_preprocessing/exploratory_notebooks/notebooks/main_evaluation.py \
  --output-dir "/share/homes/carvalhj/projects/eurosat_preprocessing/models/random_weights2" \
  --model resnet18 \
  --dataset "eurosat" \
  --n-classes 10 \
  --feature-dim 512 \
  --batch-size 256 \
  --k 5 \
  --l2norm

# Pretrained  model approach
# python /share/homes/carvalhj/projects/eurosat_preprocessing/exploratory_notebooks/notebooks/main_evaluation.py \
#   --weights "$loc_aware_weights" \
#   --model resnet18 \
#   --dataset "eurosat" \
#   --n-classes 10 \
#   --feature-dim 512 \
#   --batch-size 256 \
#   --k 5 \
#   --l2norm