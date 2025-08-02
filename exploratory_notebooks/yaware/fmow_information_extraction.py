import bz2, json

PATH_manifest = '/users/c/carvalhj/datasets/fmow/fmow-rgb/manifest.json.bz2'


# Load and inspect
with bz2.open(PATH_manifest, 'rt') as f:
    data = json.load(f)

# print the first 20 entries
for name in data[:20]:
    print(name)

GROUND_TRUTH = "/users/c/carvalhj/datasets/fmow/groundtruth/seq_gt_mapping.json"

# Load ground truth
with open(GROUND_TRUTH, 'r') as f:
    ground_truth = json.load(f)

# Print the first 20 entries of ground truth
for name in list(ground_truth)[:20]:
    print(name)