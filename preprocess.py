import os
import csv
import rasterio

base_dir = "/home/juliana/internship_LINUX/datasets/EuroSAT_MS"
output_csv = "/home/juliana/internship_LINUX/code/eurosat_preprocessing/eurosat_metadata.csv"

with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "label", "crs", "bounds_left", "bounds_bottom", "bounds_right", "bounds_top", "transform"])

    for subdir, _, files in os.walk(base_dir):
        label = os.path.basename(subdir)  # This is the class name
        for file in files:
            if file.endswith(".tif"):
                path = os.path.join(subdir, file)
                with rasterio.open(path) as src:
                    bounds = src.bounds
                    writer.writerow([
                        file,
                        label,
                        src.crs.to_string() if src.crs else "None",
                        bounds.left,
                        bounds.bottom,
                        bounds.right,
                        bounds.top,
                        list(src.transform)
                    ])

print(f"Metadata saved with labels in: {output_csv}")
