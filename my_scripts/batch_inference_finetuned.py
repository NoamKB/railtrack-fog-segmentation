import os
import subprocess

# Paths
input_dir = "data/fog"
output_dir = "output/fog_results_finetuned"
snapshot_path = "model/bisenetv2_finetuned.pth"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"result_{filename}")

        if os.path.exists(output_path):
            continue

        subprocess.run([
            "python", "scripts/segmentation/test_one_image.py",
            "-snapshot", snapshot_path,
            "-image_path", input_path,
            "-output_image_path", output_path
        ])

