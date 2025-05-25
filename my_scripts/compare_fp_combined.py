
import os
import subprocess

# Image to test
image_path = "data/fog/123-2-1-29_11_2022-09_21_00_00001344.png"
output_dir = "output/hard_example_comparison_combined"
os.makedirs(output_dir, exist_ok=True)

# Models to compare
models = {
    "before": "model/bisenetv2_finetuned.pth",
    "after_combined": "model/bisenetv2_finetuned_combined.pth"
}

# Run inference using both models
for tag, model_path in models.items():
    output_path = os.path.join(output_dir, f"result_{tag}.png")
    subprocess.run([
        "python", "scripts/segmentation/test_one_image.py",
        "-snapshot", model_path,
        "-image_path", image_path,
        "-output_image_path", output_path
    ])

print("Comparison images saved in:", output_dir)
