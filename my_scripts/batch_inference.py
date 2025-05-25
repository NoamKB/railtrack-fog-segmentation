import os
import subprocess

# path to files
input_dir = 'data/fog'
output_dir = 'output/fog_results'
snapshot_path = 'model/bisenetv2_checkpoint_BiSeNetV2_epoch_300.pth'

# create output file
os.makedirs(output_dir, exist_ok=True)

# iterate over all files
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f'result_{filename}')
        subprocess.run([
            'python', 'scripts/segmentation/test_one_image.py',
            '-snapshot', snapshot_path,
            '-image_path', input_path,
            '-output_image_path', output_path
        ])
