import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
from glob import glob
from tqdm import tqdm
import torch
from rail_marking.segmentation.deploy import RailtrackSegmentationHandler
from cfg import BiSeNetV2Config

def main():
    # Paths
    input_dir = "data/clear/images"
    output_dir = "data/clear/masks"
    snapshot_path = "model/bisenetv2_checkpoint_BiSeNetV2_epoch_300.pth"

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    handler = RailtrackSegmentationHandler(snapshot_path, BiSeNetV2Config())
    handler._model = handler._model.to(device)

    # Get image list
    image_paths = glob(os.path.join(input_dir, "*.png"))

    print(f"Generating pseudo-labels for {len(image_paths)} images...")

    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        mask = handler.run(img, only_mask=True)

        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)

        # Save grayscale mask
        cv2.imwrite(save_path, mask)

    print(f"✅ Pseudo-labels saved in: {output_dir}")

if __name__ == "__main__":
    main()
