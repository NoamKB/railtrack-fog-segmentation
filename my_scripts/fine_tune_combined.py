
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim.lr_scheduler as lr_scheduler
import albumentations as abm

from rail_marking.segmentation.models import BiSeNetV2, OHEMCELoss
from rail_marking.segmentation.trainer import BiSeNetV2Trainer
from rail_marking.segmentation.data_loader.data_transform_base import DataTransformBase
from cfg import BiSeNetV2Config
from my_scripts.clear_dataset import ClearDataset

def main():
    config = BiSeNetV2Config()
    config.saved_model_path = "model/"
    config.snapshot = "model/bisenetv2_finetuned.pth"  # starting point
    config.num_epochs = 20
    config.batch_size = 4  # moderate batch size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stronger augmentations
    transforms = [
        abm.HorizontalFlip(p=0.5),
        abm.RandomBrightnessContrast(p=0.4),
        abm.RandomGamma(p=0.3),
        abm.GaussianBlur(p=0.2)
    ]
    data_transform = DataTransformBase(
        transforms=transforms,
        input_size=(config.img_height, config.img_width),
        normalize=True
    )

    # Load both datasets
    clear_dataset = ClearDataset(data_path="data/clear", transform=data_transform)
    hard_example_dataset = ClearDataset(data_path="data/hard_example", transform=data_transform)

    combined_dataset = ConcatDataset([clear_dataset, hard_example_dataset])
    train_loader = DataLoader(combined_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    # Load model
    model = BiSeNetV2(n_classes=config.num_classes).to(device)

    if os.path.exists(config.snapshot):
        checkpoint = torch.load(config.snapshot, map_location="cpu")
        model.load_state_dict(checkpoint)

    # Loss and optimizer
    weights = [1.0 for _ in range(config.num_classes)]
    criterion = OHEMCELoss(thresh=config.ohem_ce_loss_thresh, weighted_values=weights)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-4,  # slightly higher LR than hard tuning only
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - epoch / config.num_epochs) ** 0.9)

    trainer = BiSeNetV2Trainer(
        model=model,
        criterion=criterion,
        metric_func=None,
        optimizer=optimizer,
        data_loaders_dict={"train": train_loader, "val": train_loader},
        config=config,
        scheduler=scheduler,
        device=device
    )

    trainer.train()

    torch.save(model.state_dict(), "model/bisenetv2_finetuned_combined.pth")
    print("Final combined fine-tuned model saved: bisenetv2_finetuned_combined.pth")
if __name__ == "__main__":
    main()
