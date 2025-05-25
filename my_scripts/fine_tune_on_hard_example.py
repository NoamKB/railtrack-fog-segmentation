import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
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
    config.snapshot = "model/bisenetv2_finetuned.pth"
    config.num_epochs = 10
    config.batch_size = 2  # Now we have 2 duplicated images

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Minimal augmentations
    transforms = [abm.HorizontalFlip(p=0.5)]
    data_transform = DataTransformBase(
        transforms=transforms,
        input_size=(config.img_height, config.img_width),
        normalize=True
    )

    train_dataset = ClearDataset(data_path="data/hard_example", transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    model = BiSeNetV2(n_classes=config.num_classes).to(device)

    if os.path.exists(config.snapshot):
        checkpoint = torch.load(config.snapshot, map_location="cpu")
        model.load_state_dict(checkpoint)

    # Loss and optimizer
    weights = [1.0 for _ in range(config.num_classes)]
    criterion = OHEMCELoss(thresh=config.ohem_ce_loss_thresh, weighted_values=weights)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-5,
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

    # Save model
    torch.save(model.state_dict(), "model/bisenetv2_finetuned_hard_example.pth")
    print("Fine-tuned model saved: bisenetv2_finetuned_hard_example.pth")

if __name__ == "__main__":
    main()
