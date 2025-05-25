import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import cv2
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import albumentations as abm

from rail_marking.segmentation.models import BiSeNetV2, OHEMCELoss
from rail_marking.segmentation.trainer import BiSeNetV2Trainer
from rail_marking.segmentation.data_loader.data_loader_base import BaseDataset
from rail_marking.segmentation.data_loader.data_transform_base import DataTransformBase
from my_scripts.clear_dataset import ClearDataset
from cfg import BiSeNetV2Config

def main():
    config = BiSeNetV2Config()
    config.saved_model_path = "model/"
    config.snapshot = "model/bisenetv2_checkpoint_BiSeNetV2_epoch_300.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Augmentations
    transforms = [
        abm.HorizontalFlip(p=0.5),
        abm.RandomBrightnessContrast(p=0.4),
        abm.GaussianBlur(p=0.2),
        abm.RandomGamma(p=0.2),
    ]
    data_transform = DataTransformBase(transforms=transforms,
                                       input_size=(config.img_height, config.img_width),
                                       normalize=True)

    train_dataset = ClearDataset(data_path="data/clear", transform=data_transform)
    val_dataset = ClearDataset(data_path="data/clear", transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    model = BiSeNetV2(n_classes=config.num_classes).to(device)
    weights = [1.0 for _ in range(config.num_classes)]
    criterion = OHEMCELoss(thresh=config.ohem_ce_loss_thresh, weighted_values=weights)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr_rate / (config.batch_size * config.batch_multiplier),
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - epoch / config.num_epochs) ** 0.9)

    trainer = BiSeNetV2Trainer(
        model=model,
        criterion=criterion,
        metric_func=None,
        optimizer=optimizer,
        data_loaders_dict={"train": train_loader, "val": val_loader},
        config=config,
        scheduler=scheduler,
        device=device
    )

    # Load from snapshot (on CPU if needed)
    if config.snapshot and os.path.exists(config.snapshot):
        checkpoint = torch.load(config.snapshot, map_location=torch.device("cpu"))
        trainer._model.load_state_dict(checkpoint["state_dict"])
        trainer._optimizer.load_state_dict(checkpoint["optimizer"])
        trainer._start_epoch = checkpoint["epoch"] + 1

    trainer.train()

    # Save final model manually to fixed filename
    torch.save(model.state_dict(), "model/bisenetv2_finetuned.pth")
    print("✅ Fine-tuned model saved to: model/bisenetv2_finetuned.pth")


if __name__ == "__main__":
    main()
