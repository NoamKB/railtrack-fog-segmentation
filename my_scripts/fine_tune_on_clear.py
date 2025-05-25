import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rail_marking.segmentation.models import BiSeNetV2, OHEMCELoss
from rail_marking.segmentation.trainer import BiSeNetV2Trainer
from rail_marking.segmentation.data_loader import Rs19dDataset, DataTransformBase
from cfg import BiSeNetV2Config

import cv2
import torch
from torch.utils.data import DataLoader
import albumentations as abm
import torch.optim.lr_scheduler as lr_scheduler

def main():
    config = BiSeNetV2Config()
    config.saved_model_path = "model/"
    config.snapshot = "model/bisenetv2_checkpoint_BiSeNetV2_epoch_300.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = [
        abm.HorizontalFlip(p=0.5),
        abm.RandomBrightnessContrast(p=0.5),
        abm.MotionBlur(p=0.2),
        abm.RandomGamma(p=0.2),
    ]

    data_transform = DataTransformBase(
        transforms=transforms,
        input_size=(config.img_height, config.img_width),
        normalize=True
    )

    train_dataset = Rs19dDataset(data_path="data/clear", phase="train", transform=data_transform)
    val_dataset = Rs19dDataset(data_path="data/clear", phase="val", transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    model = BiSeNetV2(n_classes=config.num_classes)
    model = model.to(device)

    weights = train_dataset.weighted_class()
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

    if config.snapshot and os.path.exists(config.snapshot):
        trainer.resume_checkpoint(config.snapshot)

    trainer.train()

if __name__ == "__main__":
    main()
