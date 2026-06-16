import os
from datetime import datetime
import statistics as stat
import argparse

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import my_utils



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="datasets")
    parser.add_argument("--save-dir", type=str, default="weights/Segformer")

    parser.add_argument("--device", type=str, default="default")

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--warmup-epochs", type=int, default=30)

    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument(
        "--effective-batchsize",
        type=int,
        default=8
    )

    parser.add_argument("--train-ex", type=int, default=64)
    parser.add_argument("--val-ex", type=int, default=32)

    parser.add_argument("--encoder", type=str, default="resnet50")

    parser.add_argument("--save-every", type=int, default=50)

    return parser.parse_args()


args = parse_args()

req_size = 512
transform = A.Compose([
    A.RandomSizedCrop(min_max_height=[256, 1024], size = [512, 512], w2h_ratio=1, p = 1),
    A.RandomBrightnessContrast(p = 0.5, brightness_limit=[-0.1, 0.1], contrast_limit=[-0.1, 0.1], brightness_by_max=False),
    A.GaussNoise(p = 0.1, std_range=(0.02, 0.05)),
    A.GaussianBlur(p = 0.1, sigma_limit = [0.1, 0.2]),

    A.CLAHE(
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    p=0.3
    ),
    A.GridDistortion(
    num_steps=5,
    distort_limit=0.2,
    p=0.2
    ),

    my_utils.PercentileNormalize(p_low=0),

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomRotate90(p=0.5),
    
    A.Normalize(),
], additional_targets={'target': 'mask'})
val_transform = A.Compose([
    A.RandomSizedCrop(min_max_height=[512, 512], size = [512, 512], w2h_ratio=1, p = 1),
    my_utils.PercentileNormalize(p_low=0),
    A.Normalize(),
], additional_targets={'target': 'mask'})

datasetus = my_utils.CoreDataset( args.dataset, train_ex=args.train_ex, val_ex=args.val_ex, train_transform=transform, val_transform=val_transform)

model = smp.Segformer(encoder_name=args.encoder, in_channels=1, classes=1)

if args.device == 'default':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = args.device
epochs = args.epochs
warmup_epochs = args.warmup_epochs
lr = args.lr
effective_batchsize = args.effective_batchsize

def lr_lambda(current_epoch):
    if current_epoch < warmup_epochs:
        return float(current_epoch + 1) / warmup_epochs
    return 1.0 

optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lr_lambda
)

plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',         
    factor=0.2,    
    patience=5,            
    min_lr=1e-9,
    threshold = 1e-3          
)

loss_fn = nn.BCEWithLogitsLoss(reduction='none')
dice_loss = my_utils.DiceLoss(reduction='none')
model = model.to(device)

log_dir = (
    f"Runs/SegFormer {args.encoder} "
    f"{datetime.now().strftime('%d_%m %H_%M')}"
)

writer = SummaryWriter(log_dir=log_dir)

writer.add_text(
    "config",
    "\n".join(
        f"{k}: {v}"
        for k, v in vars(args).items()
    )
)

log_dir = f"Runs/SegFormer Resnet50 {datetime.now().strftime('%d_%m %H_%M')}"
writer = SummaryWriter(log_dir=log_dir)

metric_names = [
    "train loss",
    "validation loss",
    "iou",
    "prauc",
    "f1",
    "precision",
    "recall",
]

for epoch in range(epochs):
    metrics = {metric: {ds: 0.0 for ds in datasetus.dataset_list} for metric in metric_names}

    optimizer.zero_grad()
    datasetus.train()
    model.train()
    train_loader = my_utils.ThreadedLoader(datasetus, max_queue_size=5).start()
    for idx, batch in enumerate(tqdm(train_loader)):
        images = batch["image"].to(device).unsqueeze(1)    # [B, С, H, W]
        targets = batch["target"].float().to(device)       # [B, H, W]
        flags = batch["flag"]

        micro_batches = torch.split(images, effective_batchsize, dim=0)
        micro_targets = torch.split(targets, effective_batchsize, dim=0)
        micro_flags = [flags[i:i+effective_batchsize] for i in range(0, len(flags), effective_batchsize)]

        for mb_img, mb_tgt, mb_flg in zip(micro_batches, micro_targets, micro_flags):
            pred = model(mb_img)

            loss = loss_fn(pred, mb_tgt.unsqueeze(1)).mean(dim=[2, 3]) + dice_loss(pred, mb_tgt)

            for i, flag in enumerate(mb_flg):
                metrics['train loss'][flag] += loss[i].item()
            
            loss.mean().backward()

        optimizer.step()
        optimizer.zero_grad()
        
    if epoch % 10 == 0:
        datasetus.eval()
        model.eval()
        val_loader = my_utils.ThreadedLoader(datasetus, max_queue_size=5).start()

        for idx, batch in enumerate(val_loader):
            images = batch["image"].to(device).unsqueeze(1)    # [B, С, H, W]
            targets = batch["target"].float().to(device) # [B, H, W]
            flags = batch["flag"]

            micro_batches = torch.split(images, effective_batchsize, dim=0)
            micro_targets = torch.split(targets, effective_batchsize, dim=0)
            micro_flags = [flags[i:i+effective_batchsize] for i in range(0, len(flags), effective_batchsize)]

            preds = []
            for mb_img, mb_tgt, mb_flg in zip(micro_batches, micro_targets, micro_flags):
                with torch.no_grad():
                    pred = model(mb_img)
                preds.append(pred.cpu().squeeze())

                loss = loss_fn(pred, mb_tgt.unsqueeze(1)).mean(dim=[2, 3]) + dice_loss(pred, mb_tgt)
                IoU = my_utils.compute_binary_iou(pred, mb_tgt, reduction='none')
                prauc = my_utils.pr_auc_score(pred, mb_tgt, reduction='none')
                f1, precision, recall = my_utils.metrix(pred, mb_tgt, reduction='none')
                
                for i, flag in enumerate(mb_flg):
                    metrics['validation loss'][flag] += loss[i].item()
                    metrics['iou'][flag] += IoU[i].item()
                    metrics['prauc'][flag] += prauc[i].item()
                    metrics['f1'][flag] += f1[i].item()
                    metrics['precision'][flag] += precision[i].item()
                    metrics['recall'][flag] += recall[i].item()
            
            preds = torch.cat(preds, dim = 0)

        for metric in list(metrics.keys())[1:]:
            for dataset in metrics[metric]: 
                    metrics[metric][dataset] /= len(val_loader)

        for metric in list(metrics.keys())[1:]:
            writer.add_scalar(f'mean Metrics/{metric}/mean', stat.mean(metrics[metric].values()), epoch)
            for dataset in metrics[metric]:
                writer.add_scalar(f'dataset Metrics/{metric}/{dataset}', metrics[metric][dataset], epoch)


    for dataset in metrics['train loss']:
        metrics['train loss'][dataset] /= args.train_ex
        writer.add_scalar(f'dataset Metrics/train loss/{dataset}', metrics['train loss'][dataset], epoch)
    writer.add_scalar(f'mean Metrics/train loss/mean', stat.mean(metrics['train loss'].values()), epoch)

    if epoch % args.save_every == 0:
        os.makedirs(args.save_dir, exist_ok=True)

        torch.save(
            model,
            os.path.join(
                args.save_dir,
                f"{args.encoder}_{epoch}.pth"
            )
        )

    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        plateau_scheduler.step(stat.mean(metrics['train loss'].values()))