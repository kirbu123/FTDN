import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import NAFNet
from loss import Lasso, LassoLPIPS
from dataset import RAWDataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import os



def train_epoch(model, loss_fn, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    # Progress bar for training
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} Train')
    for batch_idx, batch in enumerate(pbar):
        raw = batch['raw'].to(device)
        srgb = batch['srgb'].to(device)
        
        optimizer.zero_grad()
        pred = model(raw)
        loss = loss_fn(pred, srgb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)



@torch.no_grad()
def validate_epoch(model, loss_fn, dataloader, device, psnr_metric, ssim_metric):
    model.eval()
    psnr_metric.reset()
    ssim_metric.reset()
    total_val_loss = 0
    
    # Progress bar for validation
    pbar = tqdm(dataloader, desc='Validation')
    for batch in pbar:
        raw = batch['raw'].to(device)
        srgb = batch['srgb'].to(device)
        
        pred = model(raw)
        val_loss = loss_fn(pred, srgb)
        total_val_loss += val_loss.item()
        
        psnr_metric.update(pred, srgb)
        ssim_metric.update(pred, srgb)
        pbar.set_postfix({'Loss': f'{val_loss.item():.4f}', 'PSNR': f'{psnr_metric.compute().item():.2f}'})
    
    avg_val_loss = total_val_loss / len(dataloader)
    avg_psnr = psnr_metric.compute().item()
    avg_ssim = ssim_metric.compute().item()
    return avg_val_loss, avg_psnr, avg_ssim



def train_pipeline(
    dataset: list | str,
    batch_size: int,
    num_epochs: int,
    lr: float,
    device: str,
    out_dir: str,
    name: str,
    ckpt_gap: int,
    aug_p: float,
    train_part: float,
    patch_size: int,
    max_infer: int,
    use_distr_transform: bool,
    loss_fn
  ):
    # Create directories
    log_dir = os.path.join(out_dir, 'runs', name)
    checkpoint_dir = os.path.join(out_dir, 'checkpoints', name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir)

    # Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Data - train/val split
    full_dataset = RAWDataset(dataset, patch_size=patch_size, aug_p=aug_p)
    train_size = int(train_part * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    # Model
    model = NAFNet(
              in_channel=4, out_channel=3, width=16, middle_blk_num=1, 
              enc_blk_nums=[1, 1], dec_blk_nums=[1, 1],
              use_distr_transform=use_distr_transform
            ).to(device)

    # Training setup
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"Training on {device}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    best_val_loss = float('inf')
    best_epoch = 0

    # Get fixed validation samples for inference logging (no augmentation)
    val_dataset_no_aug = RAWDataset(dataset, patch_size=patch_size, aug_p=0.0)
    infer_loader = DataLoader(val_dataset_no_aug, batch_size=1, shuffle=True, num_workers=2)

    for epoch in range(num_epochs):
        # Train
        avg_train_loss = train_epoch(model, loss_fn, train_loader, optimizer, device, epoch+1)
        scheduler.step()
        
        # Validate
        avg_val_loss, avg_psnr, avg_ssim = validate_epoch(model, loss_fn, val_loader, device, psnr_metric, ssim_metric)

        # TensorBoard logging
        writer.add_scalar('Loss/Train', avg_train_loss, epoch+1)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch+1)
        writer.add_scalar('PSNR/Val', avg_psnr, epoch+1)
        writer.add_scalar('SSIM/Val', avg_ssim, epoch+1)
        writer.add_scalar('LR', optimizer.param_groups[0]["lr"], epoch+1)
        
        # Check for best model (by val loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best.pth')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
            }, best_checkpoint_path)
            print(f'New best model saved! Val Loss: {avg_val_loss:.4f} (Epoch {epoch+1})')

        # Epoch summary
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | '
              f'PSNR: {avg_psnr:.2f}dB | SSIM: {avg_ssim:.4f} | LR: {optimizer.param_groups[0]["lr"]:.2e} | '
              f'Best: {best_val_loss:.4f} (E{best_epoch})')

        # Save regular checkpoint + inference images every ckpt_gap
        if (epoch) % ckpt_gap == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
            }, checkpoint_path)

            # Log inference images to TensorBoard (max_infer samples)
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(infer_loader):
                    if i >= max_infer:
                        break
                    raw = batch['raw'].to(device)
                    srgb = batch['srgb'].to(device)
                    pred = model(raw)

                    comp_image = torch.cat([srgb, pred], dim=3)

                    writer.add_images(f'Infer/{i}', comp_image[0], epoch, dataformats='CHW')

    
    writer.close()
    print(f"Training complete! Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    return model



if __name__ == "__main__":
    train_pipeline(
        dataset=[
            "/home/buka2004/RAW-DEALL/data/train/iphone-x",
            "/home/buka2004/RAW-DEALL/data/train/lq-iphone",
            "/home/buka2004/RAW-DEALL/data/train/lq-samsung",
            "/home/buka2004/RAW-DEALL/data/train/samsung-s9"
        ],
        batch_size=4,
        num_epochs=200,
        lr=3e-4,
        device='cuda:6',
        out_dir='/home/buka2004/RAW-DEALL/results',
        name='FTDN',
        ckpt_gap=20,
        aug_p=0.5,
        train_part=0.8,
        patch_size=1024,
        max_infer=4,
        use_distr_transform=True,
        loss_fn=Lasso()
    )
