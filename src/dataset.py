import torch
import numpy as np
import random
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import gaussian_blur



class RAWDataset(Dataset):
    def __init__(
            self, 
            dataset_paths,  # Changed to list of paths
            apply_noise=True,
            patch_size=256,
            aug_p = 0.0
        ):
        self.dataset_paths = dataset_paths if isinstance(dataset_paths, list) else [dataset_paths]
        self.files = self._get_file_pairs()
        self.apply_noise = apply_noise
        self.patch_size = patch_size
        self.aug_p = aug_p


    def _poisson_noise(self, img_tensor, max_lam=0.1, eps=1e-8):
        """Add Poisson noise to tensor [0,1]"""
        lam = np.random.uniform(0, max_lam)
        noise = torch.poisson(lam * torch.ones_like(img_tensor))
        noisy = img_tensor + noise / (lam + eps)
        return torch.clamp(noisy, 0, 1)


    def _gaussian_noise(self, img_tensor, max_sigma=0.02):
        """Add Gaussian noise (mean=0) to tensor [0,1]"""
        sigma = np.random.uniform(0, max_sigma)
        noise = torch.randn_like(img_tensor) * sigma
        noisy = img_tensor + noise
        return torch.clamp(noisy, 0, 1)


    def _get_file_pairs(self):
        """Get pairs of .npy (RAW) and .png files with same base name from all paths"""
        all_pairs = []
        
        for dataset_path in self.dataset_paths:
            if not os.path.exists(dataset_path):
                print(f"Warning: Path {dataset_path} does not exist, skipping...")
                continue
                
            all_files = os.listdir(dataset_path)
            npy_bases = set()
            png_bases = set()
            
            for f in all_files:
                base = os.path.splitext(f)[0]
                if f.endswith('.npy'):
                    npy_bases.add(base)
                elif f.endswith('.png'):
                    png_bases.add(base)
            
            # Find matching pairs for this dataset
            dataset_pairs = npy_bases.intersection(png_bases)
            for base in dataset_pairs:
                all_pairs.append((dataset_path, base))  # Store (path, base_name)
        
        return all_pairs
    
    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        dataset_path, base_name = self.files[idx]
        npy_path = os.path.join(dataset_path, f"{base_name}.npy")
        srgb_path = os.path.join(dataset_path, f"{base_name}.png")
                
        raw = np.load(npy_path)
        max_val = 2**12 - 1
        raw = (raw / max_val).astype(np.float32)
        raw_tensor = torch.from_numpy(raw).permute(2, 0, 1).float()

        srgb_img = Image.open(srgb_path).convert('RGB')
        srgb_tensor = torch.from_numpy(np.array(srgb_img)).permute(2, 0, 1).float() / 255.0

        _, h, w = raw_tensor.shape
        patch_h = min(self.patch_size, h)
        patch_w = min(self.patch_size, w)

        start_h = np.random.randint(0, max(1, h - patch_h + 1))
        start_w = np.random.randint(0, max(1, w - patch_w + 1))
        
        raw_patch = raw_tensor[:, start_h//2:(start_h+patch_h)//2, start_w//2:(start_w+patch_w)//2]
        srgb_patch = srgb_tensor[:, start_h:start_h+patch_h, start_w:start_w+patch_w]

        if self.apply_noise and np.random.random() < self.aug_p:
            # Blur augmentation
            raw_patch = gaussian_blur(raw_patch, kernel_size=random.choice([3, 5]), sigma=np.random.uniform(0.5, 1.5))
            # Noise augmentation
            raw_patch = self._poisson_noise(raw_patch, max_lam=0.1)
            raw_patch = self._gaussian_noise(raw_patch, max_sigma=0.02)
        
        return {
            'raw': raw_patch,
            'srgb': srgb_patch,
            'name': base_name,
            'dataset': dataset_path,  # Track which dataset it came from
            'srgb_path': srgb_path,
            'patch_pos': (start_h, start_w)
        }




if __name__ == "__main__":
    # Single path (backward compatible)
    dataset = RAWDataset("/home/buka2004/RAW-DEALL/data/train/iphone-x")
    
    # Multiple paths
    paths = [
        "/home/buka2004/RAW-DEALL/data/train/iphone-x",
        "/home/buka2004/RAW-DEALL/data/train/samsung-s9"
    ]
    dataset_multi = RAWDataset(paths)
    
    print(f"Total samples: {len(dataset_multi)}")
    raw, srgb = dataset_multi[0]['raw'], dataset_multi[0]['srgb']
    print(raw.shape, srgb.shape)
    print(f"From dataset: {dataset_multi[0]['dataset']}")
