import numpy as np
import torch
import torch.nn as nn
import lpips


from skimage.color import rgb2lab, deltaE_ciede2000

class ChromaLoss(nn.Module):
    def __init__(self):
        super(ChromaLoss, self).__init__()
        
    def forward(self, img1, img2):
        """
        Compute CIEDE2000 color difference between two RGB images
        img1, img2: [B, C, H, W] tensors in range [0, 1]
        """
        batch_size = img1.shape[0]
        
        # Convert to numpy for skimage processing
        img1_np = img1.permute(0, 2, 3, 1).cpu().detach().numpy() * 255.0
        img2_np = img2.permute(0, 2, 3, 1).cpu().detach().numpy() * 255.0
        
        total_delta = 0.0
        
        for i in range(batch_size):
            # Convert RGB to LAB color space
            lab1 = rgb2lab(img1_np[i].astype(np.uint8))
            lab2 = rgb2lab(img2_np[i].astype(np.uint8))
            
            # Compute CIEDE2000 delta
            delta = deltaE_ciede2000(lab1, lab2)
            
            # Average over spatial dimensions
            total_delta += np.mean(delta)
        
        # Return average delta across batch
        return total_delta / batch_size

class L1Chromo(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super(L1Chromo, self).__init__()
        self.alpha = alpha  # Weight for chroma loss
        self.l1 = nn.L1Loss(reduction='mean')
        self.chroma = ChromaLoss()
        
    def forward(self, x, y):
        l1_loss = self.l1(x, y)
        chroma_loss = self.chroma(x, y)
        loss = l1_loss + self.alpha * chroma_loss
        return loss



class Lasso(nn.Module):
    def __init__(self, p: float = 1.0):
        super(Lasso, self).__init__()
        self.p = p
        self.l1 = nn.L1Loss(reduction='mean')
        self.l2 = nn.MSELoss(reduction='mean')

    def forward(self, x, y):
        l1_loss = self.l1(x, y)
        l2_loss = self.l2(x, y)
        loss = l1_loss + self.p * l2_loss
        return loss

class LassoLPIPS(nn.Module):
    def __init__(self, p: float = 1.0, lpips_weight: float = 0.1, lpips_net: str = 'vgg'):
        super(LassoLPIPS, self).__init__()
        self.p = p
        self.lpips_weight = lpips_weight
        self.l1 = nn.L1Loss(reduction='mean')
        self.l2 = nn.MSELoss(reduction='mean')
        
        try:
            self.lpips = lpips.LPIPS(net=lpips_net, verbose=False)
        except ImportError:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "lpips"])
            self.lpips = lpips.LPIPS(net=lpips_net, verbose=False)
        
        self.lpips.eval()
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        l1_loss = self.l1(x, y)
        l2_loss = self.l2(x, y)
        
        if self.lpips.device != x.device:
            self.lpips = self.lpips.to(x.device)
        
        x_norm = 2.0 * x - 1.0 if x.min() >= 0 and x.max() <= 1 else x
        y_norm = 2.0 * y - 1.0 if y.min() >= 0 and y.max() <= 1 else y
        lpips_loss = self.lpips(x_norm, y_norm).mean()
        
        total_loss = l1_loss + self.p * l2_loss + self.lpips_weight * lpips_loss
        return total_loss
