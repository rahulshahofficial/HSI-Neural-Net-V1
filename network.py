import torch
import torch.nn as nn
from config import config

class HyperspectralNet(nn.Module):
    def __init__(self):
        super(HyperspectralNet, self).__init__()

        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

        # Decoder layers with skip connections
        self.dec1 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # Final reconstruction layer
        self.final = nn.Conv2d(64, config.num_wavelengths, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder path with skip connections
        d1 = self.dec1(torch.cat([e4, e3], dim=1))
        d2 = self.dec2(torch.cat([d1, e2], dim=1))
        d3 = self.dec3(torch.cat([d2, e1], dim=1))

        # Final reconstruction
        x = self.final(d3)

        # Apply non-negativity constraint and normalization
        x = torch.relu(x)
        x_max = x.max()
        if x_max > 0:
            x = x / (x_max + 1e-8)

        return x

    def compute_loss(self, outputs, targets, criterion):
        """Compute total loss including reconstruction and regularization terms"""
        # Reconstruction loss
        recon_loss = criterion(outputs, targets)

        # Spectral smoothness loss
        spectral_diff = outputs[:, 1:, :, :] - outputs[:, :-1, :, :]
        spectral_smoothness = torch.mean(spectral_diff ** 2)

        # Spatial consistency loss
        spatial_diff_x = outputs[:, :, 1:, :] - outputs[:, :, :-1, :]
        spatial_diff_y = outputs[:, :, :, 1:] - outputs[:, :, :, :-1]
        spatial_consistency = torch.mean(spatial_diff_x ** 2) + torch.mean(spatial_diff_y ** 2)

        # Combined loss with weights
        total_loss = recon_loss + 0.1 * spectral_smoothness + 0.1 * spatial_consistency

        return total_loss
