import torch
import torch.nn as nn
from config import config

class HyperspectralNet(nn.Module):
    def __init__(self):
        super(HyperspectralNet, self).__init__() # Ensures that the class inherits the functionalities of nn.Module

        # Input: [64, 1, 1, 8, 8]
        self.encoder = nn.Sequential(
            # nn.Conv3d(16, 128, kernel_size=(1,3,3), padding=(0,1,1)), # Superpixel Size 4 x 4
            nn.Conv3d(64, 128, kernel_size=(1,3,3), padding=(0,1,1)), # Superpixel size 8 x 8
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.ReLU(),
            nn.BatchNorm3d(256)
        )

        # Decoder to reconstruct spectral information
        self.decoder = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 19, kernel_size=(1,3,3), padding=(0,1,1))  # 10 (800-900nm) + 9 (1100-1700nm)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # x = torch.clamp(x, 0.0, 1.0)
        x = x.squeeze(2)  # Remove extra dimensions

        # First ensure non-negativity while preserving relative relationships
        x = torch.relu(x)

        # Global normalization to maintain relative intensities
        x_max = x.max()
        if x_max > 0:
            x = x / (x_max + 1e-8)

        return x

    def add_spectral_constraint(self, output):
        return torch.clamp(output, min=0.0)
