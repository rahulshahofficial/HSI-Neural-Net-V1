import torch
import torch.nn as nn
from config import config

class FullImageHyperspectralNet(nn.Module):
    def __init__(self):
        super(FullImageHyperspectralNet, self).__init__()

        # Define network parameters
        self.input_channels = 1  # Single channel input (intensity measurements)
        self.num_wavelengths = config.num_wavelengths

        # Encoder path with larger receptive field
        self.encoder = nn.Sequential(
            # First block
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        # Decoder path with skip connections
        self.decoder = nn.Sequential(
            # First block
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            # Second block
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Final reconstruction layer
            nn.Conv2d(64, self.num_wavelengths, kernel_size=3, padding=1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x: Input tensor of shape (batch_size, 1, height, width)
        Returns:
            Reconstructed hyperspectral image of shape (batch_size, num_wavelengths, height, width)
        """
        # Encoder path
        x = self.encoder(x)

        # Decoder path
        x = self.decoder(x)

        # Apply non-negativity constraint
        x = torch.relu(x)

        # Normalize the output while maintaining relative spectral relationships
        # Add small epsilon to prevent division by zero
        x_max = x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x = x / (x_max + 1e-8)

        return x

    def compute_loss(self, outputs, targets, criterion):
        """
        Compute the loss between outputs and targets.
        Args:
            outputs: Predicted hyperspectral images
            targets: Ground truth hyperspectral images
            criterion: Loss function
        Returns:
            Total loss value
        """
        # Reconstruction loss
        recon_loss = criterion(outputs, targets)

        # You could add additional loss terms here if needed
        # For example: spectral smoothness, spatial consistency, etc.

        return recon_loss
