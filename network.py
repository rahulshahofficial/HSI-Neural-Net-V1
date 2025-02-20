import torch
import torch.nn as nn
from config import config

class FullImageHyperspectralNet(nn.Module):
    def __init__(self):
        super(FullImageHyperspectralNet, self).__init__()

        self.input_channels = 1
        self.num_wavelengths = config.num_output_wavelengths  # Only wavelengths in 800-1700nm range

        # Encoder with saved intermediate outputs for skip connections
        self.encoder1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        # Decoder with skip connections
        self.decoder1 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),  # Skip connection
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),  # Skip connection
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.final_layer = nn.Conv2d(64, self.num_wavelengths, kernel_size=3, padding=1)

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
        Forward pass with skip connections.
        Args:
            x: Input tensor of shape (batch_size, 1, height, width)
        Returns:
            Reconstructed hyperspectral image (batch_size, num_wavelengths, height, width)
        """
        # Encoder forward pass
        e1 = self.encoder1(x)  # First feature map
        e2 = self.encoder2(e1)  # Second feature map
        e3 = self.encoder3(e2)  # Third feature map (latent representation)

        # Decoder with skip connections
        d1 = self.decoder1(torch.cat([e3, e2], dim=1))  # Concatenating skip connection from e2
        d2 = self.decoder2(torch.cat([d1, e1], dim=1))  # Concatenating skip connection from e1

        x = self.final_layer(d2)  # Output reconstruction

        # Apply non-negativity constraint
        x = torch.relu(x)

        # Normalize the output to maintain relative spectral relationships
        x_max = x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x = x / (x_max + 1e-8)

        return x

    def compute_loss(self, outputs, targets, criterion):
        """
        Compute total loss including reconstruction, spectral smoothness, and spatial consistency.

        Args:
            outputs: Predicted hyperspectral images
            targets: Ground truth hyperspectral images
            criterion: Reconstruction loss function (e.g., MSELoss)

        Returns:
            Total loss value
        """
        # Reconstruction loss (Mean Squared Error)
        recon_loss = criterion(outputs, targets)

        # Spectral Smoothness Loss: Penalizes large spectral variations
        spectral_diff = outputs[:, 1:, :, :] - outputs[:, :-1, :, :]
        spectral_smoothness_loss = torch.mean(spectral_diff ** 2)

        # Spatial Consistency Loss: Encourages smooth changes between adjacent pixels
        dx = outputs[:, :, 1:, :] - outputs[:, :, :-1, :]
        dy = outputs[:, :, :, 1:] - outputs[:, :, :, :-1]
        spatial_consistency_loss = torch.mean(dx ** 2) + torch.mean(dy ** 2)

        # Weighted Sum of Losses
        total_loss = recon_loss + 0.1 * spatial_consistency_loss # + 0.1 * spectral_smoothness_loss

        return total_loss
