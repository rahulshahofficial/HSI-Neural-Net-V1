import torch
import torch.nn as nn

# Let's create a single Conv2d layer:
# - Input channels = 3 (like RGB)
# - Output channels = 8 (8 learnable filters)
# - Kernel size = 3, padding = 1 (so output has same spatial size if stride=1)

conv2d_layer = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)

# Create a random input tensor:
# Batch size = 2, Channels=3, Height=32, Width=32
# shape: [2, 3, 32, 32]
dummy_input_2d = torch.randn(2, 3, 32, 32)

# Pass it through the conv layer
output_2d = conv2d_layer(dummy_input_2d)

print("Input shape 2D:", dummy_input_2d.shape)      # [2, 3, 32, 32]
print("Output shape 2D:", output_2d.shape)          # [2, 8, 32, 32]
