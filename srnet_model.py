import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import numbers

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class Spectral_Atten(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.num_heads = heads
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x_in):
        """
        x_in: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x_in.shape
        q_in = self.q_dwconv(self.to_q(x_in))
        k_in = self.k_dwconv(self.to_k(x_in))
        v_in = self.v_dwconv(self.to_v(x_in))

        q = rearrange(q_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (q @ k.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj(out)

        return out


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class PreNorm(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.relu = nn.GELU()
        self.out_conv = nn.Conv2d(dim * 2, dim, 1, 1, bias=False)

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.net2(x)
        out = torch.cat((out1, out2), dim=1)
        return self.out_conv(self.relu(out))


class SAM_Spectral(nn.Module):
    def __init__(self, dim, heads, num_blocks):
        super().__init__()

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                LayerNorm(dim),
                Spectral_Atten(dim=dim, heads=heads),
                LayerNorm(dim),
                PreNorm(dim, mult=4)
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        for (norm1, attn, norm2, ffn) in self.blocks:
            x = attn(norm1(x)) + x
            x = ffn(norm2(x)) + x
        return x


class SpectralReconstructionNet(nn.Module):
    def __init__(self, input_channels=1, out_channels=31, dim=32, deep_stage=3,
                 num_blocks=[1, 1, 1], num_heads=[1, 2, 4]):
        super(SpectralReconstructionNet, self).__init__()
        """
        SRNet for hyperspectral reconstruction

        Args:
            input_channels: Number of input measurement channels
            out_channels: Number of spectral bands to reconstruct
            dim: Base feature dimension
            deep_stage: Number of encoder/decoder stages
            num_blocks: Number of SAM blocks at each stage
            num_heads: Number of attention heads at each stage
        """
        self.dim = dim
        self.out_channels = out_channels
        self.stage = deep_stage

        # Input embeddings - one for measurements, one for filter patterns
        self.embedding1 = nn.Conv2d(input_channels, dim, kernel_size=3, padding=1, bias=False)
        self.embedding2 = nn.Conv2d(out_channels, dim, kernel_size=3, padding=1, bias=False)
        self.embedding = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False)

        # Spatial down/up sampling operations
        self.down_sample = nn.Conv2d(dim, dim, 4, 2, 1, bias=False)
        self.up_sample = nn.ConvTranspose2d(dim, dim, stride=2, kernel_size=2, padding=0, output_padding=0)

        # Final mapping to output spectral bands
        self.mapping = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=False)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(deep_stage):
            self.encoder_layers.append(nn.ModuleList([
                SAM_Spectral(dim=dim_stage, heads=num_heads[i], num_blocks=num_blocks[i]),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = SAM_Spectral(
            dim=dim_stage, heads=num_heads[-1], num_blocks=num_blocks[-1])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([])
        for i in range(deep_stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                SAM_Spectral(dim=dim_stage // 2, heads=num_heads[deep_stage - 1 - i], num_blocks=num_blocks[deep_stage - 1 - i]),
            ]))
            dim_stage //= 2

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, filter_pattern):
        """
        Forward pass with explicit filter pattern input
        Args:
            x: Input filtered measurements [B, C, H, W]
            filter_pattern: Filter pattern tensor [B, num_wavelengths, H, W]
                           representing the spectral transmission of filters
        """
        # Process input measurements and filter pattern
        x = self.embedding1(x)
        mask = self.embedding2(filter_pattern)
        
        # Combine both feature maps
        x = torch.cat((x, mask), dim=1)
        fea = self.embedding(x)
        
        # Save initial features for residual connection
        residual = fea
        fea = self.down_sample(fea)
        
        # Encoder forward pass
        fea_encoder = []
        for (Attention, FeaDownSample) in self.encoder_layers:
            fea = Attention(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
        
        # Bottleneck
        fea = self.bottleneck(fea)
        
        # Decoder forward pass with skip connections
        for i, (FeaUpSample, Fusion, Attention) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            fea = Attention(fea)
        
        # Final upsampling and residual connection
        fea = self.up_sample(fea)
        out = fea + residual
        
        # Map to output spectral dimensions
        out = self.mapping(out)
        
        return out
    
    def compute_loss(self, outputs, targets, criterion):
        """
        Compute total loss including reconstruction, spectral smoothness, and spatial consistency.
        Compatible with the original loss function approach.

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
        total_loss = recon_loss + 0.1 * spatial_consistency_loss + 0.1 * spectral_smoothness_loss

        return total_loss
