"""
Section 3.2 — U-Net 2D Architecture for Liver and Tumor Segmentation

Encoder (contracting path, Sec 3.2.1): blocks of 2x[Conv3x3 -> ReLU] (Eq. 4)
followed by 2x2 max-pool stride 2 (Eq. 5). Channels double at each level:
64 -> 128 -> 256 -> 512 -> 1024.

Decoder (Sec 3.2.2): transposed convolution upsampling (Eq. 6), concatenated
with the corresponding encoder feature map via skip connections
D_i = [U_i, E_i], followed by a 1x1 conv + softmax (Eq. 7) over
C=3 classes (background, liver, tumor).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import unet_arch_cfg


class ConvBlock(nn.Module):
    """Two 3x3 convolutions each followed by ReLU (Eq. 4): f(x) = max(0, W*x + b)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock followed by 2x2 max-pool stride 2 (Eq. 5)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """Transposed conv upsampling (Eq. 6) + skip concat D_i = [U_i, E_i] + ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # handle any off-by-one spatial mismatch from odd input sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """Full U-Net 2D model, channel progression 64 -> 128 -> 256 -> 512 -> 1024.

    Output: per-pixel class logits over C classes (background, liver, tumor);
    softmax (Eq. 7) is applied via CrossEntropy/loss functions at train time,
    or explicitly in `predict_proba` for inference.
    """

    def __init__(self, in_channels: int = None, base_channels: int = None,
                 num_classes: int = None):
        super().__init__()
        in_ch = unet_arch_cfg.in_channels if in_channels is None else in_channels
        bc = unet_arch_cfg.base_channels if base_channels is None else base_channels
        n_cls = unet_arch_cfg.num_classes if num_classes is None else num_classes

        # Encoder: 64 -> 128 -> 256 -> 512
        self.enc1 = EncoderBlock(in_ch, bc)          # 64
        self.enc2 = EncoderBlock(bc, bc * 2)         # 128
        self.enc3 = EncoderBlock(bc * 2, bc * 4)     # 256
        self.enc4 = EncoderBlock(bc * 4, bc * 8)     # 512

        # Bottleneck: 1024
        self.bottleneck = ConvBlock(bc * 8, bc * 16)

        # Decoder
        self.dec4 = DecoderBlock(bc * 16, bc * 8, bc * 8)
        self.dec3 = DecoderBlock(bc * 8, bc * 4, bc * 4)
        self.dec2 = DecoderBlock(bc * 4, bc * 2, bc * 2)
        self.dec1 = DecoderBlock(bc * 2, bc, bc)

        # 1x1 conv projection to output classes
        self.out_conv = nn.Conv2d(bc, n_cls, kernel_size=1)

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        logits = self.out_conv(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        """Eq. (7): P(y=c|x) = softmax(logits)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_mask(self, x):
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=1)


if __name__ == "__main__":
    model = UNet2D()
    dummy = torch.randn(2, 1, 512, 512)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (2, 3, 512, 512)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
