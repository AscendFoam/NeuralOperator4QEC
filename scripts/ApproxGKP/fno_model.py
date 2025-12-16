# -*- coding: utf-8 -*-
"""
Fourier Neural Operator (FNO) for Approximate GKP State Recovery.

Architecture design combines:
- Gemini's SpectralConv2d with proper frequency handling
- Doubao's multi-modal feature fusion concept
- Support for both displacement prediction and state reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer - core of FNO.

    Performs convolution in Fourier space by learning weights for
    frequency modes. This is particularly suited for GKP states
    which have periodic lattice structure.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int
    ):
        """
        Args:
            in_channels: Input channel dimension
            out_channels: Output channel dimension
            modes1: Number of Fourier modes for first spatial dimension
            modes2: Number of Fourier modes for second spatial dimension
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Scale factor for initialization
        self.scale = 1 / (in_channels * out_channels)

        # Complex weight tensors for Fourier modes
        # Two sets for handling conjugate symmetry in real FFT
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )

    def compl_mul2d(
        self,
        input: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Complex multiplication in frequency domain."""
        # (batch, in_ch, x, y), (in_ch, out_ch, x, y) -> (batch, out_ch, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spectral convolution.

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Output tensor (batch, out_channels, height, width)
        """
        batch_size = x.shape[0]
        h, w = x.shape[-2], x.shape[-1]

        # Transform to Fourier space (real FFT for efficiency)
        x_ft = torch.fft.rfft2(x)

        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device
        )

        # Multiply relevant Fourier modes with learned weights
        # Top-left corner (low frequencies)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )

        # Bottom-left corner (due to FFT symmetry for real signals)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2
        )

        # Transform back to physical space
        x = torch.fft.irfft2(out_ft, s=(h, w))

        return x


class FNOBlock(nn.Module):
    """
    Single FNO block with spectral convolution and residual connection.
    """

    def __init__(
        self,
        width: int,
        modes: int,
        activation: str = 'gelu',
        use_residual: bool = True
    ):
        super().__init__()

        self.spectral_conv = SpectralConv2d(width, width, modes, modes)
        self.conv = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm2d(width)
        self.use_residual = use_residual

        # Activation function
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'silu':
            self.activation = F.silu
        else:
            self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FNO block."""
        # Spectral path
        x1 = self.spectral_conv(x)

        # Local convolution path (1x1 conv for channel mixing)
        x2 = self.conv(x)

        # Combine paths
        out = x1 + x2

        # Normalization and activation
        out = self.norm(out)
        out = self.activation(out)

        # Residual connection
        if self.use_residual:
            out = out + x

        return out


class FNODisplacementDecoder(nn.Module):
    """
    FNO-based decoder for predicting displacement correction.

    Input: Noisy Wigner function (B, 1, H, W)
    Output: Displacement vector (B, 2) representing (u, v) correction
    """

    def __init__(
        self,
        in_channels: int = 1,
        width: int = 32,
        modes: int = 16,
        n_layers: int = 4,
        activation: str = 'gelu',
        use_residual: bool = True
    ):
        super().__init__()

        self.width = width
        self.modes = modes

        # Lifting layer: project input to higher dimensional space
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)

        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes, activation, use_residual)
            for _ in range(n_layers)
        ])

        # Global pooling for displacement prediction
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection head for displacement output
        self.head = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Linear(width, 2)  # Output: (u, v)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Noisy Wigner function (batch, 1, height, width)

        Returns:
            Displacement prediction (batch, 2)
        """
        # Lift to higher dimension
        x = self.lift(x)

        # Pass through FNO blocks
        for block in self.fno_blocks:
            x = block(x)

        # Global average pooling
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # Predict displacement
        displacement = self.head(x)

        return displacement


class FNOStateReconstructor(nn.Module):
    """
    FNO-based state reconstructor.

    Input: Noisy Wigner function (B, 1, H, W)
    Output: Reconstructed clean Wigner function (B, 1, H, W)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        modes: int = 16,
        n_layers: int = 4,
        activation: str = 'gelu',
        use_residual: bool = True
    ):
        super().__init__()

        self.width = width

        # Lifting layer
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)

        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes, activation, use_residual)
            for _ in range(n_layers)
        ])

        # Projection layer back to output space
        self.project = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width // 2, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Noisy Wigner function (batch, 1, height, width)

        Returns:
            Reconstructed Wigner function (batch, 1, height, width)
        """
        # Lift
        x = self.lift(x)

        # FNO blocks
        for block in self.fno_blocks:
            x = block(x)

        # Project back
        x = self.project(x)

        return x


class FNOHybridModel(nn.Module):
    """
    Hybrid FNO model that can output both displacement and reconstructed state.

    This combines both approaches for potentially better training signals.
    """

    def __init__(
        self,
        in_channels: int = 1,
        width: int = 32,
        modes: int = 16,
        n_layers: int = 4,
        activation: str = 'gelu',
        use_residual: bool = True
    ):
        super().__init__()

        self.width = width

        # Shared encoder
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)

        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes, activation, use_residual)
            for _ in range(n_layers)
        ])

        # Displacement head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.displacement_head = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, 2)
        )

        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 1, kernel_size=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_both: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Noisy Wigner function (batch, 1, height, width)
            return_both: If True, return both displacement and reconstruction

        Returns:
            Tuple of (displacement, reconstruction) or just displacement
        """
        # Shared encoding
        x = self.lift(x)
        for block in self.fno_blocks:
            x = block(x)

        # Displacement prediction
        pooled = self.pool(x).view(x.size(0), -1)
        displacement = self.displacement_head(pooled)

        if return_both:
            # Reconstruction
            reconstruction = self.reconstruction_head(x)
            return displacement, reconstruction
        else:
            return displacement, None


def create_model(config) -> nn.Module:
    """
    Factory function to create model based on configuration.

    Args:
        config: ModelConfig object

    Returns:
        FNO model instance
    """
    if config.output_mode == 'displacement':
        return FNODisplacementDecoder(
            in_channels=config.in_channels,
            width=config.width,
            modes=config.modes,
            n_layers=config.n_layers,
            activation=config.activation,
            use_residual=config.use_residual
        )
    elif config.output_mode == 'reconstruction':
        return FNOStateReconstructor(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            width=config.width,
            modes=config.modes,
            n_layers=config.n_layers,
            activation=config.activation,
            use_residual=config.use_residual
        )
    elif config.output_mode == 'hybrid':
        return FNOHybridModel(
            in_channels=config.in_channels,
            width=config.width,
            modes=config.modes,
            n_layers=config.n_layers,
            activation=config.activation,
            use_residual=config.use_residual
        )
    else:
        raise ValueError(f"Unknown output mode: {config.output_mode}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    print("Testing FNO models...")

    batch_size = 4
    grid_size = 64
    x = torch.randn(batch_size, 1, grid_size, grid_size)

    # Test displacement decoder
    print("\n1. FNODisplacementDecoder:")
    model1 = FNODisplacementDecoder(width=32, modes=16, n_layers=4)
    out1 = model1(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out1.shape}")
    print(f"   Parameters: {count_parameters(model1):,}")

    # Test state reconstructor
    print("\n2. FNOStateReconstructor:")
    model2 = FNOStateReconstructor(width=32, modes=16, n_layers=4)
    out2 = model2(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out2.shape}")
    print(f"   Parameters: {count_parameters(model2):,}")

    # Test hybrid model
    print("\n3. FNOHybridModel:")
    model3 = FNOHybridModel(width=32, modes=16, n_layers=4)
    disp, recon = model3(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Displacement shape: {disp.shape}")
    print(f"   Reconstruction shape: {recon.shape}")
    print(f"   Parameters: {count_parameters(model3):,}")

    print("\nAll model tests passed!")
