# -*- coding: utf-8 -*-
"""
Configuration for FNO-based Approximate GKP State Recovery.

Combines insights from Gemini and Doubao proposals for optimal performance.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch


@dataclass
class PhysicsConfig:
    """Configuration for GKP physics simulation."""

    # Hilbert space truncation dimension
    n_hilbert: int = 50

    # Squeezing parameter Delta (smaller = closer to ideal GKP)
    # Typical experimental range: 0.2-0.4 (corresponding to ~10-14 dB)
    delta: float = 0.3

    # Delta range for training data augmentation
    delta_range: Tuple[float, float] = None  # Set to (0.2, 0.4) for augmentation

    # Grid size for Wigner function computation
    grid_size: int = 64

    # Phase space extent for Wigner function
    phase_space_extent: float = 6.0

    # Noise parameters
    noise_sigma: float = 0.15  # Gaussian displacement noise std

    # Loss channel parameters (for realistic noise)
    kappa: float = 0.01       # Photon loss rate
    kappa_phi: float = 0.005  # Dephasing rate


@dataclass
class ModelConfig:
    """Configuration for FNO model architecture."""

    # Input channels (Wigner function is single channel)
    in_channels: int = 1

    # Output channels for state reconstruction mode
    out_channels: int = 1

    # Hidden channel width
    width: int = 32

    # Fourier modes to keep (important for GKP lattice structure)
    # Higher modes preserve fine interference fringes
    modes: int = 16

    # Number of Fourier layers
    n_layers: int = 4

    # Output mode: 'displacement' (2D vector) or 'reconstruction' (Wigner function)
    output_mode: str = 'displacement'

    # Activation function
    activation: str = 'gelu'

    # Whether to use residual connections
    use_residual: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Training parameters
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Learning rate scheduler
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    # Loss weights (for combined loss)
    lambda_mse: float = 1.0
    lambda_fidelity: float = 0.5
    lambda_stabilizer: float = 0.1

    # Data generation (reduced for faster iteration)
    train_samples: int = 5000
    val_samples: int = 500
    test_samples: int = 500

    # Online learning (generate new data each epoch)
    # Set to False for faster training with pre-generated data
    online_learning: bool = False

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_every: int = 20

    # Early stopping
    early_stopping_patience: int = 30

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""

    # Metrics to compute
    compute_fidelity: bool = True
    compute_squeezing_error: bool = True
    compute_logical_error: bool = True

    # Baseline comparisons
    compare_homodyne_binning: bool = True
    compare_ideal_phase_estimation: bool = False

    # Visualization
    plot_wigner: bool = True
    plot_error_curves: bool = True


@dataclass
class Config:
    """Main configuration combining all sub-configs."""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Experiment name
    exp_name: str = 'fno_approx_gkp'


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_high_squeezing_config() -> Config:
    """Configuration for high squeezing regime (delta=0.2, ~14dB)."""
    config = Config()
    config.physics.delta = 0.2
    config.physics.delta_range = (0.15, 0.25)
    config.exp_name = 'fno_approx_gkp_high_squeezing'
    return config


def get_low_squeezing_config() -> Config:
    """Configuration for low squeezing regime (delta=0.4, ~8dB)."""
    config = Config()
    config.physics.delta = 0.4
    config.physics.delta_range = (0.35, 0.45)
    config.model.modes = 12  # Lower modes for broader peaks
    config.exp_name = 'fno_approx_gkp_low_squeezing'
    return config


def get_reconstruction_config() -> Config:
    """Configuration for state reconstruction mode."""
    config = Config()
    config.model.output_mode = 'reconstruction'
    config.model.out_channels = 1
    config.training.lambda_fidelity = 1.0
    config.training.lambda_stabilizer = 0.2
    config.exp_name = 'fno_approx_gkp_reconstruction'
    return config
