# -*- coding: utf-8 -*-
"""
Dataset module for FNO-based Approximate GKP State Recovery.

Provides both offline (pre-generated) and online (on-the-fly) data generation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle


class GKPDataset(Dataset):
    """
    PyTorch Dataset for GKP state data.

    Supports both pre-generated data and on-the-fly generation.
    """

    def __init__(
        self,
        simulator=None,
        n_samples: int = 1000,
        noise_sigma: float = 0.15,
        noise_type: str = 'displacement',
        delta_range: Optional[Tuple[float, float]] = None,
        online: bool = False,
        return_clean: bool = False,
        data_path: Optional[str] = None,
        transform=None
    ):
        """
        Initialize the dataset.

        Args:
            simulator: ApproxGKPSimulator instance (required for generation)
            n_samples: Number of samples to generate
            noise_sigma: Noise standard deviation
            noise_type: Type of noise ('displacement', 'loss', 'combined')
            delta_range: (min, max) for delta sampling, None for fixed delta
            online: If True, generate samples on-the-fly
            return_clean: Whether to include clean Wigner functions
            data_path: Path to load/save pre-generated data
            transform: Optional transform to apply to samples
        """
        self.simulator = simulator
        self.n_samples = n_samples
        self.noise_sigma = noise_sigma
        self.noise_type = noise_type
        self.delta_range = delta_range
        self.online = online
        self.return_clean = return_clean
        self.transform = transform

        # Storage for offline data
        self.wigners = None
        self.displacements = None
        self.deltas = None
        self.wigners_clean = None

        if not online:
            if data_path and os.path.exists(data_path):
                self._load_data(data_path)
            elif simulator is not None:
                self._generate_offline_data()
                if data_path:
                    self._save_data(data_path)
            else:
                raise ValueError(
                    "Either provide simulator for generation or data_path for loading"
                )

    def _generate_offline_data(self):
        """Generate data offline for faster training."""
        print(f"Generating {self.n_samples} samples offline...")

        batch = self.simulator.generate_batch(
            batch_size=self.n_samples,
            noise_sigma=self.noise_sigma,
            noise_type=self.noise_type,
            delta_range=self.delta_range,
            return_clean=self.return_clean
        )

        self.wigners = batch['wigner']
        self.displacements = batch['displacement']
        self.deltas = batch['delta']

        if self.return_clean and 'wigner_clean' in batch:
            self.wigners_clean = batch['wigner_clean']

        print(f"Data generation complete. Shape: {self.wigners.shape}")

    def _save_data(self, path: str):
        """Save generated data to disk."""
        data = {
            'wigners': self.wigners,
            'displacements': self.displacements,
            'deltas': self.deltas,
            'wigners_clean': self.wigners_clean,
            'noise_sigma': self.noise_sigma,
            'noise_type': self.noise_type,
            'delta_range': self.delta_range
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {path}")

    def _load_data(self, path: str):
        """Load pre-generated data from disk."""
        print(f"Loading data from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.wigners = data['wigners']
        self.displacements = data['displacements']
        self.deltas = data['deltas']
        self.wigners_clean = data.get('wigners_clean', None)
        self.n_samples = len(self.wigners)

        print(f"Loaded {self.n_samples} samples")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with 'wigner', 'displacement', 'delta',
            and optionally 'wigner_clean'
        """
        if self.online:
            # Generate sample on-the-fly
            delta = None
            if self.delta_range is not None:
                delta = np.random.uniform(*self.delta_range)

            sample = self.simulator.generate_sample(
                noise_sigma=self.noise_sigma,
                noise_type=self.noise_type,
                delta=delta
            )

            result = {
                'wigner': torch.tensor(
                    sample.wigner[np.newaxis, :, :], dtype=torch.float32
                ),
                'displacement': torch.tensor(
                    sample.displacement, dtype=torch.float32
                ),
                'delta': torch.tensor([sample.delta], dtype=torch.float32)
            }
        else:
            result = {
                'wigner': torch.tensor(self.wigners[idx], dtype=torch.float32),
                'displacement': torch.tensor(
                    self.displacements[idx], dtype=torch.float32
                ),
                'delta': torch.tensor([self.deltas[idx]], dtype=torch.float32)
            }

            if self.wigners_clean is not None:
                result['wigner_clean'] = torch.tensor(
                    self.wigners_clean[idx], dtype=torch.float32
                )

        if self.transform:
            result = self.transform(result)

        return result


class OnlineGKPDataset(Dataset):
    """
    Dataset that generates new samples each epoch.

    More memory efficient for large datasets and prevents overfitting.
    """

    def __init__(
        self,
        simulator,
        n_samples: int = 1000,
        noise_sigma: float = 0.15,
        noise_type: str = 'displacement',
        delta_range: Optional[Tuple[float, float]] = None,
        return_clean: bool = False
    ):
        self.simulator = simulator
        self.n_samples = n_samples
        self.noise_sigma = noise_sigma
        self.noise_type = noise_type
        self.delta_range = delta_range
        self.return_clean = return_clean

        # Pre-compute some states for efficiency
        self._cache = None
        self._cache_idx = 0
        self._cache_size = min(100, n_samples)

    def _refill_cache(self):
        """Refill the sample cache."""
        self._cache = self.simulator.generate_batch(
            batch_size=self._cache_size,
            noise_sigma=self.noise_sigma,
            noise_type=self.noise_type,
            delta_range=self.delta_range,
            return_clean=self.return_clean
        )
        self._cache_idx = 0

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Refill cache if needed
        if self._cache is None or self._cache_idx >= self._cache_size:
            self._refill_cache()

        i = self._cache_idx
        self._cache_idx += 1

        result = {
            'wigner': torch.tensor(self._cache['wigner'][i], dtype=torch.float32),
            'displacement': torch.tensor(
                self._cache['displacement'][i], dtype=torch.float32
            ),
            'delta': torch.tensor([self._cache['delta'][i]], dtype=torch.float32)
        }

        if self.return_clean and 'wigner_clean' in self._cache:
            result['wigner_clean'] = torch.tensor(
                self._cache['wigner_clean'][i], dtype=torch.float32
            )

        return result


def create_dataloaders(
    simulator,
    train_samples: int = 5000,
    val_samples: int = 500,
    test_samples: int = 200,
    batch_size: int = 32,
    noise_sigma: float = 0.15,
    noise_type: str = 'displacement',
    delta_range: Optional[Tuple[float, float]] = None,
    online: bool = False,
    return_clean: bool = False,
    num_workers: int = 0,
    data_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        simulator: ApproxGKPSimulator instance
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        batch_size: Batch size
        noise_sigma: Noise standard deviation
        noise_type: Type of noise channel
        delta_range: Range for delta sampling
        online: Whether to generate data online
        return_clean: Whether to return clean Wigner functions
        num_workers: Number of dataloader workers
        data_dir: Directory for saving/loading data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Data paths
    train_path = os.path.join(data_dir, 'train.pkl') if data_dir else None
    val_path = os.path.join(data_dir, 'val.pkl') if data_dir else None
    test_path = os.path.join(data_dir, 'test.pkl') if data_dir else None

    if online:
        train_dataset = OnlineGKPDataset(
            simulator, train_samples, noise_sigma, noise_type,
            delta_range, return_clean
        )
        val_dataset = OnlineGKPDataset(
            simulator, val_samples, noise_sigma, noise_type,
            delta_range, return_clean
        )
        test_dataset = OnlineGKPDataset(
            simulator, test_samples, noise_sigma, noise_type,
            delta_range, return_clean
        )
    else:
        train_dataset = GKPDataset(
            simulator, train_samples, noise_sigma, noise_type,
            delta_range, online=False, return_clean=return_clean,
            data_path=train_path
        )
        val_dataset = GKPDataset(
            simulator, val_samples, noise_sigma, noise_type,
            delta_range, online=False, return_clean=return_clean,
            data_path=val_path
        )
        test_dataset = GKPDataset(
            simulator, test_samples, noise_sigma, noise_type,
            delta_range, online=False, return_clean=return_clean,
            data_path=test_path
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


class DataNormalizer:
    """Normalize Wigner function data for better training."""

    def __init__(self, method: str = 'standard'):
        """
        Args:
            method: 'standard' (zero mean, unit std) or 'minmax' ([0, 1])
        """
        self.method = method
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None

    def fit(self, data: np.ndarray):
        """Fit normalizer to data."""
        if self.method == 'standard':
            self.mean = data.mean()
            self.std = data.std() + 1e-8
        elif self.method == 'minmax':
            self.min_val = data.min()
            self.max_val = data.max()

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data."""
        if self.method == 'standard':
            return (data - self.mean) / self.std
        elif self.method == 'minmax':
            return (data - self.min_val) / (self.max_val - self.min_val + 1e-8)
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform."""
        if self.method == 'standard':
            return data * self.std + self.mean
        elif self.method == 'minmax':
            return data * (self.max_val - self.min_val) + self.min_val
        return data


if __name__ == '__main__':
    # Test dataset
    print("Testing GKP Dataset...")

    from physics_simulator import ApproxGKPSimulator

    # Create simulator
    sim = ApproxGKPSimulator(n_hilbert=40, delta=0.3, grid_size=32)

    # Test offline dataset
    print("\n1. Testing offline dataset:")
    dataset = GKPDataset(
        simulator=sim,
        n_samples=10,
        noise_sigma=0.15,
        online=False
    )
    sample = dataset[0]
    print(f"   Wigner shape: {sample['wigner'].shape}")
    print(f"   Displacement: {sample['displacement']}")
    print(f"   Delta: {sample['delta']}")

    # Test online dataset
    print("\n2. Testing online dataset:")
    online_dataset = OnlineGKPDataset(
        simulator=sim,
        n_samples=10,
        noise_sigma=0.15
    )
    sample = online_dataset[0]
    print(f"   Wigner shape: {sample['wigner'].shape}")
    print(f"   Displacement: {sample['displacement']}")

    # Test dataloader
    print("\n3. Testing dataloader:")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"   Batch wigner shape: {batch['wigner'].shape}")
    print(f"   Batch displacement shape: {batch['displacement'].shape}")

    print("\nDataset tests passed!")
