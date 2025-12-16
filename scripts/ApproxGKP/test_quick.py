# -*- coding: utf-8 -*-
"""
Quick test script to verify the FNO GKP implementation.

Run this to check if all components work correctly.
"""

import sys
import numpy as np
import torch

def test_physics_simulator():
    """Test the physics simulator module."""
    print("\n" + "=" * 50)
    print("Testing Physics Simulator...")
    print("=" * 50)

    from physics_simulator import ApproxGKPSimulator, compute_squeezing_db

    # Create simulator with smaller parameters for quick testing
    sim = ApproxGKPSimulator(n_hilbert=30, delta=0.35, grid_size=32)

    # Test single sample generation
    sample = sim.generate_sample(noise_sigma=0.15)
    print(f"  Single sample Wigner shape: {sample.wigner.shape}")
    print(f"  Displacement: {sample.displacement}")
    print(f"  Delta: {sample.delta} ({compute_squeezing_db(sample.delta):.1f} dB)")

    # Test batch generation
    batch = sim.generate_batch(batch_size=4, noise_sigma=0.15)
    print(f"  Batch Wigner shape: {batch['wigner'].shape}")
    print(f"  Batch displacement shape: {batch['displacement'].shape}")

    print("  Physics simulator: OK")
    return sim


def test_fno_model():
    """Test the FNO model module."""
    print("\n" + "=" * 50)
    print("Testing FNO Model...")
    print("=" * 50)

    from fno_model import (
        FNODisplacementDecoder, FNOStateReconstructor,
        FNOHybridModel, count_parameters
    )

    batch_size = 2
    grid_size = 32
    x = torch.randn(batch_size, 1, grid_size, grid_size)

    # Test displacement decoder
    model1 = FNODisplacementDecoder(width=16, modes=8, n_layers=2)
    out1 = model1(x)
    print(f"  Displacement Decoder:")
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {out1.shape}")
    print(f"    Parameters: {count_parameters(model1):,}")

    # Test state reconstructor
    model2 = FNOStateReconstructor(width=16, modes=8, n_layers=2)
    out2 = model2(x)
    print(f"  State Reconstructor:")
    print(f"    Output shape: {out2.shape}")

    # Test hybrid model
    model3 = FNOHybridModel(width=16, modes=8, n_layers=2)
    disp, recon = model3(x)
    print(f"  Hybrid Model:")
    print(f"    Displacement shape: {disp.shape}")
    print(f"    Reconstruction shape: {recon.shape}")

    print("  FNO models: OK")
    return model1


def test_dataset(simulator):
    """Test the dataset module."""
    print("\n" + "=" * 50)
    print("Testing Dataset...")
    print("=" * 50)

    from dataset import GKPDataset, OnlineGKPDataset
    from torch.utils.data import DataLoader

    # Test offline dataset
    dataset = GKPDataset(
        simulator=simulator,
        n_samples=8,
        noise_sigma=0.15,
        online=False
    )
    sample = dataset[0]
    print(f"  Offline dataset sample:")
    print(f"    Wigner shape: {sample['wigner'].shape}")
    print(f"    Displacement: {sample['displacement']}")

    # Test dataloader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"  DataLoader batch:")
    print(f"    Wigner shape: {batch['wigner'].shape}")
    print(f"    Displacement shape: {batch['displacement'].shape}")

    print("  Dataset: OK")
    return loader


def test_training_step(simulator, model):
    """Test a single training step."""
    print("\n" + "=" * 50)
    print("Testing Training Step...")
    print("=" * 50)

    from dataset import GKPDataset
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim

    # Create small dataset
    dataset = GKPDataset(
        simulator=simulator,
        n_samples=8,
        noise_sigma=0.15,
        online=False
    )
    loader = DataLoader(dataset, batch_size=4)

    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Single training step
    model.train()
    batch = next(iter(loader))
    wigner = batch['wigner'].to(device)
    displacement = batch['displacement'].to(device)

    optimizer.zero_grad()
    pred = model(wigner)
    loss = criterion(pred, displacement)
    loss.backward()
    optimizer.step()

    print(f"  Training step completed")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Training: OK")


def test_evaluation():
    """Test evaluation functions."""
    print("\n" + "=" * 50)
    print("Testing Evaluation Functions...")
    print("=" * 50)

    from evaluate import homodyne_binning_decoder, compute_logical_error_rate

    # Test binning decoder
    grid_size = 32
    xvec = np.linspace(-6, 6, grid_size)
    pvec = np.linspace(-6, 6, grid_size)

    # Create a simple test Wigner (peaked at some location)
    X, P = np.meshgrid(xvec, pvec)
    wigner = np.exp(-((X - 0.5) ** 2 + (P - 0.3) ** 2))

    disp = homodyne_binning_decoder(wigner, xvec, pvec)
    print(f"  Binning decoder test:")
    print(f"    Estimated displacement: {disp}")

    # Test error rate computation
    true_disp = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    pred_disp = np.array([[0.12, 0.18], [0.35, 0.45], [0.48, 0.55]])
    error_rate = compute_logical_error_rate(true_disp, pred_disp)
    print(f"  Logical error rate: {error_rate:.4f}")

    print("  Evaluation: OK")


def run_quick_training():
    """Run a quick training test (few epochs)."""
    print("\n" + "=" * 50)
    print("Running Quick Training Test (3 epochs)...")
    print("=" * 50)

    from config import get_default_config
    from physics_simulator import ApproxGKPSimulator
    from fno_model import FNODisplacementDecoder
    from dataset import GKPDataset
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim

    # Minimal configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Create simulator (small for testing)
    print("  Creating simulator...")
    sim = ApproxGKPSimulator(n_hilbert=30, delta=0.35, grid_size=32)

    # Create dataset
    print("  Creating dataset...")
    train_dataset = GKPDataset(sim, n_samples=32, noise_sigma=0.15, online=False)
    val_dataset = GKPDataset(sim, n_samples=8, noise_sigma=0.15, online=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Create model
    print("  Creating model...")
    model = FNODisplacementDecoder(width=16, modes=8, n_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Training loop
    print("  Training...")
    for epoch in range(3):
        model.train()
        train_loss = 0
        for batch in train_loader:
            wigner = batch['wigner'].to(device)
            displacement = batch['displacement'].to(device)

            optimizer.zero_grad()
            pred = model(wigner)
            loss = criterion(pred, displacement)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                wigner = batch['wigner'].to(device)
                displacement = batch['displacement'].to(device)
                pred = model(wigner)
                val_loss += criterion(pred, displacement).item()
        val_loss /= len(val_loader)

        print(f"    Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    print("  Quick training: OK")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  FNO Approximate GKP State Recovery - Test Suite")
    print("=" * 60)

    try:
        # Test components
        sim = test_physics_simulator()
        model = test_fno_model()
        test_dataset(sim)
        test_training_step(sim, model)
        test_evaluation()

        # Quick training test
        run_quick_training()

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou can now run full training with:")
        print("  python train.py --epochs 100")
        print("\nOr with different configurations:")
        print("  python train.py --config high_squeezing")
        print("  python train.py --config low_squeezing")
        print("  python train.py --config reconstruction")

    except Exception as e:
        print(f"\n  TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
