# -*- coding: utf-8 -*-
"""
Evaluation script for FNO-based Approximate GKP State Recovery.

Includes:
- Model evaluation metrics
- Baseline comparisons (Homodyne + Binning)
- Visualization utilities
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

from config import Config, get_default_config
from physics_simulator import ApproxGKPSimulator, compute_squeezing_db
from fno_model import create_model, FNODisplacementDecoder
from dataset import GKPDataset


def homodyne_binning_decoder(
    wigner: np.ndarray,
    xvec: np.ndarray,
    pvec: np.ndarray
) -> np.ndarray:
    """
    Baseline decoder using homodyne measurement + binning.

    Based on Section II D1 of PRX Quantum paper.
    Simulates homodyne measurement and rounds to nearest lattice point.

    Args:
        wigner: Wigner function (H, W)
        xvec: x-axis grid points
        pvec: p-axis grid points

    Returns:
        Estimated displacement (u, v)
    """
    sqrt_pi = np.sqrt(np.pi)

    # Find the maximum of Wigner function (simple peak detection)
    # This simulates a rough estimate from homodyne measurement
    idx = np.unravel_index(np.argmax(np.abs(wigner)), wigner.shape)
    x_peak = xvec[idx[1]]
    p_peak = pvec[idx[0]]

    # Round to nearest GKP lattice point (binning)
    x_binned = np.round(x_peak / sqrt_pi) * sqrt_pi
    p_binned = np.round(p_peak / sqrt_pi) * sqrt_pi

    # Estimated displacement is the deviation from lattice
    u_est = x_peak - x_binned
    v_est = p_peak - p_binned

    return np.array([u_est, v_est])


def compute_logical_error_rate(
    true_displacement: np.ndarray,
    pred_displacement: np.ndarray,
    threshold: float = None
) -> float:
    """
    Compute logical error rate.

    A logical error occurs when the corrected displacement
    moves across a GKP lattice boundary.

    Args:
        true_displacement: True displacement (N, 2)
        pred_displacement: Predicted displacement (N, 2)
        threshold: Decision boundary (default: sqrt(pi)/2)

    Returns:
        Logical error rate
    """
    if threshold is None:
        threshold = np.sqrt(np.pi) / 2

    # Residual displacement after correction
    residual = true_displacement - pred_displacement

    # Logical error if residual crosses lattice boundary
    errors = np.any(np.abs(residual) > threshold, axis=1)

    return np.mean(errors)


def evaluate_reconstruction_model(
    model: torch.nn.Module,
    simulator: ApproxGKPSimulator,
    n_samples: int = 200,
    noise_sigmas: list = None,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate reconstruction model using Wigner function MSE.

    Args:
        model: Trained FNO reconstruction model
        simulator: Physics simulator
        n_samples: Number of test samples per noise level
        noise_sigmas: List of noise levels to test
        device: Computation device

    Returns:
        Dictionary of evaluation metrics
    """
    if noise_sigmas is None:
        noise_sigmas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    model.eval()
    device = torch.device(device)

    results = {
        'noise_sigmas': noise_sigmas,
        'fno_mse': [],
        'fno_psnr': [],
        'noisy_mse': [],
    }

    print("Evaluating reconstruction model across noise levels...")

    for sigma in noise_sigmas:
        print(f"\n  Noise sigma = {sigma:.2f}")

        # Generate test data with clean references
        batch = simulator.generate_batch(
            batch_size=n_samples,
            noise_sigma=sigma,
            noise_type='displacement',
            return_clean=True
        )

        wigners_noisy = torch.tensor(batch['wigner'], dtype=torch.float32).to(device)
        wigners_clean = batch.get('wigner_clean', batch['wigner'])

        # FNO reconstruction
        with torch.no_grad():
            fno_recon = model(wigners_noisy).cpu().numpy()

        # Compute metrics
        # MSE between reconstructed and clean
        fno_mse = np.mean((fno_recon - wigners_clean) ** 2)

        # MSE between noisy and clean (baseline)
        noisy_mse = np.mean((batch['wigner'] - wigners_clean) ** 2)

        # PSNR (Peak Signal-to-Noise Ratio)
        max_val = np.max(np.abs(wigners_clean))
        fno_psnr = 10 * np.log10(max_val ** 2 / (fno_mse + 1e-10))

        results['fno_mse'].append(fno_mse)
        results['fno_psnr'].append(fno_psnr)
        results['noisy_mse'].append(noisy_mse)

        print(f"    FNO Recon MSE: {fno_mse:.6f}, PSNR: {fno_psnr:.2f} dB")
        print(f"    Noisy MSE:     {noisy_mse:.6f}")
        print(f"    Improvement:   {noisy_mse/fno_mse:.2f}x")

    return results


def evaluate_model(
    model: torch.nn.Module,
    simulator: ApproxGKPSimulator,
    n_samples: int = 200,
    noise_sigmas: list = None,
    device: str = 'cuda',
    output_mode: str = 'displacement'
) -> Dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained FNO model
        simulator: Physics simulator
        n_samples: Number of test samples per noise level
        noise_sigmas: List of noise levels to test
        device: Computation device
        output_mode: 'displacement' or 'reconstruction'

    Returns:
        Dictionary of evaluation metrics
    """
    if noise_sigmas is None:
        noise_sigmas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    model.eval()
    device = torch.device(device)

    # Detect output mode by doing a test forward pass
    test_input = torch.randn(1, 1, simulator.grid_size, simulator.grid_size).to(device)
    with torch.no_grad():
        test_output = model(test_input)

    # Check output shape to determine mode
    if test_output.shape[-1] == 2 and len(test_output.shape) == 2:
        detected_mode = 'displacement'
    else:
        detected_mode = 'reconstruction'

    print(f"Detected model output mode: {detected_mode}")

    if detected_mode == 'reconstruction':
        return evaluate_reconstruction_model(model, simulator, n_samples, noise_sigmas, device)

    results = {
        'noise_sigmas': noise_sigmas,
        'fno_mae': [],
        'fno_rmse': [],
        'fno_error_rate': [],
        'binning_mae': [],
        'binning_rmse': [],
        'binning_error_rate': []
    }

    print("Evaluating model across noise levels...")

    for sigma in noise_sigmas:
        print(f"\n  Noise sigma = {sigma:.2f}")

        # Generate test data
        batch = simulator.generate_batch(
            batch_size=n_samples,
            noise_sigma=sigma,
            noise_type='displacement'
        )

        wigners = torch.tensor(batch['wigner'], dtype=torch.float32).to(device)
        true_displacements = batch['displacement']

        # FNO predictions
        with torch.no_grad():
            fno_preds = model(wigners).cpu().numpy()

        # Baseline predictions
        binning_preds = np.array([
            homodyne_binning_decoder(
                batch['wigner'][i, 0],
                simulator.xvec,
                simulator.pvec
            )
            for i in range(n_samples)
        ])

        # Compute metrics for FNO
        fno_errors = fno_preds - true_displacements
        fno_mae = np.mean(np.abs(fno_errors))
        fno_rmse = np.sqrt(np.mean(fno_errors ** 2))
        fno_error_rate = compute_logical_error_rate(true_displacements, fno_preds)

        # Compute metrics for baseline
        binning_errors = binning_preds - true_displacements
        binning_mae = np.mean(np.abs(binning_errors))
        binning_rmse = np.sqrt(np.mean(binning_errors ** 2))
        binning_error_rate = compute_logical_error_rate(true_displacements, binning_preds)

        # Store results
        results['fno_mae'].append(fno_mae)
        results['fno_rmse'].append(fno_rmse)
        results['fno_error_rate'].append(fno_error_rate)
        results['binning_mae'].append(binning_mae)
        results['binning_rmse'].append(binning_rmse)
        results['binning_error_rate'].append(binning_error_rate)

        print(f"    FNO:     MAE={fno_mae:.4f}, RMSE={fno_rmse:.4f}, ErrorRate={fno_error_rate:.4f}")
        print(f"    Binning: MAE={binning_mae:.4f}, RMSE={binning_rmse:.4f}, ErrorRate={binning_error_rate:.4f}")

    return results


def evaluate_across_deltas(
    model: torch.nn.Module,
    n_hilbert: int = 50,
    grid_size: int = 64,
    n_samples: int = 100,
    deltas: list = None,
    noise_sigma: float = 0.15,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate model performance across different squeezing levels.

    Args:
        model: Trained FNO model
        n_hilbert: Hilbert space dimension
        grid_size: Wigner function grid size
        n_samples: Number of samples per delta
        deltas: List of delta values to test
        noise_sigma: Fixed noise level
        device: Computation device

    Returns:
        Evaluation results across deltas
    """
    if deltas is None:
        deltas = [0.2, 0.25, 0.3, 0.35, 0.4]

    model.eval()
    device = torch.device(device)

    results = {
        'deltas': deltas,
        'squeezing_db': [compute_squeezing_db(d) for d in deltas],
        'fno_mae': [],
        'fno_error_rate': [],
        'binning_mae': [],
        'binning_error_rate': []
    }

    print("\nEvaluating across squeezing levels...")

    for delta in deltas:
        print(f"\n  Delta = {delta:.2f} ({compute_squeezing_db(delta):.1f} dB)")

        # Create simulator for this delta
        sim = ApproxGKPSimulator(
            n_hilbert=n_hilbert,
            delta=delta,
            grid_size=grid_size
        )

        # Generate test data
        batch = sim.generate_batch(
            batch_size=n_samples,
            noise_sigma=noise_sigma
        )

        wigners = torch.tensor(batch['wigner'], dtype=torch.float32).to(device)
        true_displacements = batch['displacement']

        # FNO predictions
        with torch.no_grad():
            fno_preds = model(wigners).cpu().numpy()

        # Baseline predictions
        binning_preds = np.array([
            homodyne_binning_decoder(batch['wigner'][i, 0], sim.xvec, sim.pvec)
            for i in range(n_samples)
        ])

        # Store metrics
        results['fno_mae'].append(np.mean(np.abs(fno_preds - true_displacements)))
        results['fno_error_rate'].append(
            compute_logical_error_rate(true_displacements, fno_preds)
        )
        results['binning_mae'].append(np.mean(np.abs(binning_preds - true_displacements)))
        results['binning_error_rate'].append(
            compute_logical_error_rate(true_displacements, binning_preds)
        )

    return results


def plot_results(results: Dict, save_path: Optional[str] = None):
    """
    Plot evaluation results.

    Args:
        results: Dictionary from evaluate_model()
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # MAE vs noise
    ax = axes[0]
    ax.plot(results['noise_sigmas'], results['fno_mae'], 'b-o', label='FNO', linewidth=2)
    ax.plot(results['noise_sigmas'], results['binning_mae'], 'r--s', label='Homodyne+Binning', linewidth=2)
    ax.set_xlabel('Noise Sigma', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('MAE vs Noise Level', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE vs noise
    ax = axes[1]
    ax.plot(results['noise_sigmas'], results['fno_rmse'], 'b-o', label='FNO', linewidth=2)
    ax.plot(results['noise_sigmas'], results['binning_rmse'], 'r--s', label='Homodyne+Binning', linewidth=2)
    ax.set_xlabel('Noise Sigma', fontsize=12)
    ax.set_ylabel('Root Mean Square Error', fontsize=12)
    ax.set_title('RMSE vs Noise Level', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error rate vs noise
    ax = axes[2]
    ax.plot(results['noise_sigmas'], results['fno_error_rate'], 'b-o', label='FNO', linewidth=2)
    ax.plot(results['noise_sigmas'], results['binning_error_rate'], 'r--s', label='Homodyne+Binning', linewidth=2)
    ax.set_xlabel('Noise Sigma', fontsize=12)
    ax.set_ylabel('Logical Error Rate', fontsize=12)
    ax.set_title('Error Rate vs Noise Level', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_delta_results(results: Dict, save_path: Optional[str] = None):
    """Plot results across squeezing levels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # MAE vs squeezing
    ax = axes[0]
    ax.plot(results['squeezing_db'], results['fno_mae'], 'b-o', label='FNO', linewidth=2)
    ax.plot(results['squeezing_db'], results['binning_mae'], 'r--s', label='Homodyne+Binning', linewidth=2)
    ax.set_xlabel('Squeezing (dB)', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('MAE vs Squeezing Level', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error rate vs squeezing
    ax = axes[1]
    ax.plot(results['squeezing_db'], results['fno_error_rate'], 'b-o', label='FNO', linewidth=2)
    ax.plot(results['squeezing_db'], results['binning_error_rate'], 'r--s', label='Homodyne+Binning', linewidth=2)
    ax.set_xlabel('Squeezing (dB)', fontsize=12)
    ax.set_ylabel('Logical Error Rate', fontsize=12)
    ax.set_title('Error Rate vs Squeezing Level', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_reconstruction_results(results: Dict, save_path: Optional[str] = None):
    """Plot results for reconstruction model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # MSE vs noise
    ax = axes[0]
    ax.plot(results['noise_sigmas'], results['fno_mse'], 'b-o', label='FNO Reconstruction', linewidth=2)
    ax.plot(results['noise_sigmas'], results['noisy_mse'], 'r--s', label='Noisy (No correction)', linewidth=2)
    ax.set_xlabel('Noise Sigma', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Reconstruction MSE vs Noise Level', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # PSNR vs noise
    ax = axes[1]
    ax.plot(results['noise_sigmas'], results['fno_psnr'], 'b-o', label='FNO PSNR', linewidth=2)
    ax.set_xlabel('Noise Sigma', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Reconstruction PSNR vs Noise Level', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def visualize_prediction(
    model: torch.nn.Module,
    simulator: ApproxGKPSimulator,
    noise_sigma: float = 0.15,
    device: str = 'cuda',
    save_path: Optional[str] = None
):
    """
    Visualize a single prediction example.

    Args:
        model: Trained FNO model
        simulator: Physics simulator
        noise_sigma: Noise level
        device: Computation device
        save_path: Optional save path
    """
    model.eval()
    device = torch.device(device)

    # Generate a sample
    sample = simulator.generate_sample(noise_sigma=noise_sigma)
    wigner_noisy = sample.wigner
    true_disp = sample.displacement

    # Get clean state for comparison
    clean_state = simulator.get_logical_state(logical_val=0)
    wigner_clean = simulator.compute_wigner(clean_state)

    # Predict
    wigner_tensor = torch.tensor(
        wigner_noisy[np.newaxis, np.newaxis, :, :],
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        pred_disp = model(wigner_tensor).cpu().numpy()[0]

    # Binning baseline
    binning_disp = homodyne_binning_decoder(wigner_noisy, simulator.xvec, simulator.pvec)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Clean Wigner
    ax = axes[0]
    im = ax.pcolormesh(simulator.xvec, simulator.pvec, wigner_clean,
                       shading='auto', cmap='RdBu_r')
    ax.set_xlabel('q', fontsize=12)
    ax.set_ylabel('p', fontsize=12)
    ax.set_title('Clean GKP State', fontsize=14)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    # Noisy Wigner
    ax = axes[1]
    im = ax.pcolormesh(simulator.xvec, simulator.pvec, wigner_noisy,
                       shading='auto', cmap='RdBu_r')
    ax.set_xlabel('q', fontsize=12)
    ax.set_ylabel('p', fontsize=12)
    ax.set_title(f'Noisy (sigma={noise_sigma})', fontsize=14)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    # Displacement comparison
    ax = axes[2]
    width = 0.25
    x = np.arange(2)
    ax.bar(x - width, true_disp, width, label='True', alpha=0.8)
    ax.bar(x, pred_disp, width, label='FNO', alpha=0.8)
    ax.bar(x + width, binning_disp, width, label='Binning', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['u', 'v'])
    ax.set_ylabel('Displacement', fontsize=12)
    ax.set_title('Displacement Prediction', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    # Print values
    print(f"\nTrue displacement: u={true_disp[0]:.4f}, v={true_disp[1]:.4f}")
    print(f"FNO prediction:   u={pred_disp[0]:.4f}, v={pred_disp[1]:.4f}")
    print(f"Binning estimate: u={binning_disp[0]:.4f}, v={binning_disp[1]:.4f}")

    fno_error = np.sqrt(np.sum((pred_disp - true_disp) ** 2))
    binning_error = np.sqrt(np.sum((binning_disp - true_disp) ** 2))
    print(f"\nFNO error:     {fno_error:.4f}")
    print(f"Binning error: {binning_error:.4f}")


def load_model(checkpoint_path: str, device: str = 'cuda') -> Tuple[torch.nn.Module, Dict]:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Target device

    Returns:
        Tuple of (model, config_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    # Create model with saved configuration
    from config import ModelConfig
    mc = ModelConfig(**model_config) if model_config else ModelConfig()
    model = create_model(mc).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint:
        print(f"  Val Loss: {checkpoint['metrics'].get('loss', 'N/A'):.6f}")

    return model, config


def main():
    parser = argparse.ArgumentParser(description='Evaluate FNO GKP model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of test samples')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    model, config = load_model(args.checkpoint, device)

    # Create simulator
    physics_config = config.get('physics', {})
    simulator = ApproxGKPSimulator(
        n_hilbert=physics_config.get('n_hilbert', 50),
        delta=physics_config.get('delta', 0.3),
        grid_size=physics_config.get('grid_size', 64)
    )

    # Detect model type
    test_input = torch.randn(1, 1, simulator.grid_size, simulator.grid_size).to(device)
    with torch.no_grad():
        test_output = model(test_input)

    is_reconstruction = not (test_output.shape[-1] == 2 and len(test_output.shape) == 2)

    # Run evaluation
    print("\n" + "=" * 50)
    print("Evaluating across noise levels...")
    print("=" * 50)

    results = evaluate_model(
        model, simulator,
        n_samples=args.n_samples,
        device=device
    )

    # Save results
    with open(save_dir / 'noise_results.json', 'w') as f:
        json.dump({k: [float(v) for v in vals] if isinstance(vals, list) else vals
                   for k, vals in results.items()}, f, indent=2)

    # Plot based on mode
    if is_reconstruction:
        plot_reconstruction_results(results, save_dir / 'reconstruction_comparison.png')
    else:
        plot_results(results, save_dir / 'noise_comparison.png')

        # Evaluate across deltas (only for displacement mode)
        print("\n" + "=" * 50)
        print("Evaluating across squeezing levels...")
        print("=" * 50)

        delta_results = evaluate_across_deltas(
            model,
            n_hilbert=physics_config.get('n_hilbert', 50),
            grid_size=physics_config.get('grid_size', 64),
            n_samples=args.n_samples // 2,
            device=device
        )

        with open(save_dir / 'delta_results.json', 'w') as f:
            json.dump({k: [float(v) for v in vals] if isinstance(vals, list) else vals
                       for k, vals in delta_results.items()}, f, indent=2)

        plot_delta_results(delta_results, save_dir / 'delta_comparison.png')

        # Visualize example (only for displacement mode)
        print("\n" + "=" * 50)
        print("Visualizing example prediction...")
        print("=" * 50)

        visualize_prediction(
            model, simulator,
            noise_sigma=0.15,
            device=device,
            save_path=save_dir / 'example_prediction.png'
        )

    print(f"\nResults saved to {save_dir}")


if __name__ == '__main__':
    main()
