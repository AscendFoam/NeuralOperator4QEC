# -*- coding: utf-8 -*-
"""
Training script for FNO-based Approximate GKP State Recovery.

Combines:
- MSE loss for displacement prediction
- Optional fidelity-based loss
- Online data generation for better generalization
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json

# Optional tensorboard support
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from config import Config, get_default_config
from physics_simulator import ApproxGKPSimulator, compute_squeezing_db
from fno_model import (
    FNODisplacementDecoder, FNOStateReconstructor,
    FNOHybridModel, create_model, count_parameters
)
from dataset import GKPDataset, OnlineGKPDataset, create_dataloaders


class CombinedLoss(nn.Module):
    """
    Combined loss function for GKP state recovery.

    Includes:
    - MSE loss for displacement prediction
    - Sobolev-like smoothness loss for reconstruction
    """

    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_smooth: float = 0.1,
        output_mode: str = 'displacement'
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_smooth = lambda_smooth
        self.output_mode = output_mode
        self.mse = nn.MSELoss()

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient (Sobolev) loss for smoothness."""
        # Spatial gradients
        pred_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_dy = target[:, :, :, 1:] - target[:, :, :, :-1]

        loss_dx = self.mse(pred_dx, target_dx)
        loss_dy = self.mse(pred_dy, target_dy)

        return loss_dx + loss_dy

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_recon: torch.Tensor = None,
        target_recon: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            pred: Predicted displacement (B, 2)
            target: Target displacement (B, 2)
            pred_recon: Predicted reconstruction (B, 1, H, W)
            target_recon: Target clean Wigner (B, 1, H, W)

        Returns:
            Combined loss scalar
        """
        loss = self.lambda_mse * self.mse(pred, target)

        if pred_recon is not None and target_recon is not None:
            loss_recon = self.mse(pred_recon, target_recon)
            loss_grad = self.gradient_loss(pred_recon, target_recon)
            loss = loss + loss_recon + self.lambda_smooth * loss_grad

        return loss


class Trainer:
    """Trainer class for FNO GKP recovery."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.training.device)

        # Create directories
        self.exp_dir = Path(config.training.checkpoint_dir) / config.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self._save_config()

        # Initialize components
        self._init_simulator()
        self._init_model()
        self._init_dataloaders()
        self._init_optimizer()
        self._init_loss()

        # Logging
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(self.exp_dir / 'logs')
        else:
            self.writer = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _save_config(self):
        """Save configuration to file."""
        config_dict = {
            'physics': vars(self.config.physics),
            'model': vars(self.config.model),
            'training': vars(self.config.training),
            'evaluation': vars(self.config.evaluation),
            'exp_name': self.config.exp_name
        }
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

    def _init_simulator(self):
        """Initialize physics simulator."""
        print("Initializing physics simulator...")
        self.simulator = ApproxGKPSimulator(
            n_hilbert=self.config.physics.n_hilbert,
            delta=self.config.physics.delta,
            grid_size=self.config.physics.grid_size,
            phase_space_extent=self.config.physics.phase_space_extent
        )

    def _init_model(self):
        """Initialize FNO model."""
        print("Initializing FNO model...")
        self.model = create_model(self.config.model).to(self.device)
        print(f"Model parameters: {count_parameters(self.model):,}")

    def _init_dataloaders(self):
        """Initialize data loaders."""
        print("Creating dataloaders...")
        return_clean = self.config.model.output_mode in ['reconstruction', 'hybrid']

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            simulator=self.simulator,
            train_samples=self.config.training.train_samples,
            val_samples=self.config.training.val_samples,
            test_samples=self.config.training.test_samples,
            batch_size=self.config.training.batch_size,
            noise_sigma=self.config.physics.noise_sigma,
            delta_range=self.config.physics.delta_range,
            online=self.config.training.online_learning,
            return_clean=return_clean
        )

    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

        if self.config.training.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs
            )
        elif self.config.training.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=50,
                gamma=0.5
            )
        elif self.config.training.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.training.scheduler_patience,
                factor=self.config.training.scheduler_factor
            )
        else:
            self.scheduler = None

    def _init_loss(self):
        """Initialize loss function."""
        self.criterion = CombinedLoss(
            lambda_mse=self.config.training.lambda_mse,
            lambda_smooth=self.config.training.lambda_stabilizer,
            output_mode=self.config.model.output_mode
        )

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for batch in pbar:
            wigner = batch['wigner'].to(self.device)
            displacement = batch['displacement'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            if self.config.model.output_mode == 'displacement':
                pred = self.model(wigner)
                loss = self.criterion(pred, displacement)
            elif self.config.model.output_mode == 'reconstruction':
                pred_recon = self.model(wigner)
                wigner_clean = batch.get('wigner_clean', wigner).to(self.device)
                loss = nn.MSELoss()(pred_recon, wigner_clean)
            elif self.config.model.output_mode == 'hybrid':
                pred_disp, pred_recon = self.model(wigner)
                wigner_clean = batch.get('wigner_clean', wigner).to(self.device)
                loss = self.criterion(
                    pred_disp, displacement,
                    pred_recon, wigner_clean
                )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0

        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            wigner = batch['wigner'].to(self.device)
            displacement = batch['displacement'].to(self.device)

            if self.config.model.output_mode == 'displacement':
                pred = self.model(wigner)
                loss = self.criterion(pred, displacement)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(displacement.cpu().numpy())
            elif self.config.model.output_mode == 'reconstruction':
                pred_recon = self.model(wigner)
                wigner_clean = batch.get('wigner_clean', wigner).to(self.device)
                loss = nn.MSELoss()(pred_recon, wigner_clean)
            elif self.config.model.output_mode == 'hybrid':
                pred_disp, pred_recon = self.model(wigner)
                wigner_clean = batch.get('wigner_clean', wigner).to(self.device)
                loss = self.criterion(
                    pred_disp, displacement,
                    pred_recon, wigner_clean
                )
                all_preds.append(pred_disp.cpu().numpy())
                all_targets.append(displacement.cpu().numpy())

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Compute additional metrics for displacement mode
        metrics = {'loss': avg_loss}

        if all_preds:
            preds = np.concatenate(all_preds)
            targets = np.concatenate(all_targets)

            # Mean absolute error
            mae = np.mean(np.abs(preds - targets))
            metrics['mae'] = mae

            # RMSE
            rmse = np.sqrt(np.mean((preds - targets) ** 2))
            metrics['rmse'] = rmse

            # Per-component errors
            metrics['mae_u'] = np.mean(np.abs(preds[:, 0] - targets[:, 0]))
            metrics['mae_v'] = np.mean(np.abs(preds[:, 1] - targets[:, 1]))

        return metrics

    def train(self):
        """Full training loop."""
        print(f"\nStarting training for {self.config.training.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Output mode: {self.config.model.output_mode}")

        for epoch in range(self.config.training.epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics['loss']

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('LR', current_lr, epoch)

                for key, value in val_metrics.items():
                    if key != 'loss':
                        self.writer.add_scalar(f'Metrics/{key}', value, epoch)

            # Print progress
            print(f"Epoch {epoch+1}/{self.config.training.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")

            if 'mae' in val_metrics:
                print(f"  MAE: {val_metrics['mae']:.6f} | "
                      f"RMSE: {val_metrics['rmse']:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint('best.pt', epoch, val_metrics)
                print(f"  New best model saved!")
            else:
                self.patience_counter += 1

            # Save periodic checkpoint
            if (epoch + 1) % self.config.training.save_every == 0:
                self._save_checkpoint(f'epoch_{epoch+1}.pt', epoch, val_metrics)

            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Final checkpoint
        self._save_checkpoint('final.pt', epoch, val_metrics)
        if self.writer is not None:
            self.writer.close()

        print(f"\nTraining complete. Best validation loss: {self.best_val_loss:.6f}")
        return self.best_val_loss

    def _save_checkpoint(self, filename: str, epoch: int, metrics: dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': {
                'physics': vars(self.config.physics),
                'model': vars(self.config.model)
            }
        }
        torch.save(checkpoint, self.exp_dir / filename)


def main():
    parser = argparse.ArgumentParser(description='Train FNO for GKP recovery')
    parser.add_argument('--config', type=str, default='default',
                        choices=['default', 'high_squeezing', 'low_squeezing', 'reconstruction'],
                        help='Configuration preset to use')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--delta', type=float, default=None,
                        help='Override squeezing parameter')
    parser.add_argument('--noise_sigma', type=float, default=None,
                        help='Override noise sigma')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (cuda/cpu)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Override experiment name')

    args = parser.parse_args()

    # Load configuration
    if args.config == 'high_squeezing':
        from config import get_high_squeezing_config
        config = get_high_squeezing_config()
    elif args.config == 'low_squeezing':
        from config import get_low_squeezing_config
        config = get_low_squeezing_config()
    elif args.config == 'reconstruction':
        from config import get_reconstruction_config
        config = get_reconstruction_config()
    else:
        config = get_default_config()

    # Apply overrides
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.delta is not None:
        config.physics.delta = args.delta
    if args.noise_sigma is not None:
        config.physics.noise_sigma = args.noise_sigma
    if args.device is not None:
        config.training.device = args.device
    if args.exp_name is not None:
        config.exp_name = args.exp_name

    # Set random seeds
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    # Create trainer and run
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
