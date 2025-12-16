# -*- coding: utf-8 -*-
"""
FNO-based Approximate GKP State Recovery.

This package implements a Fourier Neural Operator approach for recovering
approximate GKP (Gottesman-Kitaev-Preskill) quantum error correction codes.

Modules:
- config: Configuration management
- physics_simulator: GKP state generation and noise simulation
- fno_model: FNO neural network architectures
- dataset: Data loading and management
- train: Training routines
- evaluate: Evaluation and visualization
"""

from .config import Config, get_default_config
from .physics_simulator import ApproxGKPSimulator
from .fno_model import (
    FNODisplacementDecoder,
    FNOStateReconstructor,
    FNOHybridModel,
    create_model
)

__version__ = '0.1.0'
__all__ = [
    'Config',
    'get_default_config',
    'ApproxGKPSimulator',
    'FNODisplacementDecoder',
    'FNOStateReconstructor',
    'FNOHybridModel',
    'create_model'
]
