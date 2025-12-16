# -*- coding: utf-8 -*-
"""Quick speed test for data generation."""

from physics_simulator import ApproxGKPSimulator
import time

print('Creating simulator...')
start = time.time()
sim = ApproxGKPSimulator(n_hilbert=40, delta=0.3, grid_size=32)
print(f'Simulator created in {time.time()-start:.1f}s')

print('Generating 50 samples...')
start = time.time()
batch = sim.generate_batch(batch_size=50, noise_sigma=0.15)
print(f'50 samples generated in {time.time()-start:.1f}s')
print(f'Wigner shape: {batch["wigner"].shape}')
print('Speed test passed!')
