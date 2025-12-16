# -*- coding: utf-8 -*-
"""
Physics Simulator for Approximate GKP States.

Based on PRX Quantum review article Section II B:
- Approximate GKP states with Gaussian envelope
- Gaussian displacement noise channel
- Photon loss and dephasing channels

Combines Gemini's state construction with Doubao's multi-modal data approach.
"""

import numpy as np
import qutip as qt
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import warnings
from tqdm import tqdm


@dataclass
class GKPStateData:
    """Container for GKP state data."""
    wigner: np.ndarray          # Wigner function (grid_size x grid_size)
    displacement: np.ndarray    # Applied displacement noise (u, v)
    delta: float                # Squeezing parameter
    logical_value: int          # 0 or 1 for |0_L> or |1_L>
    superposition_coeffs: Optional[Tuple[complex, complex]] = None


class ApproxGKPSimulator:
    """
    Simulator for approximate GKP states with noise channels.

    Implements the physical model from PRX Quantum Sec II B:
    |tilde{mu}_L> ∝ exp(-Delta^2 * n_hat) |mu_L>

    Where the ideal GKP state is constructed as a sum of squeezed states
    with Gaussian envelope weighting.
    """

    def __init__(
        self,
        n_hilbert: int = 50,
        delta: float = 0.3,
        grid_size: int = 64,
        phase_space_extent: float = 6.0
    ):
        """
        Initialize the GKP simulator.

        Args:
            n_hilbert: Hilbert space truncation dimension
            delta: Squeezing parameter (smaller = closer to ideal)
            grid_size: Resolution of Wigner function grid
            phase_space_extent: Phase space range [-extent, extent]
        """
        self.n_hilbert = n_hilbert
        self.delta = delta
        self.grid_size = grid_size
        self.extent = phase_space_extent

        # GKP lattice constant
        self.sqrt_pi = np.sqrt(np.pi)

        # Phase space grid for Wigner function
        self.xvec = np.linspace(-self.extent, self.extent, grid_size)
        self.pvec = np.linspace(-self.extent, self.extent, grid_size)

        # Pre-compute logical basis states
        self._logical_0 = None
        self._logical_1 = None
        self._precompute_logical_states()

    def _precompute_logical_states(self):
        """Pre-compute the logical |0_L> and |1_L> states."""
        print(f"Pre-computing approximate GKP logical states (Delta={self.delta})...")
        self._logical_0 = self._construct_approx_state(logical_val=0)
        self._logical_1 = self._construct_approx_state(logical_val=1)
        print("Logical states ready.")

    def _construct_approx_state(self, logical_val: int) -> qt.Qobj:
        """
        Construct approximate GKP code state.

        Implements Eq. (11) from the paper:
        Sum of squeezed states with Gaussian envelope weighting.

        Args:
            logical_val: 0 for |0_L>, 1 for |1_L>

        Returns:
            Normalized QuTiP quantum state
        """
        psi = qt.Qobj(np.zeros(self.n_hilbert), dims=[[self.n_hilbert], [1]])

        # Determine summation range based on truncation
        # Only terms with significant weight within Hilbert space dimension
        max_displacement = np.sqrt(self.n_hilbert) * 0.7
        n_max = int(max_displacement / (2 * self.sqrt_pi)) + 2

        # Squeezing parameter (converts Delta to squeezing strength r)
        # For approximate GKP: r ≈ -log(Delta)
        r = max(0.1, -np.log(self.delta + 1e-6))

        for k in range(-n_max, n_max + 1):
            # Lattice position: (2k + mu) * sqrt(pi)
            shift = (2 * k + logical_val) * self.sqrt_pi

            # Gaussian envelope weight from Eq. (11)
            weight = np.exp(-0.5 * (self.delta * shift) ** 2)

            if weight < 1e-8:
                continue

            # Phase space displacement alpha = x / sqrt(2) for position shift
            alpha = shift / np.sqrt(2)

            try:
                # Create squeezed vacuum and displace it
                squeezed_vac = qt.squeeze(self.n_hilbert, r) * qt.basis(self.n_hilbert, 0)
                displaced_state = qt.displace(self.n_hilbert, alpha) * squeezed_vac
                psi = psi + weight * displaced_state
            except Exception as e:
                warnings.warn(f"State construction warning at k={k}: {e}")
                continue

        # Normalize the state
        norm = psi.norm()
        if norm > 1e-10:
            return psi.unit()
        else:
            warnings.warn("State has near-zero norm, returning vacuum")
            return qt.basis(self.n_hilbert, 0)

    def get_logical_state(
        self,
        logical_val: Optional[int] = None,
        alpha: Optional[complex] = None,
        beta: Optional[complex] = None,
        delta: Optional[float] = None
    ) -> qt.Qobj:
        """
        Get a logical GKP state.

        Args:
            logical_val: 0 for |0_L>, 1 for |1_L>, None for superposition
            alpha: Coefficient for |0_L> in superposition
            beta: Coefficient for |1_L> in superposition
            delta: Optional different delta (will recompute if different)

        Returns:
            QuTiP quantum state
        """
        # Check if we need to recompute with different delta
        if delta is not None and abs(delta - self.delta) > 1e-6:
            old_delta = self.delta
            self.delta = delta
            self._precompute_logical_states()

        if logical_val is not None:
            if logical_val == 0:
                return self._logical_0.copy()
            else:
                return self._logical_1.copy()
        else:
            # Superposition state
            if alpha is None:
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                alpha = np.cos(theta / 2)
                beta = np.sin(theta / 2) * np.exp(1j * phi)
            psi = alpha * self._logical_0 + beta * self._logical_1
            return psi.unit()

    def apply_displacement_noise(
        self,
        state: qt.Qobj,
        sigma: float = 0.1
    ) -> Tuple[qt.Qobj, np.ndarray]:
        """
        Apply Gaussian displacement noise channel.

        Based on Eq. (26) from paper: D(zeta) where zeta ~ N(0, sigma^2)

        Args:
            state: Input quantum state
            sigma: Standard deviation of displacement noise

        Returns:
            Tuple of (noisy_state, displacement_vector [u, v])
        """
        # Sample displacement from Gaussian
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, sigma)

        # Complex displacement parameter zeta = (u + iv) / sqrt(2)
        zeta = (u + 1j * v) / np.sqrt(2)

        # Apply displacement operator
        D = qt.displace(self.n_hilbert, zeta)
        noisy_state = D * state

        return noisy_state, np.array([u, v])

    def apply_loss_channel(
        self,
        state: qt.Qobj,
        kappa: float = 0.01,
        time: float = 1.0
    ) -> qt.Qobj:
        """
        Apply photon loss channel via Lindblad evolution.

        Based on Eq. (16): d rho/dt = kappa * D[a] rho

        Args:
            state: Input quantum state (ket or density matrix)
            kappa: Loss rate
            time: Evolution time

        Returns:
            Evolved density matrix
        """
        # Convert to density matrix if needed
        if state.isket:
            rho = state * state.dag()
        else:
            rho = state

        # Annihilation operator
        a = qt.destroy(self.n_hilbert)

        # Lindblad collapse operator
        c_ops = [np.sqrt(kappa) * a]

        # Hamiltonian (free evolution, can add cavity frequency if needed)
        H = 0 * a.dag() * a

        # Solve master equation
        times = [0, time]
        result = qt.mesolve(H, rho, times, c_ops, [])

        return result.states[-1]

    def apply_dephasing_channel(
        self,
        state: qt.Qobj,
        kappa_phi: float = 0.005,
        time: float = 1.0
    ) -> qt.Qobj:
        """
        Apply dephasing channel.

        Based on Eq. (16): kappa_phi * D[a^dag a] rho

        Args:
            state: Input quantum state
            kappa_phi: Dephasing rate
            time: Evolution time

        Returns:
            Evolved density matrix
        """
        if state.isket:
            rho = state * state.dag()
        else:
            rho = state

        a = qt.destroy(self.n_hilbert)
        n = a.dag() * a

        c_ops = [np.sqrt(kappa_phi) * n]
        H = 0 * n

        times = [0, time]
        result = qt.mesolve(H, rho, times, c_ops, [])

        return result.states[-1]

    def compute_wigner(
        self,
        state: qt.Qobj,
        xvec: Optional[np.ndarray] = None,
        pvec: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute Wigner function of a quantum state.

        Args:
            state: QuTiP quantum state (ket or density matrix)
            xvec: Optional x-axis grid points
            pvec: Optional p-axis grid points

        Returns:
            2D Wigner function array
        """
        if xvec is None:
            xvec = self.xvec
        if pvec is None:
            pvec = self.pvec

        return qt.wigner(state, xvec, pvec)

    def generate_sample(
        self,
        noise_sigma: float = 0.1,
        noise_type: str = 'displacement',
        random_logical: bool = True,
        random_superposition: bool = True,
        delta: Optional[float] = None,
        kappa: float = 0.01,
        kappa_phi: float = 0.005
    ) -> GKPStateData:
        """
        Generate a single training sample.

        Args:
            noise_sigma: Displacement noise standard deviation
            noise_type: 'displacement', 'loss', 'combined'
            random_logical: Randomly choose |0_L> or |1_L>
            random_superposition: Include random superposition states
            delta: Optional specific delta value
            kappa: Photon loss rate (for loss/combined noise)
            kappa_phi: Dephasing rate (for loss/combined noise)

        Returns:
            GKPStateData containing Wigner function and labels
        """
        # Select logical state
        if random_superposition and np.random.rand() > 0.5:
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            alpha = np.cos(theta / 2)
            beta = np.sin(theta / 2) * np.exp(1j * phi)
            state = self.get_logical_state(alpha=alpha, beta=beta, delta=delta)
            logical_val = -1  # Indicates superposition
            coeffs = (alpha, beta)
        else:
            logical_val = np.random.randint(0, 2) if random_logical else 0
            state = self.get_logical_state(logical_val=logical_val, delta=delta)
            coeffs = None

        # Apply noise
        displacement = np.array([0.0, 0.0])

        if noise_type == 'displacement' or noise_type == 'combined':
            state, displacement = self.apply_displacement_noise(state, noise_sigma)

        if noise_type == 'loss' or noise_type == 'combined':
            state = self.apply_loss_channel(state, kappa)
            state = self.apply_dephasing_channel(state, kappa_phi)

        # Compute Wigner function
        wigner = self.compute_wigner(state)

        # Get current delta (may have been changed)
        current_delta = delta if delta is not None else self.delta

        return GKPStateData(
            wigner=wigner,
            displacement=displacement,
            delta=current_delta,
            logical_value=logical_val,
            superposition_coeffs=coeffs
        )

    def generate_batch(
        self,
        batch_size: int,
        noise_sigma: float = 0.1,
        noise_type: str = 'displacement',
        delta_range: Optional[Tuple[float, float]] = None,
        return_clean: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate a batch of training data.

        Args:
            batch_size: Number of samples
            noise_sigma: Displacement noise std
            noise_type: Type of noise channel
            delta_range: Optional (min, max) for random delta sampling
            return_clean: Whether to also return clean Wigner functions

        Returns:
            Dictionary with:
                'wigner': (batch, 1, grid, grid) noisy Wigner functions
                'displacement': (batch, 2) displacement vectors
                'delta': (batch,) squeezing parameters
                'wigner_clean': (batch, 1, grid, grid) clean Wigner (if return_clean)
        """
        wigners = []
        wigners_clean = []
        displacements = []
        deltas = []

        # Use tqdm for progress display
        pbar = tqdm(range(batch_size), desc="Generating samples", leave=False)

        for i in pbar:
            # Sample delta if range is provided
            if delta_range is not None:
                delta = np.random.uniform(delta_range[0], delta_range[1])
            else:
                delta = None

            # Generate clean state first if needed
            if return_clean:
                if delta is not None and abs(delta - self.delta) > 1e-6:
                    self.delta = delta
                    self._precompute_logical_states()

                logical_val = np.random.randint(0, 2) if np.random.rand() > 0.5 else -1
                if logical_val >= 0:
                    clean_state = self.get_logical_state(logical_val=logical_val)
                else:
                    clean_state = self.get_logical_state()

                wigner_clean = self.compute_wigner(clean_state)
                wigners_clean.append(wigner_clean)

                # Apply noise to same state
                noisy_state, displacement = self.apply_displacement_noise(
                    clean_state, noise_sigma
                )
                wigner_noisy = self.compute_wigner(noisy_state)
            else:
                sample = self.generate_sample(
                    noise_sigma=noise_sigma,
                    noise_type=noise_type,
                    delta=delta
                )
                wigner_noisy = sample.wigner
                displacement = sample.displacement
                delta = sample.delta

            wigners.append(wigner_noisy)
            displacements.append(displacement)
            deltas.append(delta if delta is not None else self.delta)

        result = {
            'wigner': np.array(wigners)[:, np.newaxis, :, :],  # Add channel dim
            'displacement': np.array(displacements),
            'delta': np.array(deltas)
        }

        if return_clean:
            result['wigner_clean'] = np.array(wigners_clean)[:, np.newaxis, :, :]

        return result


def compute_squeezing_db(delta: float) -> float:
    """Convert delta parameter to squeezing in dB."""
    return -10 * np.log10(delta ** 2)


def delta_from_squeezing_db(s_db: float) -> float:
    """Convert squeezing dB to delta parameter."""
    return 10 ** (-s_db / 20)


if __name__ == '__main__':
    # Quick test
    print("Testing ApproxGKPSimulator...")

    sim = ApproxGKPSimulator(n_hilbert=40, delta=0.3, grid_size=32)

    # Generate a single sample
    sample = sim.generate_sample(noise_sigma=0.15)
    print(f"Wigner shape: {sample.wigner.shape}")
    print(f"Displacement: {sample.displacement}")
    print(f"Delta: {sample.delta} ({compute_squeezing_db(sample.delta):.1f} dB)")

    # Generate a batch
    batch = sim.generate_batch(batch_size=4, noise_sigma=0.15)
    print(f"\nBatch Wigner shape: {batch['wigner'].shape}")
    print(f"Batch displacement shape: {batch['displacement'].shape}")

    print("\nSimulator test passed!")
