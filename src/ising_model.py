import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numba

@numba.jit(nopython=True)
def get_neighbors_sum_numba(lattice, L, i, j):
    """
    Numba-optimized function to calculate the sum of the nearest neighbor spins
    for site (i, j) using periodic boundary conditions.
    """
    top = lattice[(i - 1) % L, j]
    bottom = lattice[(i + 1) % L, j]
    left = lattice[i, (j - 1) % L]
    right = lattice[i, (j + 1) % L]
    return top + bottom + left + right

@numba.jit(nopython=True)
def calculate_delta_E_numba(lattice, L, J, H, i, j):
    """
    Numba-optimized function to calculate the change in energy if the spin
    at (i, j) is flipped.
    ΔE = 2 * s_ij * (J * sum_neighbors + H)
    """
    spin_ij = lattice[i, j]
    neighbors_sum = get_neighbors_sum_numba(lattice, L, i, j)
    delta_E = 2 * spin_ij * (J * neighbors_sum + H)
    return delta_E

@numba.jit(nopython=True)
def monte_carlo_step_numba(lattice, L, N, J, H, beta):
    """
    Numba-optimized function for one Monte Carlo step using the Metropolis algorithm.
    One step consists of N attempted spin flips (where N = L*L).
    """
    for _ in range(N):
        # 1. Choose a random spin site
        i = random.randrange(L)
        j = random.randrange(L)

        # 2. Calculate the energy change if this spin is flipped
        delta_E = calculate_delta_E_numba(lattice, L, J, H, i, j)

        # 3. Metropolis acceptance criterion
        if delta_E < 0 or random.random() < np.exp(-delta_E * beta):
            lattice[i, j] *= -1 # Flip the spin
    return lattice

@numba.jit(nopython=True)
def calculate_energy_numba(lattice, L, J, H):
    """
    Numba-optimized function to calculate the total energy of the lattice.
    E = -J Σ_<i,j> s_i s_j - H Σ_i s_i
    The sum Σ_<i,j> is over nearest neighbor pairs (each pair counted once).
    """
    energy = 0.0
    for i in range(L):
        for j in range(L):
            spin_ij = lattice[i, j]
            neighbor_sum = lattice[i, (j + 1) % L] + lattice[(i + 1) % L, j]
            energy += -J * spin_ij * neighbor_sum
            energy += -H * spin_ij
    return energy

@numba.jit(nopython=True)
def calculate_magnetization_numba(lattice):
    """
    Numba-optimized function to calculate the total magnetization of the lattice.
    M = Σ_i s_i
    """
    return np.sum(lattice)

class Ising2D:
    """
    Implements a 2D Ising model simulation using the Metropolis Monte Carlo algorithm.
    Optimized with Numba for speed.
    """
    def __init__(self, L, initial_state='random', J=1.0, H=0.0):
        """
        Initializes the Ising model on an LxL lattice.

        Args:
            L (int): Linear size of the square lattice.
            initial_state (str): 'random' for random initial spins, 'up' for all spins +1,
                                    'down' for all spins -1.
            J (float): Interaction strength (J > 0 for ferromagnetic).
            H (float): External magnetic field strength.
        """
        self.L = L
        self.N = L * L  # Total number of spins
        self.J = J
        self.H = H

        if initial_state == 'random':
            self.lattice = np.random.choice([1, -1], size=(L, L))
        elif initial_state == 'up':
            self.lattice = np.ones((L, L), dtype=np.int8)
        elif initial_state == 'down':
            self.lattice = -np.ones((L, L), dtype=np.int8)
        else:
            raise ValueError("initial_state must be 'random', 'up', or 'down'")

    def monte_carlo_step(self, beta):
        """
        Performs one Monte Carlo step using the Metropolis algorithm, optimized with Numba.
        One step consists of N attempted spin flips (where N = L*L).

        Args:
            beta (float): Inverse temperature (1 / (k_B * T)). k_B is assumed to be 1.
        """
        self.lattice = monte_carlo_step_numba(self.lattice, self.L, self.N, self.J, self.H, beta)

    def calculate_energy(self):
        """
        Calculates the total energy of the lattice using the Numba-optimized function.
        E = -J Σ_<i,j> s_i s_j - H Σ_i s_i
        The sum Σ_<i,j> is over nearest neighbor pairs (each pair counted once).
        """
        return calculate_energy_numba(self.lattice, self.L, self.J, self.H)

    def calculate_magnetization(self):
        """
        Calculates the total magnetization of the lattice using the Numba-optimized function.
        M = Σ_i s_i
        """
        return calculate_magnetization_numba(self.lattice)

    def run_simulation(self, T, num_steps, equilibration_steps=500):
        """
        Runs the Monte Carlo simulation for a given temperature.

        Args:
            T (float): Temperature (k_B is assumed to be 1).
            num_steps (int): Total number of Monte Carlo steps (each step = N spin flips).
            equilibration_steps (int): Number of initial steps to discard for equilibration.

        Returns:
            tuple: (energies, magnetizations)
                   - energies (list): List of total energy per site (E/N) at each step after equilibration.
                   - magnetizations (list): List of magnetization per site (M/N) at each step after equilibration.
        """
        if T <= 0:
            if T == 0:
                beta = np.inf
            else:
                raise ValueError("Temperature must be positive.")
        else:
            beta = 1.0 / T

        energies = []
        magnetizations = []

        print(f"Equilibrating for {equilibration_steps} steps at T={T:.2f}...")
        for _ in tqdm(range(equilibration_steps), desc="Equilibration"):
            self.monte_carlo_step(beta)

        print(f"Running simulation for {num_steps - equilibration_steps} steps at T={T:.2f}...")
        for _ in tqdm(range(num_steps - equilibration_steps), desc="Simulation"):
            self.monte_carlo_step(beta)
            energy = self.calculate_energy()
            magnetization = self.calculate_magnetization()
            energies.append(energy / self.N)
            magnetizations.append(magnetization / self.N)

        return energies, magnetizations

    def get_lattice(self):
        """
        Returns the current state of the lattice.
        """
        return self.lattice.copy()