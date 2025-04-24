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

def monte_carlo_step_fully_vectorized(lattice, L, N, J, H, beta):
    """
    Fully vectorized Monte Carlo step for the Metropolis algorithm.
    Performs N attempted spin flips in a single batch.
    """
    # 1. Generate random indices for spin sites
    i_indices = np.random.randint(0, L, size=N)
    j_indices = np.random.randint(0, L, size=N)
    
    # 2. Calculate delta_E for all flips in batch (vectorized)
    # Get all neighbors at once using periodic boundary conditions
    i_up = (i_indices - 1) % L
    i_down = (i_indices + 1) % L
    j_left = (j_indices - 1) % L
    j_right = (j_indices + 1) % L
    
    # Current spins at the selected positions
    current_spins = lattice[i_indices, j_indices]
    
    # Sum of neighboring spins
    neighbor_sum = (lattice[i_up, j_indices] + lattice[i_down, j_indices] + 
                    lattice[i_indices, j_left] + lattice[i_indices, j_right])
    
    # Delta E calculation: 2 * spin * (J * sum_neighbors + H)
    delta_Es = 2 * current_spins * (J * neighbor_sum + H)
    
    # 3. Calculate acceptance probabilities
    acceptance_probs = np.exp(-delta_Es * beta)
    random_values = np.random.random(N)
    
    # 4. Determine which spins to flip
    flip_mask = (delta_Es < 0) | (random_values < acceptance_probs)
    
    # 5. Flip spins in a single operation
    lattice[i_indices[flip_mask], j_indices[flip_mask]] *= -1
    
    return lattice

def calculate_energy_vectorized_lowmem(lattice, L, J, H):
    """
    Vectorized version with reduced memory usage
    """
    # Calculate right and down neighbor sums in one operation
    neighbor_sum = np.roll(lattice, -1, axis=1) + np.roll(lattice, -1, axis=0)
    
    # Compute both energy terms
    interaction_energy = -J * np.sum(lattice * neighbor_sum)
    field_energy = -H * np.sum(lattice)
    
    return interaction_energy + field_energy



@numba.jit(nopython=True)
def calculate_correlation_function_numba(lattice, L, max_distance=None):
    """
    Calculate the spin-spin correlation function C(r) = ⟨σ(0)σ(r)⟩ - ⟨σ⟩²
    for all distances r up to max_distance.
    
    Args:
        lattice: 2D numpy array of spins
        L: System size
        max_distance: Maximum distance to calculate (default L//2)
        
    Returns:
        C(r) as a 1D numpy array for r = 0 to max_distance-1
    """
    if max_distance is None:
        max_distance = L // 2
    
    # Initialize correlation array
    correlations = np.zeros(max_distance)
    counts = np.zeros(max_distance)
    
    # Calculate average magnetization
    avg_magnetization = np.mean(lattice)
    
    # Compute all possible correlations
    for i in range(L):
        for j in range(L):
            current_spin = lattice[i, j]
            for r in range(1, max_distance+1):
                # Check neighbors at distance r in x direction
                j_right = (j + r) % L
                correlations[r-1] += current_spin * lattice[i, j_right]
                counts[r-1] += 1
                
                # Check neighbors at distance r in y direction
                i_down = (i + r) % L
                correlations[r-1] += current_spin * lattice[i_down, j]
                counts[r-1] += 1
    
    # Normalize and subtract ⟨σ⟩²
    correlations /= counts
    correlations -= avg_magnetization**2
    
    return correlations


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
            self.lattice = self.lattice.astype(np.int8)  # Uses only 1 byte per spin
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
        self.lattice = monte_carlo_step_fully_vectorized(self.lattice, self.L, self.N, self.J, self.H, beta)

    def calculate_energy(self):
        """
        Calculates the total energy of the lattice using the Numba-optimized function.
        E = -J Σ_<i,j> s_i s_j - H Σ_i s_i
        The sum Σ_<i,j> is over nearest neighbor pairs (each pair counted once).
        """
        return calculate_energy_vectorized_lowmem(self.lattice, self.L, self.J, self.H)

    def calculate_magnetization(self):
        """
        Calculates the total magnetization of the lattice using the Numba-optimized function.
        M = Σ_i s_i
        """
        return calculate_magnetization_numba(self.lattice)

    def run_simulation(self, T, num_steps:600, equilibration_steps=500, correlation_only=False):
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
        # Pre-allocate arrays for results
        if not correlation_only:
            energies = np.empty(num_steps - equilibration_steps)    
            magnetizations = np.empty(num_steps - equilibration_steps)
        correlations = np.empty(num_steps - equilibration_steps)

        print(f"Equilibrating for {equilibration_steps} steps at T={T:.2f}...")
        for _ in tqdm(range(equilibration_steps), desc="Equilibration"):
            self.monte_carlo_step(beta)

        print(f"Running simulation for {num_steps - equilibration_steps} steps at T={T:.2f}...")
        
        if correlation_only:
            for step in tqdm(range(num_steps - equilibration_steps), desc="Simulation"):
                self.monte_carlo_step(beta)
                correlations[step] = self.calculate_correlation_function()
            return correlations
        else: 
            for step in tqdm(range(num_steps - equilibration_steps), desc="Simulation"):
                self.monte_carlo_step(beta)
                energy = self.calculate_energy()
                magnetization = self.calculate_magnetization()
                energies[step] = energy / self.N
                magnetizations[step] = magnetization / self.N
            return energies, magnetizations

    def get_lattice(self):
        """
        Returns the current state of the lattice.
        """
        return self.lattice.copy()



    def calculate_correlation_function(self, max_distance=None):
        """
        Wrapper method to calculate the correlation function for the current lattice state.
        
        Args:
            max_distance: Maximum distance to calculate (default L//2)
            
        Returns:
            C(r) as a 1D numpy array for r = 0 to max_distance-1
        """
        if max_distance is None:
            max_distance = self.L // 2
        return calculate_correlation_function_numba(self.lattice, self.L, max_distance)
