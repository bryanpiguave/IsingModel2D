# Calculate correlation function
from ising_model import Ising2D
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba

class questionF_model(Ising2D):
    def __init__(self, L=20, J=1.0, H=0.0, initial_state='random'):
        super().__init__(L=L, J=J, H=H, initial_state=initial_state)
        self._precompute_distances()
        
    def _precompute_distances(self):
        """Precompute all possible distances between spins"""
        self.distance_matrix = np.zeros((self.L, self.L, self.L, self.L), dtype=int)
        for i in range(self.L):
            for j in range(self.L):
                for k in range(self.L):
                    for l in range(self.L):
                        dx = min(abs(i-k), self.L-abs(i-k))
                        dy = min(abs(j-l), self.L-abs(j-l))
                        self.distance_matrix[i,j,k,l] = dx + dy  # Manhattan distance
    
    @staticmethod
    @numba.jit(nopython=True)
    def _metropolis_step(lattice, L, N, J, H, beta):
        """Single Monte Carlo step using Metropolis algorithm"""
        for _ in range(N):
            i, j = np.random.randint(0, L), np.random.randint(0, L)
            spin = lattice[i, j]
            
            # Periodic boundary conditions
            neighbors = (lattice[(i+1)%L, j] + lattice[(i-1)%L, j] +
                        lattice[i, (j+1)%L] + lattice[i, (j-1)%L])
            
            delta_E = 2 * spin * (J * neighbors + H)
            
            if delta_E < 0 or np.random.random() < np.exp(-beta * delta_E):
                lattice[i, j] *= -1
    
    def run_simulation(self, T, steps=10000, eq_steps=2000, sample_interval=10):
        """
        Run Monte Carlo simulation
        
        Args:
            T: Temperature
            steps: Total MC steps
            eq_steps: Equilibration steps
            sample_interval: Interval between measurements
            
        Returns:
            lattice_samples: List of sampled lattice configurations
        """
        beta = 1.0/T if T > 0 else float('inf')
        
        # Equilibration
        for _ in tqdm(range(eq_steps), desc=f"Equilibrating at T={T:.2f}"):
            self._metropolis_step(self.lattice, self.L, self.N, self.J, self.H, beta)
            
        # Production run
        lattice_samples = []
        for step in tqdm(range(steps), desc=f"Sampling at T={T:.2f}"):
            self._metropolis_step(self.lattice, self.L, self.N, self.J, self.H, beta)
            if step % sample_interval == 0:
                lattice_samples.append(self.lattice.copy())        
        return lattice_samples
    
    def calculate_correlation_function(self, lattice_samples):
        """
        Calculate spin-spin correlation function:
        C(r) = ⟨s_i s_j⟩ - ⟨s_i⟩⟨s_j⟩ for spins at distance r
        
        Args:
            lattice_samples: List of lattice configurations
            
        Returns:
            (distances, correlations): Unique distances and corresponding correlations
        """
        # First calculate average magnetization
        avg_mag = np.mean([np.mean(lattice) for lattice in lattice_samples])
        
        # Get all unique distances
        unique_dists = np.unique(self.distance_matrix)
        max_dist = min(self.L//2, 10)  # Only calculate up to half system size or 10
        unique_dists = unique_dists[unique_dists <= max_dist]
        
        correlations = np.zeros(len(unique_dists))
        counts = np.zeros(len(unique_dists))
        
        for lattice in lattice_samples:
            # Make sure lattice is 2D
            if len(lattice.shape) == 1:
                lattice = lattice.reshape((self.L, self.L))
                
            for i in range(self.L):
                for j in range(self.L):
                    for k in range(self.L):
                        for l in range(self.L):
                            dist = self.distance_matrix[i,j,k,l]
                            if dist > max_dist:
                                continue
                            idx = np.where(unique_dists == dist)[0][0]
                            correlations[idx] += lattice[i,j] * lattice[k,l]
                            counts[idx] += 1
        
        correlations /= counts
        correlations -= avg_mag**2
        return unique_dists, correlations

def analyze_critical_behavior():
    """Run simulations above, at, and below critical temperature"""
    # Critical temperature for 2D Ising model
    Tc = 2.0 / np.log(1 + np.sqrt(2))  # ≈ 2.269
    
    temperatures = [1.5, Tc, 3.0]  # Below, at, and above Tc
    results = {}
    
    for T in temperatures:
        model = questionF_model(L=20)
        lattices = model.run_simulation(T=T, steps=100, eq_steps=10000)
        dists, corrs = model.calculate_correlation_function(lattices)
        
        # Calculate average magnetization for each sample
        mags = [np.mean(lattice) for lattice in lattices]
        
        results[T] = {
            'avg_mag': np.mean(mags),
            'avg_mag_std': np.std(mags),
            'dists': dists,
            'corrs': corrs,
        }
    # Save results
    pd.DataFrame(results).to_csv('outputs/question_f.csv')
    # Plot results
    plt.figure(figsize=(10, 8))
    
    # Plot magnetization
    plt.subplot(1, 2, 1)
    for T, data in results.items():
        plt.errorbar(T, data['avg_mag'], yerr=data['avg_mag_std'], fmt='o', label=f'T={T:.2f}')
        
    plt.axvline(Tc, color='r', linestyle='--', label='Tc')
    plt.xlabel('Temperature')
    plt.ylabel('Average Magnetization')
    plt.title('Magnetization vs Temperature')
    plt.legend()
    plt.grid()
    
    # Plot correlation functions
    plt.subplot(1, 2, 2)
    for T, data in results.items():
        plt.plot(data['dists'], data['corrs'], 'o-', label=f'T={T:.2f}')
    plt.xlabel('Distance r')
    plt.ylabel('Correlation C(r)')
    plt.title('Spin-Spin Correlation Function')
    plt.legend()
    plt.grid()
    plt.show()

# Run the analysis
analyze_critical_behavior()