import numpy as np
from numba import njit
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

@njit(fastmath=True)
def metropolis(S, prob, L):
    """Optimized Metropolis algorithm focusing only on magnetization"""
    dm_total = 0.0
    inv_L2 = 1.0 / (L * L)
    
    for _ in range(L * L):
        i = np.random.randint(L)
        j = np.random.randint(L)
        
        # Calculate nearest neighbors with periodic boundary conditions
        S_nn = (S[(i-1)%L, j] + S[(i+1)%L, j] + 
                S[i, (j-1)%L] + S[i, (j+1)%L])
        
        dE = 2 * S[i, j] * S_nn
        
        if dE <= 0 or np.random.rand() < prob[int(dE/4 - 1)]:
            S[i, j] *= -1
            delta = 2 * S[i, j]
            dm_total += delta * inv_L2
            
    return S, dm_total

def simulate_ising(L=30, nequilibrium=2000, naverage=20000):
    """Main simulation function focusing on magnetization only"""
    # Temperature parameters
    beta_min = 1/1.5
    beta_max = 1/3.5
    n_temps = 50
    beta = np.linspace(beta_min, beta_max, n_temps)
    T = 1.0 / beta
    
    # Initialize arrays for results
    M = np.zeros(n_temps)        # Average magnetization
    M2 = np.zeros(n_temps)       # Average squared magnetization
    chi = np.zeros(n_temps)      # Susceptibility
    
    # Initial state - all spins up
    S = np.ones((L, L), dtype=np.int8)
    
    # Pre-calculate probabilities for all temperatures
    prob_table = np.empty((n_temps, 2))
    for i in range(n_temps):
        prob_table[i, 0] = np.exp(-4 * beta[i])
        prob_table[i, 1] = np.exp(-8 * beta[i])
    
    # Main simulation loop
    for t_idx in range(n_temps):
        current_prob = prob_table[t_idx]
        
        # Thermalization
        for _ in range(nequilibrium):
            S, _ = metropolis(S, current_prob, L)
        
        # Measurement
        m = np.zeros(naverage)
        m[0] = np.mean(S)
        
        MMacro2 = (m[0] * L**2)**2
        m_avg = abs(m[0])
        M2_avg = MMacro2
        
        # Running averages to avoid storing all steps
        for n in range(1, naverage):
            S, dm = metropolis(S, current_prob, L)
            m[n] = m[n-1] + dm
            
            # Update running averages
            m_avg = (n * m_avg + abs(m[n])) / (n + 1)
            
            current_MMacro2 = (m[n] * L**2)**2
            M2_avg = (n * M2_avg + current_MMacro2) / (n + 1)
        
        # Store results
        M[t_idx] = m_avg
        M2[t_idx] = M2_avg
        chi[t_idx] = beta[t_idx] * (M2[t_idx] - (M[t_idx] * L**2)**2) / L**2
    
    return T, M, M2, chi

def save(L, T, M, M2, chi):
    """Save data and create plots"""
    # Save data
    data = pd.DataFrame({
        'Temperature': T,
        'Average Magnetization': M,
        'Average Squared Magnetization': M2,
        'Susceptibility': chi,
        'L': L
    })
    
    os.makedirs('data', exist_ok=True)
    data.to_csv(f'data/ising_simulation_L{L}.csv', index=False)
 

if __name__ == "__main__":
    L = 30  # Lattice size
    nequilibrium = 2000  # Steps to reach equilibrium
    naverage = 20000  # Steps for averaging
    L_vector = np.array([10,20,50,100,1000])
    for L in tqdm(L_vector):
        print(f"Running simulation for L={L}...")
        T, M, M2, chi = simulate_ising(L, nequilibrium, naverage)
        save(L, T, M, M2, chi)