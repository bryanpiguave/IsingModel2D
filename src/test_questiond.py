import numpy as np
from numba import njit
import os
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

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

def simulate_for_temperature(beta, L, nequilibrium, naverage):
    """Simulate for a single temperature point"""
    # Pre-calculate probabilities
    prob = np.array([np.exp(-4 * beta), np.exp(-8 * beta)])
    
    # Initial state - all spins up
    S = np.ones((L, L), dtype=np.int8)
    
    # Thermalization
    for _ in range(nequilibrium):
        S, _ = metropolis(S, prob, L)
    
    # Measurement
    m = np.zeros(naverage)
    m[0] = np.mean(S)
    
    MMacro2 = (m[0] * L**2)**2
    m_avg = abs(m[0])
    M2_avg = MMacro2
    
    # Running averages
    for n in range(1, naverage):
        S, dm = metropolis(S, prob, L)
        m[n] = m[n-1] + dm
        
        # Update running averages
        m_avg = (n * m_avg + abs(m[n])) / (n + 1)
        current_MMacro2 = (m[n] * L**2)**2
        M2_avg = (n * M2_avg + current_MMacro2) / (n + 1)
    
    # Calculate results for this temperature
    M = m_avg
    M2 = M2_avg
    chi = beta * (M2 - (M * L**2)**2) / L**2
    
    return M, M2, chi

def simulate_ising_joblib(L=30, nequilibrium=2000, naverage=20000):
    """Main parallel simulation function using joblib"""
    # Temperature parameters
    beta_min = 1/1.5
    beta_max = 1/3.5
    n_temps = 50
    beta = np.linspace(beta_min, beta_max, n_temps)
    T = 1.0 / beta
    
    # Initialize arrays for results
    M = np.zeros(n_temps)
    M2 = np.zeros(n_temps)
    chi = np.zeros(n_temps)
    
    # Determine number of CPUs to use
    num_cores = multiprocessing.cpu_count()
    n_jobs = min(n_temps, max(1, num_cores - 1))  # Leave one core free
    
    # Run simulations in parallel with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_for_temperature)(b, L, nequilibrium, naverage)
        for b in tqdm(beta, desc=f"L={L}", position=0)
    )
    
    # Unpack results
    for i, (m, m2, c) in enumerate(results):
        M[i] = m
        M2[i] = m2
        chi[i] = c
    
    return T, M, M2, chi

def save(L, T, M, M2, chi):
    """Save data and create plots"""
    data = pd.DataFrame({
        'Temperature': T,
        'Average Magnetization': M,
        'Average Squared Magnetization': M2,
        'Susceptibility': chi,
        'L': L
    })
    
    os.makedirs('testdata', exist_ok=True)
    data.to_csv(f'testdata/ising_simulation_L{L}.csv', index=False)

if __name__ == "__main__":
    L_vector = np.array([10, 20, 50, 100, 1000])
    nequilibrium = 2000
    naverage = 20000
    
    for L in tqdm(L_vector, desc="Overall progress"):
        print(f"\nRunning simulation for L={L}...")
        T, M, M2, chi = simulate_ising_joblib(L, nequilibrium, naverage)
        save(L, T, M, M2, chi)