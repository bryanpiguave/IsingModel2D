import numpy as np
from ising_model import Ising2D
import os
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing

def calculate_magnetization_and_susceptibility(L, T, num_steps=50, equilibration_steps=10000):
    """
    Runs simulation and calculates the average absolute magnetization per site
    and magnetic susceptibility per site.

    Args:
        L (int): Lattice size.
        T (float): Temperature.
        num_steps (int): Total simulation steps.
        equilibration_steps (int): Steps to discard for equilibration.

    Returns:
        tuple: (avg_abs_mag_per_site, susceptibility_per_site)
    """
    if T <= 0:
        beta = np.inf
    else:
        beta = 1.0 / T

    model = Ising2D(L=L, initial_state='random')
    N = model.N
    _, magnetizations_norm = model.run_simulation(T, num_steps, equilibration_steps)

    magnetizations_total = [m * N for m in magnetizations_norm]
    M_vals = np.array(magnetizations_total)
    M_abs_vals = np.abs(M_vals)

    avg_abs_mag_per_site = np.mean(M_abs_vals) / N

    M_sq_avg = np.mean(M_vals**2)
    M_avg_sq = np.mean(M_vals)**2
    susceptibility_per_site = (beta / N) * (M_sq_avg - M_avg_sq) if T > 0 else 0.0

    return avg_abs_mag_per_site, susceptibility_per_site

def process_parameters(args):
    L, T = args
    avg_abs_mag, susc = calculate_magnetization_and_susceptibility(L, T)
    return {'L': L, 'T': T, 'Magnetization': avg_abs_mag, 'Susceptibility': susc}

if __name__ == '__main__':
    Length_vector = np.array([10, 20, 50, 100, 1000])
    Temperature_vector = np.linspace(1.5, 3.5, 20) # Fine grid around Tc_exact ≈ 2.269

    params = [(L, T) for L in Length_vector for T in Temperature_vector]
    
    os.makedirs(name='outputs', exist_ok=True)
    
    num_cores = max(1, int(multiprocessing.cpu_count() * 0.75))
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_parameters, params)

    
    df = pd.DataFrame(results)
    df.to_csv('outputs/question_d.csv')