import numpy as np
from ising_model import Ising2D
import os
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
import numpy as np

def calculate_magnetization_and_susceptibility(L, T, num_steps=3, equilibration_steps=10000):
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

    model = Ising2D(L=L, initial_state='random', J=1.0, H=0.00)
    N = model.N
    _, magnetizations = model.run_simulation(T, num_steps, equilibration_steps)

    # Average absolute magnetization per site (your original calculation, which is fine for this quantity)
    M_abs_avg = np.mean(np.abs(magnetizations))
    avg_abs_mag_per_site = M_abs_avg / N

    # Magnetic susceptibility per site (corrected calculation)
    avg_magnetization = np.mean(magnetizations)
    avg_mag_per_site = avg_magnetization / N
    avg_sq_magnetization = np.mean(magnetizations**2)
    avg_sq_mag_per_site = avg_sq_magnetization / N
    susceptibility_per_site = (avg_sq_mag_per_site - avg_mag_per_site**2) / T

    return avg_abs_mag_per_site, susceptibility_per_site

def process_parameters(args):
    L, T = args
    avg_abs_mag, susc = calculate_magnetization_and_susceptibility(L, T)
    return {'L': L, 'T': T, 'Magnetization': avg_abs_mag, 'Susceptibility': susc}

if __name__ == '__main__':
    Length_vector = np.array([10, 20,50, 100, 1000])
    # Original temperature range and number of points
    T_critical_approx = 2.27
    T_start = 1.135 # 0.5*T_critical_approx
    T_end = 3.404 # 1.5*T_critical_approx
    # Create a fine grid around the critical temperature
    T_vector1 = np.linspace(T_start, 2.2,10)
    T_vector2 = np.linspace(2.5, T_end, 10)
    T_vector3 = np.linspace(2.2, 10, 20)
    Temperature_vector = np.concatenate((T_vector1, T_vector2, T_vector3))
    Temperature_vector = np.unique(Temperature_vector)  # Remove duplicates
    params = [(L, T) for L in Length_vector for T in Temperature_vector]
    
    os.makedirs(name='outputs', exist_ok=True)
    
    num_cores = max(1, int(multiprocessing.cpu_count() * 0.75))
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_parameters, params)

    
    df = pd.DataFrame(results)
    df.to_csv('outputs/question_d.csv')