import numpy as np
from ising_model import Ising2D
import os 
import matplotlib.pyplot as plt
import pandas as pd 

def calculate_magnetization_and_susceptibility(L, T, num_steps=1000, equilibration_steps=1000):
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
               - avg_abs_mag_per_site: <|M|>/N
               - susceptibility_per_site: χ/N = (β / N) * (<M^2> - <M>^2)
    """
    if T <= 0: beta = np.inf
    else: beta = 1.0 / T

    model = Ising2D(L=L, initial_state='random') # Start fresh for each T
    N = model.N
    _, magnetizations_norm = model.run_simulation(T, num_steps, equilibration_steps)

    # Convert normalized magnetizations (M/N) back to total M for calculations
    magnetizations_total = [m * N for m in magnetizations_norm]

    # Use measurements *after* equilibration
    M_vals = np.array(magnetizations_total) # Total M values
    M_abs_vals = np.abs(M_vals)             # Absolute |M| values

    # Average absolute magnetization per site: <|M|> / N
    avg_abs_mag_per_site = np.mean(M_abs_vals) / N

    # Susceptibility per site: χ/N = (β / N) * (<M^2> - <M>^2)
    # Note: Using <M>^2 is standard. If M fluctuates around 0 (high T), <M> ≈ 0.
    # If M fluctuates around non-zero M_0 (low T), <M> ≈ M_0.
    M_sq_avg = np.mean(M_vals**2)
    M_avg_sq = np.mean(M_vals)**2
    susceptibility_per_site = (beta / N) * (M_sq_avg - M_avg_sq) if T > 0 else 0.0

    return avg_abs_mag_per_site, susceptibility_per_site

Length_vector = np.array([10,20,50,100,1000])
Temperature_vector = np.linspace(0.5, 1.7,10)
results = []

os.makedirs(name='output',exist_ok=True)
for l_value in Length_vector:
    for temp_value in Temperature_vector:
        avg_abs_mag_per_site, susceptibility_per_site = calculate_magnetization_and_susceptibility(L=l_value,T=Temperature_vector)

        results.append({'L': l_value, 'T': temp_value, 'Magnetization': avg_abs_mag_per_site, 'Susceptibility': susceptibility_per_site})

df = pd.DataFrame(results)

# Plotting Magnetization
plt.figure(figsize=(10, 6))
for L in df['L'].unique():
    data = df[df['L'] == L]
    plt.plot(data['T'], data['Magnetization'], marker='o', linestyle='-', markersize=3, label=f'L={L}')
plt.xlabel('Temperature (T)')
plt.ylabel('Average Absolute Magnetization per Site')
plt.title('Magnetization vs. Temperature for Different Lattice Sizes')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Susceptibility
plt.figure(figsize=(10, 6))
for L in df['L'].unique():
    data = df[df['L'] == L]
    plt.plot(data['T'], data['Susceptibility'], marker='o', linestyle='-', markersize=3, label=f'L={L}')
plt.xlabel('Temperature (T)')
plt.ylabel('Susceptibility per Site')
plt.title('Susceptibility vs. Temperature for Different Lattice Sizes')
plt.legend()
plt.grid(True)
plt.show()
