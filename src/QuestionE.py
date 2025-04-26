import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit
from plot_aesthetics import axis_fontdict, title_fontdict

# Add xlabel setting by default


"""
Use finite size scaling to determine the critical temperature Tc in the limit L → ∞
and the critical exponent β defined by ⟨|ML|⟩ ∼ |T − Tc|β.
Compare with Onsager’s solution.

"""
def power_law(x, a, beta, Tc):
    """Power law function for fitting near critical point"""
    return a * np.abs(x - Tc)**beta

def finite_size_scaling_analysis():
    """
    Perform finite size scaling analysis to determine Tc and critical exponent β
    Compare with Onsager's exact solution (Tc = 2.269)
    """
    # Load all simulation data
    list_of_csv = glob.glob('data/ising_simulation_L*.csv')
    dfs = []
    
    for csv_file in list_of_csv:
        df = pd.read_csv(csv_file)
        # Extract lattice size from filename
        L = int(csv_file.split('_L')[-1].split('.')[0])
        df['L'] = L
        dfs.append(df)
    
    main_df = pd.concat(dfs)
    
    # Normalize temperature by Onsager's Tc (for comparison)
    main_df['T_normalized'] = main_df['Temperature'] / 2.269
    
    # Prepare figure
    plt.figure(figsize=(12, 8))
    
    # Plot magnetization curves for different lattice sizes
    plt.subplot(2, 2, 1)
    for L, group in main_df.groupby('L'):
        plt.plot(group['Temperature'], np.abs(group['Average Magnetization']), 
                'o-', markersize=3, label=f'L={L}')
    plt.xlabel('Temperature (T)',fontdict=axis_fontdict)
    plt.ylabel('Average Magnetization |M|',fontdict=axis_fontdict)
    plt.title('Magnetization vs Temperature')
    plt.legend()
    plt.grid(True)
    
    # Estimate critical temperature using Binder cumulant method
    plt.subplot(2, 2, 2)
    for L, group in main_df.groupby('L'):
        # Calculate Binder cumulant U_L = 1 - <M^4>/(3<M^2>^2)
        U_L = 1 - group['Average Squared Magnetization']**2 / (3 * group['Average Magnetization']**4)
        plt.plot(group['Temperature'], U_L, 'o-', markersize=3, label=f'L={L}')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Binder Cumulant U_L')
    plt.title('Binder Cumulant vs Temperature')
    plt.legend()
    plt.grid(True)
    
    # Estimate Tc from intersection point 
    Tc_estimate = 2.3  # Initial guess
    print(f"Initial Tc estimate: {Tc_estimate:.3f}")
    
    # Fit power law near critical point for largest L
    largest_L = main_df['L'].max()
    df_largest_L = main_df[main_df['L'] == largest_L]
    
    # Select data points below Tc for fitting
    fit_data = df_largest_L[df_largest_L['Temperature'] < Tc_estimate]
    
    try:
        popt, pcov = curve_fit(power_law, 
                              fit_data['Temperature'], 
                              np.abs(fit_data['Average Magnetization']),
                              p0=[1.0, 0.125, Tc_estimate],
                              bounds=([0, 0.01, 2.0], [10, 0.2, 2.5]))
        
        a, beta, Tc = popt
        perr = np.sqrt(np.diag(pcov))
        
        print(f"Fitted parameters: a = {a:.3f} ± {perr[0]:.3f}")
        print(f"Critical exponent β = {beta:.3f} ± {perr[1]:.3f}")
        print(f"Critical temperature Tc = {Tc:.3f} ± {perr[2]:.3f}")
        print(f"Onsager's Tc = 2.269")
        
        # Plot fitted curve
        T_fit = np.linspace(fit_data['Temperature'].min(), Tc, 100)
        plt.subplot(2, 2, 1)
        plt.plot(T_fit, power_law(T_fit, *popt), 'k--', 
                label=f'Fit: β={beta:.3f}\nTc={Tc:.3f}')
        plt.legend()
        
        # Plot log-log scale for critical exponent
        plt.subplot(2, 2, 3)
        plt.loglog(Tc - fit_data['Temperature'], 
                  np.abs(fit_data['Average Magnetization']), 'o', 
                  label=f'L={largest_L} data')
        plt.loglog(Tc - T_fit, power_law(T_fit, *popt), 'k--',
                  label=f'β={beta:.3f} fit')
        plt.xlabel('log(Tc - T)')
        plt.ylabel('log|M|')
        plt.title('Critical Exponent Analysis')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(Tc - fit_data['Temperature'], 
                  np.abs(fit_data['Average Magnetization']), 'o', 
                  label=f'L={largest_L} data')
        plt.plot(Tc - T_fit, power_law(T_fit, *popt), 'k--',
                  label=f'β={beta:.3f} fit')
        #Plot Onager's solution
        T_onsager = np.linspace(0, 2.269, 100)
        M_onsager = power_law(T_onsager, 1, beta, 2.269)
        plt.plot(2.269 - T_onsager, M_onsager, 'r--', label='Onsager Solution')
        plt.xlabel('(Tc - T)')
        plt.ylabel('|M|')
        plt.title('Critical Exponent Analysis')
        plt.legend()
        plt.grid(True)
        
        
    except Exception as e:
        print(f"Fitting failed: {str(e)}")
    
    plt.tight_layout()
    plt.savefig('outputs/question_e_finite_size_scaling_results.png')


    # Finite size scaling plot optimization for each L
    plt.figure(figsize=(12, 8))
    Tc_values = []
    beta_values = []
    error_temps = []
    error_betas = []
    L_values = []
    Tc_estimate = 2.3  # Initial guess for Tc
    for L, group in main_df.groupby('L'):
        # Fit power law for each L
        try:
            fit_data = group[group['Temperature'] < Tc_estimate]
            popt, pcov = curve_fit(power_law, 
                                  fit_data['Temperature'], 
                                  np.abs(fit_data['Average Magnetization']),
                                  p0=[1.0, 0.125, Tc_estimate],
                                  bounds=([0, 0.01, 2.0], [10, 0.2, 2.5]))
            a, beta, Tc = popt
            Tc_values.append(Tc)
            beta_values.append(beta)
            perr = np.sqrt(np.diag(pcov))
            L_values.append(L)
            error_temps.append(perr[2])
            error_betas.append(perr[1])
            print(f"L={L}: a = {a:.3f} ± {perr[0]:.3f}, β = {beta:.3f} ± {perr[1]:.3f}, Tc = {Tc:.3f} ± {perr[2]:.3f}")

        except Exception as e:
            print(f"Fitting failed for L={L}: {str(e)}")
    # Plot Tc values with error bars
    plt.subplot(2,1, 1)
    plt.errorbar(L_values, Tc_values, yerr=error_temps, fmt='o', markersize=6, label='Fitted Tc')
    plt.axhline(y=2.269, color='r', linestyle='--', label='Onsager Tc')
    plt.legend(fontsize=16)
    plt.xlabel('Lattice Size (L)',fontdict=axis_fontdict)
    plt.ylabel('Critical Temperature Tc',fontdict=axis_fontdict)
    plt.title('Critical Temperature vs Lattice Size', fontdict=title_fontdict)
    plt.grid(True)  

    # Plot critical exponent β
    plt.subplot(2, 1, 2)
    plt.errorbar(L_values, beta_values, yerr=error_betas, fmt='o', markersize=3, label='Fitted β')
    plt.xlabel('Lattice Size (L)',fontdict=axis_fontdict)
    plt.ylabel('Critical Exponent β',fontdict=axis_fontdict)
    plt.title('Critical Exponent vs Lattice Size', fontdict=title_fontdict)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig('outputs/question_e_finite_size_scaling_results_L.png')



if __name__ == "__main__":
    finite_size_scaling_analysis()