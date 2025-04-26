import pandas as pd
import matplotlib.pyplot as plt 
from plot_aesthetics import axis_fontdict, title_fontdict
import numpy as np
import glob

list_of_csv =glob.glob(f'data/ising_simulation_L*.csv')
main_df = pd.DataFrame()
for csv_file in list_of_csv:
    df = pd.read_csv(csv_file)
    df['Temperature'] = df['Temperature'].astype(float)
    df['Temperature'] = df['Temperature']/2.27
    df['Average Magnetization'] = np.abs(df['Average Magnetization'])
    main_df = pd.concat((main_df,df))

# Plotting Magnetization
plt.figure(figsize=(12, 10))
for L in main_df['L'].unique():
    data = main_df[main_df['L'] == L]
    plt.scatter(data['Temperature'], data['Average Magnetization'], label=f'L={L}', marker='o', s=100)
plt.xlabel('T/Tc', fontdict=axis_fontdict)
plt.ylabel('Average Absolute Magnetization per Site',fontdict=axis_fontdict)
plt.title('Magnetization vs. Temperature for Different Lattice Sizes',fontdict=title_fontdict)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/question_d_magnetization.png')

# Plotting Susceptibility
plt.figure(figsize=(12, 10))
for L in main_df['L'].unique():
    data = main_df[main_df['L'] == L]
    plt.plot(data['Temperature'], data['Susceptibility'], 'o-', label=f'L={L}', markersize = 10)
plt.xlabel('T/Tc', fontdict=axis_fontdict)
plt.ylabel('Susceptibility per Site',fontdict=axis_fontdict)
plt.title('Susceptibility vs. Temperature for Different Lattice Sizes',fontdict=title_fontdict)
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig('outputs/question_d_susceptibility.png')