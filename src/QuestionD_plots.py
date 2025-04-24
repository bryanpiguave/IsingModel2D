import pandas as pd
import matplotlib.pyplot as plt 
from plot_aesthetics import axis_fontdict, title_fontdict


df = pd.read_csv('outputs/question_d.csv')
# Plotting Magnetization
plt.figure(figsize=(12, 10))
for L in df['L'].unique():
    data = df[df['L'] == L]
    plt.plot(data['T'], data['Magnetization'], marker='o', linestyle='-', markersize=10, label=f'L={L}')
plt.xlabel('Temperature (T)', fontdict=axis_fontdict)
plt.ylabel('Average Absolute Magnetization per Site',fontdict=axis_fontdict)
plt.title('Magnetization vs. Temperature for Different Lattice Sizes',fontdict=title_fontdict)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/magnetization')

# Plotting Susceptibility
plt.figure(figsize=(12, 10))
for L in df['L'].unique():
    data = df[df['L'] == L]
    plt.plot(data['T'], data['Susceptibility'], marker='o', linestyle='-', markersize=3, label=f'L={L}')
plt.xlabel('Temperature (T)',fontdict=axis_fontdict)
plt.ylabel('Susceptibility per Site',fontdict=axis_fontdict)
plt.title('Susceptibility vs. Temperature for Different Lattice Sizes',fontdict=title_fontdict)
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig('outputs/susceptibility')