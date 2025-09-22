#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:33:37 2025

@author: jawetz
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
import os

plt.style.use('ggplot') 

sns.set_context("paper")
sns.set_style("whitegrid")

# Load files from 1_step directory
npy_files_1 = [f for f in os.listdir('./1_step') if f.endswith('.npy')]
data_1 = {}
for file in npy_files_1:
    data_1[file] = np.load(os.path.join('./1_step', file))
print("Loaded files from 1_step:", list(data_1.keys()))

# Load files from 12_step directory
npy_files_12 = [f for f in os.listdir('./12_step') if f.endswith('.npy')]
data_12 = {}
for file in npy_files_12:
    data_12[file] = np.load(os.path.join('./12_step', file))
print("Loaded files from 12_step:", list(data_12.keys()))


plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.figsize': (8, 6),
    'axes.grid': False,
    'legend.frameon': False,
    'lines.markersize': 6,
})


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

time_1 = data_1['time_series.npy']
mid_quantum_1 = data_1['mid_boundary.npy']
mid_classical_1 = data_1['mid_boundary_c.npy']
left_quantum_1 = data_1['left_boundary.npy']
left_classical_1 = data_1['left_boundary_c.npy']
q_interp_1 = data_1['q_interp.npy']
c_interp_1 = data_1['c_interp.npy']
a_interp_1 = data_1['a_interp.npy']
rho_q_1 = data_1["rho_q.npy"]
rho_c_1 = data_1["rho_c.npy"]
rho_a_1 = data_1["rho_a.npy"]
q_interp_12 = data_12['q_interp.npy']
c_interp_12 = data_12['c_interp.npy']
a_interp_12 = data_12['a_interp.npy']
rho_q_12 = data_12["rho_q.npy"]
rho_c_12 = data_12["rho_c.npy"]
rho_a_12 = data_12["rho_a.npy"]


# First figure - Mid boundary
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

N = 10 

ax1.plot(time_1, mid_classical_1[:, 0], 'r-', label='Classical')
ax1.plot(time_1[::N], mid_classical_1[::N, 0], 'ro')
ax1.plot(time_1, mid_classical_1[:, 1], 'r-')
ax1.plot(time_1[::N], mid_classical_1[::N, 1], 'ro')
ax1.plot(time_1, mid_classical_1[:, 2], 'r-')
ax1.plot(time_1[::N], mid_classical_1[::N, 2], 'ro')
ax1.plot(time_1, mid_quantum_1[:, 0], 'b--', label='Quantum')
ax1.plot(time_1[::N], mid_quantum_1[::N, 0], 'bx')
ax1.plot(time_1, mid_quantum_1[:, 1], 'b--')
ax1.plot(time_1[::N], mid_quantum_1[::N, 1], 'bx')
ax1.plot(time_1, mid_quantum_1[:, 2], 'b--')
ax1.plot(time_1[::N], mid_quantum_1[::N, 2], 'bx')

ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Interfacial LBM Values')
ax1.legend()

plt.tight_layout()
plt.savefig('mid_boundary_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Second figure - Left boundary
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

# Left boundary plots
ax2.plot(time_1, left_classical_1[:, 0], 'r-', label='Classical')
ax2.plot(time_1[::N], left_classical_1[::N, 0], 'ro')
ax2.plot(time_1, left_classical_1[:, 1], 'r-')
ax2.plot(time_1[::N], left_classical_1[::N, 1], 'ro')
ax2.plot(time_1, left_classical_1[:, 2], 'r-')
ax2.plot(time_1[::N], left_classical_1[::N, 2], 'ro')
ax2.plot(time_1, left_quantum_1[:, 0], 'b--', label='Quantum')
ax2.plot(time_1[::N], left_quantum_1[::N, 0], 'bx')
ax2.plot(time_1, left_quantum_1[:, 1], 'b--')
ax2.plot(time_1[::N], left_quantum_1[::N, 1], 'bx')
ax2.plot(time_1, left_quantum_1[:, 2], 'b--')
ax2.plot(time_1[::N], left_quantum_1[::N, 2], 'bx')

ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Left boundary LBM Values')
ax2.legend()

plt.tight_layout()
plt.savefig('left_boundary_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Third figure - Temperature Profile
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)

# Temperature profile plots
ax3.plot(q_interp_1[-1,:], rho_q_1[-1,:], 'bs', label='QLBM (1)', markerfacecolor='none')
ax3.plot(q_interp_12[-1,:], rho_q_12[-1,:], 'os', label='QLBM (12)', markerfacecolor='none')
ax3.plot(c_interp_1[-1,:], rho_c_1[-1,:], 'ro', label='Classical LBM', markerfacecolor='none')
ax3.plot(rho_a_1[-1,:], a_interp_1[-1,:], 'k-', label='Theory')

ax3.set_xlabel('Lattice Site')
ax3.set_ylabel('Temperature')
ax3.legend()

plt.tight_layout()
plt.savefig('temperature_profile.pdf', dpi=300, bbox_inches='tight')
plt.show()

## Fourth figure - Temperature Profile, 12 steps
#fig4 = plt.figure()
#ax4 = fig4.add_subplot(111)

# # Temperature profile plots
# ax4.plot(q_interp_12[-1,:], rho_q_12[-1,:], 'bs', label='QLBM', markerfacecolor='none')
# ax4.plot(c_interp_12[-1,:], rho_c_12[-1,:], 'ro', label='Classical LBM', markerfacecolor='none')
# ax4.plot(rho_a_12[-1,:], a_interp_12[-1,:], 'k-', label='Theory')

# ax4.set_xlabel('Lattice Site')
# ax4.set_ylabel('Temperature')
# ax4.legend()

# plt.tight_layout()
# plt.savefig('temperature_profile_12step.pdf', dpi=300, bbox_inches='tight')
# plt.show()


# Define colors
quantum_color = 'blue'
classical_color = 'red'
analytical_color = 'green'
# First figure (1_step)
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)

# Get data from 1_step
time_series = data_1['time_series.npy']
temp_liq = data_1['temp_liq.npy']
temp_sol = data_1['temp_sol.npy']
temp_liq_c = data_1['temp_liq_c.npy']
temp_sol_c = data_1['temp_sol_c.npy']
temp_liq_a = data_1['temp_liq_a.npy']
temp_sol_a = data_1['temp_sol_a.npy']

ax5.plot(time_series, temp_liq, color=quantum_color, linestyle='-', label='QLBM')
ax5.plot(time_series, temp_sol, color=quantum_color, linestyle='--')

ax5.plot(time_series, temp_liq_c, color=classical_color, linestyle='-', label='Classical LBM')
ax5.plot(time_series, temp_sol_c, color=classical_color, linestyle='--')

ax5.plot(time_series, temp_liq_a, color=analytical_color, linestyle='-', label='Theory')
ax5.plot(time_series, temp_sol_a, color=analytical_color, linestyle='--')

ax5.set_xlabel('Time')
ax5.set_ylabel('Temperature')
ax5.legend()

plt.tight_layout()
plt.savefig('temperature_evolution_1step.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Second figure (12_step)
fig6 = plt.figure()
ax6 = fig6.add_subplot(111)

# Get data from 12_step
time_series = data_12['time_series.npy']
temp_liq = data_12['temp_liq.npy']
temp_sol = data_12['temp_sol.npy']
temp_liq_c = data_12['temp_liq_c.npy']
temp_sol_c = data_12['temp_sol_c.npy']
temp_liq_a = data_12['temp_liq_a.npy']
temp_sol_a = data_12['temp_sol_a.npy']

ax6.plot(time_series, temp_liq, color=quantum_color, linestyle='-', label='QLBM')
ax6.plot(time_series, temp_sol, color=quantum_color, linestyle='--')

ax6.plot(time_series, temp_liq_c, color=classical_color, linestyle='-', label='Classical LBM')
ax6.plot(time_series, temp_sol_c, color=classical_color, linestyle='--')

ax6.plot(time_series, temp_liq_a, color=analytical_color, linestyle='-', label='Theory')
ax6.plot(time_series, temp_sol_a, color=analytical_color, linestyle='--')

ax6.set_xlabel('Time')
ax6.set_ylabel('Temperature')
ax6.legend()

plt.tight_layout()
plt.savefig('temperature_evolution_12step.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Calculate RMS error for 1-step
time_1 = data_1['time_series.npy']
rho_q_1 = data_1['rho_q.npy']
rho_c_1 = data_1['rho_c.npy']

# Calculate RMS error for 12-step
time_12 = data_12['time_series.npy']
rho_q_12 = data_12['rho_q.npy']
rho_c_12 = data_12['rho_c.npy']

# Calculate RMS error at each time step
rms_error_1 = np.sqrt(np.mean((rho_q_1 - rho_c_1)**2, axis=1))
rms_error_12 = np.sqrt(np.mean((rho_q_12 - rho_c_12)**2, axis=1))

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111)

# Plot both RMS errors
ax.plot(time_1, rms_error_1, 'g-', label='1-step')
ax.plot(time_12, rms_error_12, 'o-', label='12-step')

ax.set_xlabel('Time')
ax.set_ylabel('RMS Error')
ax.legend()

plt.tight_layout()
plt.savefig('rms_error_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Print some statistics
print(f"Maximum RMS error (1-step): {np.max(rms_error_1):.6f}")
print(f"Maximum RMS error (12-step): {np.max(rms_error_12):.6f}")
print(f"Average RMS error (1-step): {np.mean(rms_error_1):.6f}")
print(f"Average RMS error (12-step): {np.mean(rms_error_12):.6f}")

