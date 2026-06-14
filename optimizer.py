#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimizer.py
2D sweep over mass ratio and scavenger particle size.
Generates contour plots of T95 and Cd_eq across the design space.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from materials import Rgas, Dd0, Qd, Ds0, Qs, Csat_dirty, tantalum_solubility

# ---------------------------
# Fixed conditions
# ---------------------------
T       = 1400 + 273.15  # K
Cd0     = 700.0          # wt ppm
Cs0     = 50.0           # wt ppm
mass_dirty = 1.0
Rd      = 18e-6          # m (36 um diameter, D50 target)
t_end   = 2e6            # s (~277 hours)
T95_max = 500            # hours, practical ceiling

Csat_scav = tantalum_solubility(T)
Dd = Dd0 * np.exp(-Qd / (Rgas * T))

# ---------------------------
# Sweep ranges
# ---------------------------
mass_ratio_values = np.linspace(0.05, 0.5, 15)   # m_dirty / m_scav
Rs_values         = np.linspace(5e-6, 100e-6, 15) # m, scavenger radius

# ---------------------------
# Chemical potential
# ---------------------------
def mu(C, Csat):
    C = max(C, 1e-6)
    return np.log(C / Csat)

# ---------------------------
# Single simulation
# ---------------------------
def simulate(mass_ratio, Rs):
    mass_scav = mass_dirty / mass_ratio
    Ds  = Ds0 * np.exp(-Qs / (Rgas * T))
    tau = Rd**2 / Dd + Rs**2 / Ds

    def rhs(t, y):
        Cd, Cs = y
        flux = (mu(Cd, Csat_dirty) - mu(Cs, Csat_scav)) / tau
        return [-flux, (mass_dirty / mass_scav) * flux]

    sol = solve_ivp(rhs, [0, t_end], [Cd0, Cs0],
                    max_step=1000, rtol=1e-6, atol=1e-9)

    Cd    = sol.y[0]
    Cd_eq = Cd[-1]
    target = Cd_eq + 0.05 * (Cd0 - Cd_eq)
    t95   = sol.t[np.argmin(np.abs(Cd - target))] / 3600  # hours

    converged = t95 / (t_end / 3600) < 0.9

    return Cd_eq, t95, converged

# ---------------------------
# 2D grid sweep
# ---------------------------
T95_grid   = np.zeros((len(Rs_values), len(mass_ratio_values)))
Cdeq_grid  = np.zeros((len(Rs_values), len(mass_ratio_values)))

print("Running 2D sweep...")
for i, Rs in enumerate(Rs_values):
    for j, mr in enumerate(mass_ratio_values):
        Cd_eq, t95, converged = simulate(mr, Rs)
        T95_grid[i, j]  = t95
        Cdeq_grid[i, j] = Cd_eq
        if not converged:
            print(f"  WARNING: Rs={Rs*1e6:.1f} um, ratio={mr:.3f} did not converge")

print("Done.")
print(f"T95 range: {T95_grid.min():.1f} to {T95_grid.max():.1f} hours")
print(f"Cd_eq range: {Cdeq_grid.min():.1f} to {Cdeq_grid.max():.1f} ppm")

# ---------------------------
# Axis labels
# ---------------------------
Rs_um      = Rs_values * 1e6           # um for plotting
ratio_axis = mass_ratio_values         # m_dirty / m_scav
MR, RS     = np.meshgrid(ratio_axis, Rs_um)

# ---------------------------
# Plot T95 contour
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# T95
cf1 = axes[0].contourf(MR, RS, T95_grid, levels=20, cmap='plasma')
cs1 = axes[0].contour(MR, RS, T95_grid,
                      levels=[T95_max], colors='white', linewidths=2)
axes[0].clabel(cs1, fmt={T95_max: f'{T95_max}h limit'}, fontsize=9)
plt.colorbar(cf1, ax=axes[0], label='T95 (hours)')
axes[0].set_xlabel('Mass ratio m_dirty / m_scav')
axes[0].set_ylabel('Scavenger radius (µm)')
axes[0].set_title('Time to 95% equilibration (hours)')

# Cd_eq
cf2 = axes[1].contourf(MR, RS, Cdeq_grid, levels=20, cmap='viridis')
cs2 = axes[1].contour(MR, RS, Cdeq_grid,
                      levels=[50], colors='white', linewidths=2)
axes[1].clabel(cs2, fmt='50 ppm target', fontsize=9)
plt.colorbar(cf2, ax=axes[1], label='Cd_eq (ppm)')
axes[1].set_xlabel('Mass ratio m_dirty / m_scav')
axes[1].set_ylabel('Scavenger radius (µm)')
axes[1].set_title('Equilibrium dirty powder oxygen (ppm)')

plt.suptitle(f'T={T-273.15:.0f}°C  |  Rd=18µm  |  Cd0={Cd0:.0f} ppm', fontsize=12)
plt.tight_layout()
plt.savefig('optimizer_2d.png', dpi=150)
plt.show()

print(f"Dd = {Dd:.2e} m^2/s")
print(f"tau_d at Rd=18um = {Rd**2/Dd:.1f} s = {Rd**2/Dd/3600:.1f} hours")
