#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimizer.py
Sweep over mass ratio to find equilibrium oxygen content.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from materials import Rgas, Dd0, Qd, Ds0, Qs, Csat_dirty, tantalum_solubility

# ---------------------------
# Fixed conditions
# ---------------------------
T = 1400 + 273.15  # K
Cd0 = 700.0
Cs0 = 50.0
mass_dirty = 1.0
Rd = 50e-6 #radius of dirty powder particle (m)
Rs = 500e-6

Csat_scav = tantalum_solubility(T)
Dd = Dd0 * np.exp(-Qd / (Rgas * T))
Ds = Ds0 * np.exp(-Qs / (Rgas * T))
tau = Rd**2 / Dd + Rs**2 / Ds

def mu(C, Csat):
    C = max(C, 1e-6)
    return np.log(C / Csat)

# ---------------------------
# Sweep
# ---------------------------
mass_ratio_values = np.logspace(-2, 0, 12)  # m_dirty / m_scav

def simulate_ratio(ratio, t_end=1e7):
    mass_scav = mass_dirty / ratio
    def rhs(t, y):
        Cd, Cs = y
        flux = (mu(Cd, Csat_dirty) - mu(Cs, Csat_scav)) / tau
        return [-flux, (mass_dirty / mass_scav) * flux]
    sol = solve_ivp(rhs, [0, t_end], [Cd0, Cs0], max_step=1000, rtol=1e-6, atol=1e-9)
    Cd = sol.y[0]
    Cd_eq = Cd[-1]
    t95 = sol.t[np.argmin(np.abs(Cd - (Cd_eq + 0.05 * (Cd0 - Cd_eq))))]
    if t95 / t_end > 0.9:
        print(f"  WARNING: ratio={ratio:.3f} may not have converged")
    return {'ratio': ratio, 'mass_scav': mass_scav, 'Cd_eq': Cd_eq,
            'Cs_eq': sol.y[1][-1], 't95_h': t95 / 3600}

results = [simulate_ratio(r) for r in mass_ratio_values]

print('Sweep over m_dirty / m_scav:')
for res in results:
    print(f"  ratio={res['ratio']:.3f}, m_scav={res['mass_scav']:.2f}, "
          f"Cd_eq={res['Cd_eq']:.2f} ppm, T95={res['t95_h']:.2f} h")

# ---------------------------
# Plot
# ---------------------------
ratio_values = np.array([res['ratio'] for res in results])
Cd_eq_values = np.array([res['Cd_eq'] for res in results])

plt.figure(figsize=(8, 5))
plt.plot(ratio_values, Cd_eq_values, 'o-', color='tab:blue')
plt.axhline(50, color='red', linestyle='--', label='Target 50 ppm')
plt.xscale('log')
plt.xlabel('Mass ratio m_dirty / m_scav')
plt.ylabel('Cd_eq (ppm)')
plt.title('Equilibrium oxygen content vs mass ratio')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('Cd_eq_vs_mass_ratio.png', dpi=150)
plt.show()