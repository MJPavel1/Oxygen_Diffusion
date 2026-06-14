#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:46:02 2026

@author: michaelpavel
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------------
# User inputs
# ---------------------------

T = 1400 + 273.15       # K

Cd0 = 700.0            # ppm oxygen in dirty powder
Cs0 = 50.0              # ppm oxygen in Ta

mass_dirty = 1.0
mass_ratio_values = np.logspace(-2, 0, 12)  # sweep m_dirty/m_scav from 0.01 to 1.0
Rd = 50e-6              # m
Rs = 500e-6              # m

# Diffusion constants
# Diffusion constants, dirty powder
Dd0 = 1e-7
Qd  = 200e3

# Diffusion constants, scavenger pure Ta
Ds0 = np.exp(-13.72) #(https://doi.org/10.1016/0001-6160(86)90240-3)
Qs  = 115.5e3 #J/mol

Rgas = 8.314 #J/mol/K

# Oxygen solubility in Ta 600 - 1800C : https://doi.org/10.1016/0022-5088(72)90062-8
def tantalum_solubility(T):
    # wt ppm, unified Arrhenius fit to Ta-O phase boundary data
    # log(Csat) = 9.7657 - 2564.86/T  (T in Kelvin)
    return np.exp(9.7657 - 2564.86/T)




# oxygen affinity / solubility PPM

Csat_dirty = 1000 #assume 1000 ppm for dirty powder
Csat_scav  = tantalum_solubility(T)

# ---------------------------
# Diffusivities
# ---------------------------

Dd = Dd0*np.exp(-Qd/(Rgas*T))
Ds = Ds0*np.exp(-Qs/(Rgas*T))

# characteristic diffusion times
tau_d = Rd**2/Dd
tau_s = Rs**2/Ds

tau = tau_d + tau_s

# ---------------------------
# Chemical potential model
# ---------------------------

def mu(C, Csat):
    C = max(C,1e-6)
    return np.log(C/Csat)

# ---------------------------
# Sweep over mass ratio
# ---------------------------


def simulate_ratio(ratio, t_end=1e7):
    """Return equilibrium concentration and 95% equilibration time for one mass ratio."""
    mass_scav = mass_dirty / ratio
    Csat_scav = tantalum_solubility(T)

    def rhs_local(t, y):
        Cd, Cs = y
        mud = mu(Cd, Csat_dirty)
        mus = mu(Cs, Csat_scav)
        flux = (mud - mus) / tau
        dCd = -flux
        dCs = (mass_dirty / mass_scav) * flux
        return [dCd, dCs]

    sol = solve_ivp(
        rhs_local,
        [0, t_end],
        [Cd0, Cs0],
        max_step=1000,
        rtol=1e-6,
        atol=1e-9,
    )

    Cd = sol.y[0]
    Cd_eq = Cd[-1]
    target = Cd_eq + 0.05 * (Cd0 - Cd_eq)
    idx = np.argmin(np.abs(Cd - target))
    t95 = sol.t[idx]

    return {
        'ratio': ratio,
        'mass_scav': mass_scav,
        'Cd_eq': Cd_eq,
        'Cs_eq': sol.y[1][-1],
        't95_h': t95 / 3600,
        'sol': sol,
    }


results = [simulate_ratio(r) for r in mass_ratio_values]
Cd_eq_values = np.array([res['Cd_eq'] for res in results])
ratio_values = np.array([res['ratio'] for res in results])

print('Sweep over m_dirty / m_scav:')
for res in results:
    print(
        f"ratio={res['ratio']:.3f}, m_scav={res['mass_scav']:.2f}, "
        f"Cd_eq={res['Cd_eq']:.2f} ppm, T95={res['t95_h']:.2f} h"
    )

# ---------------------------
# Plot Cd_eq versus mass ratio
# ---------------------------

plt.figure(figsize=(8, 5))
plt.plot(ratio_values, Cd_eq_values, 'o-', color='tab:blue')
plt.xscale('log')
plt.xlabel('Mass ratio m_dirty / m_scav')
plt.ylabel('Cd_eq (ppm)')
plt.title('Equilibrium oxygen content vs mass ratio')
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('Cd_eq_vs_mass_ratio.png', dpi=150)
if 'agg' not in plt.get_backend().lower():
    plt.show()
else:
    plt.close()
