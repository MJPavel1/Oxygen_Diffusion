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
mass_scav = 10.0         # change ratio here
Rd = 50e-6              # m
Rs = 50e-6              # m

# Diffusion constants
# Diffusion constants, dirty powder
Dd0 = 1e-7
Qs  = 180e3

# Diffusion constants, scavenger pure Ta
Ds0 = np.exp(-13.72) #(https://doi.org/10.1016/0001-6160(86)90240-3)
Qd  = 115.5e3 #J/mol

Rgas = 8.314 #J/mol/K

# Oxygen solubility in Ta (up to 900C: https://apps.dtic.mil/sti/tr/pdf/ADA382682.pdf)
#1100C - 1800C : https://doi.org/10.1016/0022-5088(72)90062-8
def tantalum_solubility(T):
    if T < 1373:
        return 10 ** (4.130 - 1279/T)
    else:
        return 3.322 * T - 1833.6




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
# ODE system
# ---------------------------

def rhs(t,y):

    Cd, Cs = y

    mud = mu(Cd,Csat_dirty)
    mus = mu(Cs,Csat_scav)

    flux = (mud-mus)/tau

    dCd = -flux
    dCs = (mass_dirty/mass_scav)*flux

    return [dCd,dCs]

# ---------------------------
# Solve
# ---------------------------

sol = solve_ivp(
    rhs,
    [0,1e7],
    [Cd0,Cs0],
    max_step=1000
)

# ---------------------------
# Find 95% equilibration time
# ---------------------------

Cd = sol.y[0]

Cd_eq = Cd[-1]

target = Cd_eq + 0.05*(Cd0-Cd_eq)

idx = np.argmin(np.abs(Cd-target))

t95 = sol.t[idx]

print(f"T95 = {t95/3600:.2f} hours")
print(f"Scavenger solubility: {Csat_scav:.2f} ppm")

# ---------------------------
# Plot
# ---------------------------

plt.plot(sol.t/3600,Cd,label='Dirty Powder')
plt.plot(sol.t/3600,sol.y[1],label='Scavenger')
plt.xlim(-5,100)
plt.ylim(0,Csat_dirty+0.1*Csat_dirty)
plt.axhline(50, color='gray', linestyle='--', label='Target 50 ppm')
plt.xlabel('Time (hours)')
plt.ylabel('Oxygen (ppm)')
plt.legend()
plt.show()