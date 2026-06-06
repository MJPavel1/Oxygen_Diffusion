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

Cd0 = 1000.0            # ppm oxygen in dirty powder
Cs0 = 50.0              # ppm oxygen in scavenger

mass_dirty = 1.0
mass_scav = 5.0         # change ratio here
Rd = 50e-6              # m
Rs = 500e-6              # m

Dd0 = 1e-7
Qs  = 180e3

Ds0 = 1e-7
Qd  = 200e3

Rgas = 8.314

# oxygen affinity / solubility
Csat_dirty = 5000
Csat_scav  = 50000

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

# ---------------------------
# Plot
# ---------------------------

plt.plot(sol.t/3600,Cd,label='Dirty Powder')
plt.plot(sol.t/3600,sol.y[1],label='Scavenger')
plt.xlabel('Time (hours)')
plt.ylabel('Oxygen (ppm)')
plt.legend()
plt.show()