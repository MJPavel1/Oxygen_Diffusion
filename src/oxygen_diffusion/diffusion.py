#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusion.py
Single condition oxygen diffusion simulation between dirty and scavenger powder.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from materials import Rgas, Dd0, Qd, Ds0, Qs, Csat_dirty, tantalum_solubility

# ---------------------------
# User inputs
# ---------------------------
T = 1200 + 273.15  # K
Cd0 = 700.0        # wt ppm oxygen in dirty powder
Cs0 = 50.0         # wt ppm oxygen in Ta
mass_dirty = 1.0
mass_scav = 50.0
Rd = 17e-6         # m
Rs = 25e-6        # m

# ---------------------------
# Diffusivities
# ---------------------------
Csat_scav = tantalum_solubility(T)
Dd = Dd0 * np.exp(-Qd / (Rgas * T))
Ds = Ds0 * np.exp(-Qs / (Rgas * T))
tau_d = Rd**2 / Dd
tau_s = Rs**2 / Ds
tau = tau_d + tau_s

# ---------------------------
# Chemical potential model
# ---------------------------
def mu(C, Csat):
    C = max(C, 1e-6)
    return np.log(C / Csat)

# ---------------------------
# ODE system
# ---------------------------
def rhs(t, y):
    Cd, Cs = y
    mud = mu(Cd, Csat_dirty)
    mus = mu(Cs, Csat_scav)
    flux = (mud - mus) / tau
    return [-flux, (mass_dirty / mass_scav) * flux]

# ---------------------------
# Solve
# ---------------------------
sol = solve_ivp(rhs, [0, 1e7], [Cd0, Cs0], max_step=1000, rtol=1e-6, atol=1e-9)

Cd = sol.y[0]
Cd_eq = Cd[-1]
target = Cd_eq + 0.05 * (Cd0 - Cd_eq)
t95 = sol.t[np.argmin(np.abs(Cd - target))]

# Analytical equilibrium
ratio = Csat_scav / Csat_dirty
Cd_eq_analytical = (mass_dirty * Cd0 + mass_scav * Cs0) / (mass_dirty + mass_scav * ratio)
Cs_eq_analytical = Cd_eq_analytical * ratio

print(f"T95            = {t95/3600:.2f} hours")
print(f"Csat_scav      = {Csat_scav:.2f} ppm")
print(f"Cd_eq (solver) = {Cd_eq:.2f} ppm")
print(f"Cs_eq (solver) = {sol.y[1][-1]:.2f} ppm")
print(f"Cd_eq (analytical) = {Cd_eq_analytical:.2f} ppm")
print(f"Cs_eq (analytical) = {Cs_eq_analytical:.2f} ppm")

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(sol.t / 3600, Cd, label='Dirty Powder')
plt.plot(sol.t / 3600, sol.y[1], label='Scavenger')
plt.axhline(50, color='gray', linestyle='--', label='Target 50 ppm')
plt.xlim(-5, 100)
plt.ylim(0, Csat_dirty * 1.1)
plt.xlabel('Time (hours)')
plt.ylabel('Oxygen (ppm)')
plt.legend()
plt.tight_layout()
plt.show()