#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimizer.py
2D sweep over mass ratio and scavenger particle size.
Generates contour plots of T95 and Cd_eq across the design space.
"""
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from oxygen_diffusion.materials import Rgas, Dd0, Qd, Ds0, Qs, Csat_dirty, tantalum_solubility

# ---------------------------
# Chemical potential
# ---------------------------
def mu(C: float, Csat: float) -> float:
    C = max(C, 1e-6)
    return np.log(C / Csat)

# ---------------------------
# Single simulation
# ---------------------------
def simulate(
    mass_ratio: float,
    Rs: float,
    T: float,
    Cd0: float,
    Cs0: float,
    mass_dirty: float,
    Rd: float,
    t_end: float,
    T95_max: float,
) -> tuple[float, float, bool]:
    """
    Run a single diffusion simulation.

    Returns:
        Cd_eq: equilibrium oxygen in dirty powder (ppm)
        t95: time to 95% equilibration (hours)
        converged: whether simulation reached equilibrium within t_end
    """
    Csat_scav = tantalum_solubility(T)
    Dd = Dd0 * np.exp(-Qd / (Rgas * T))
    Ds = Ds0 * np.exp(-Qs / (Rgas * T))
    mass_scav = mass_dirty / mass_ratio
    tau = Rd**2 / Dd + Rs**2 / Ds

    def rhs(t: float, y: NDArray) -> list[float]:
        Cd, Cs = y
        flux = (mu(Cd, Csat_dirty) - mu(Cs, Csat_scav)) / tau
        return [-flux, (mass_dirty / mass_scav) * flux]

    sol = solve_ivp(rhs, [0, t_end], [Cd0, Cs0], max_step=1000, rtol=1e-6, atol=1e-9)

    Cd = sol.y[0]
    Cd_eq = Cd[-1]
    target = Cd_eq + 0.05 * (Cd0 - Cd_eq)
    t95 = sol.t[np.argmin(np.abs(Cd - target))] / 3600
    converged = t95 / (t_end / 3600) < 0.9

    return Cd_eq, t95, converged

# ---------------------------
# 2D sweep
# ---------------------------
def run_sweep(
    T: float = 1200 + 273.15,
    Cd0: float = 700.0,
    Cs0: float = 50.0,
    mass_dirty: float = 1.0,
    Rd: float = 18e-6,
    t_end: float = 2e6,
    T95_max: float = 500.0,
    n_points: int = 15,
) -> None:

    mass_ratio_values: NDArray = np.linspace(0.05, 0.5, n_points)
    Rs_values: NDArray = np.linspace(5e-6, 100e-6, n_points)

    T95_grid = np.zeros((len(Rs_values), len(mass_ratio_values)))
    Cdeq_grid = np.zeros((len(Rs_values), len(mass_ratio_values)))

    print("Running 2D sweep...")
    for i, Rs in enumerate(Rs_values):
        for j, mr in enumerate(mass_ratio_values):
            Cd_eq, t95, converged = simulate(
                mr, Rs, T, Cd0, Cs0, mass_dirty, Rd, t_end, T95_max
            )
            T95_grid[i, j] = t95
            Cdeq_grid[i, j] = Cd_eq
            if not converged:
                print(f"  WARNING: Rs={Rs*1e6:.1f} um, ratio={mr:.3f} did not converge")

    print("Done.")
    print(f"T95 range:   {T95_grid.min():.1f} to {T95_grid.max():.1f} hours")
    print(f"Cd_eq range: {Cdeq_grid.min():.1f} to {Cdeq_grid.max():.1f} ppm")

    Rs_um = Rs_values * 1e6
    MR, RS = np.meshgrid(mass_ratio_values, Rs_um)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cf1 = axes[0].contourf(MR, RS, T95_grid, levels=20, cmap='plasma')
    cs1 = axes[0].contour(MR, RS, T95_grid, levels=[T95_max], colors='white', linewidths=2)
    axes[0].clabel(cs1, fmt={T95_max: f'{T95_max}h limit'}, fontsize=9)
    plt.colorbar(cf1, ax=axes[0], label='T95 (hours)')
    axes[0].set_xlabel('Mass ratio m_dirty / m_scav')
    axes[0].set_ylabel('Scavenger radius (µm)')
    axes[0].set_title('Time to 95% equilibration (hours)')

    cf2 = axes[1].contourf(MR, RS, Cdeq_grid, levels=20, cmap='viridis')
    cs2 = axes[1].contour(MR, RS, Cdeq_grid, levels=[50], colors='white', linewidths=2)
    axes[1].clabel(cs2, fmt='50 ppm target', fontsize=9)
    plt.colorbar(cf2, ax=axes[1], label='Cd_eq (ppm)')
    axes[1].set_xlabel('Mass ratio m_dirty / m_scav')
    axes[1].set_ylabel('Scavenger radius (µm)')
    axes[1].set_title('Equilibrium dirty powder oxygen (ppm)')

    plt.suptitle(f'T={T-273.15:.0f}°C  |  Rd=18µm  |  Cd0={Cd0:.0f} ppm', fontsize=12)
    plt.tight_layout()
    plt.savefig('optimizer_2d.png', dpi=150)
    plt.show()

    Dd = Dd0 * np.exp(-Qd / (Rgas * T))
    print(f"Dd = {Dd:.2e} m^2/s")
    print(f"tau_d at Rd=18um = {Rd**2/Dd:.1f} s = {Rd**2/Dd/3600:.1f} hours")

if __name__ == "__main__":
    run_sweep()