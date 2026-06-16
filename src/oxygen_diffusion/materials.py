#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
materials.py
Shared material constants and property functions for oxygen diffusion model.
"""
import numpy as np

# ---------------------------
# Universal constants
# ---------------------------
Rgas: float = 8.314  # J/mol/K

# ---------------------------
# Dirty powder (W assumption)
# ---------------------------
Dd0: float = 1e-7        # m^2/s
Qd: float  = 150e3       # J/mol
Csat_dirty: float = 1000 # wt ppm, assumed fixed for now

# ---------------------------
# Scavenger (pure Ta)
# ---------------------------
Ds0: float = np.exp(-13.72)  # m^2/s (https://doi.org/10.1016/0001-6160(86)90240-3)
Qs: float  = 115.5e3          # J/mol

def tantalum_solubility(T: float) -> float:
    """
    Oxygen solubility in Ta (wt ppm) as a function of temperature (K).
    Unified Arrhenius fit to Ta-O phase boundary data.
    Source: https://doi.org/10.1016/0022-5088(72)90062-8
    Valid range: ~600-1900 C
    """
    return np.exp(9.7657 - 2564.86 / T)