This script calculates the diffusion profile of an impurity solute across a diffusion couple defined as an ODE-IVP 
Specifically for two powder particles in contact with one dirty and one clean scavenger particle.
Material diffusivity constants are defined in materials.py (D0, Ea, solubility limits etc).
The optimizer.py script runs a two dimensional sweep across mass ratio (scavenger sink to solute reservoir) and size ratio to determine 
when a threshold target solute concentration in the dirty powder may be reached.
