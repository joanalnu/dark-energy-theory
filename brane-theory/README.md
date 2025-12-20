# Brane Theory Sims

The goal of this repository was to naively learn something about brane theory. Put in a few words, a universe where the Friedmann equations are modified by a factor next to the energy density in order to explain the universe as a 5-dimensional brane (with the brane concept from string theory).

In this directory you can find:
1. `distances.py`: compares the redshift evolution of distances (proper, luminosity and angular) between standard $\Lambda$CDM model and brane theory (assuming standard values for brane-theory-specific parameters).
2. `simulation.py`: computes the friedmann equations for the brane theory scenario plotting the energy density term and the Hubble parameter both in function of redshift for a range of brane-theory-specific parameters.

### brane-theory-specific parameters

The first Friedmann equation is:

$$\frac{\dot{a}}{a} = \frac{8\pi G}{3} \rho - \frac{k}{a^2} + \frac{\Lambda}{3}$$.

For the brane world cosmology, we multiply a new term to the energy density resulting in

$$ \frac{\dot{a}}{a} = \frac{8 \pi G}{3} \rho \text{TERMS}$$


### Disclaimer

We refer to brane theory or brane world cosmologies when we only are using the RSII (Randall-Sundrum brane world cosmology).