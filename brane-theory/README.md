# Brane Theory Sims

The goal of this repository was to naively learn something about brane theory. Put in a few words, a universe where the Friedmann equations are modified by a factor next to the energy density in order to explain the universe as a 5-dimensional brane (with the brane concept from string theory).

In this directory you can find:
1. `distances.py`: compares the redshift evolution of distances (proper, luminosity and angular) between standard $\Lambda$CDM model and brane theory (assuming standard values for brane-theory-specific parameters).
2. `simulation.py`: computes the friedmann equations for the brane theory scenario plotting the energy density term and the Hubble parameter both in function of redshift for a range of brane-theory-specific parameters.

### brane-theory-specific parameters

The first Friedmann equation is:

$$ H^2 = \left(\frac{\dot{a}}{a}\right)^2 = \frac{8\pi G}{3} \rho - \frac{k}{a^2} + \frac{\Lambda}{3}.$$

For the RSII brane world cosmology (Randall-Sundrum), this equation gets modified to:

$$ H^2 = \frac{8\pi G_4}{3} \rho \left(1+\frac{\rho}{2\lambda}\right)-\frac{k}{a^2}+\frac{\Lambda_4}{3}+\frac{\mathcal{E}}{a^4}$$

1. **multiplicative correction factor:** at low energies $\rho<<\lambda$ and the term becomes the previous standard cosmology, at high energies $\rho>>\lambda$ and thus $H^2propto \rho^2$ with a much faster early universe expansion.

2. **Dark radiation term:** This originates from the projection of the 5D Weyl tensor onto the brane. This radiation scales exactly like known radition ($\propto a^4$) but it is not composed of ordinary neutrinos and photons. This term is usually set to $0$ (like in our case) or very much constrained by BBN and CMB data.

3. Also note that in comparison with the previous equation, in the brane world cosmology we explicitly indicate the use of 4-dim gravitational ($G_4$) and cosmological ($\Lambda_4$) constants, to avoid confusions with brane-related 5-dimensional terms.