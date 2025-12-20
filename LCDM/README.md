# LCDM directory

The code in this directory is my adaption from @gcanasherrera's _CalculadoraCosmològica_ code to Python.

The aim of this small project was to learn about the computation tricks in Python for cosmological functions as well as to continue to learn about the physics per se.

To proceeding for the directory was to copy and read the original notes and read throughout. Then I went on to write the `friedmannequations.py` script which computes
1. the scale factor (a),
2. the curvature parameter (k),
3. the redshift (z),
4. the comoving (Mpc),
5. proper (Mpc),
6. angular (Mpc),
7. and luminosity distances (Mpc),
8. the light travel time (Gyr),
9. big bang to $z$ travel time (Gyr),
10. and the universe time (Gyr)

using the pre-established parameters in `parameters.py`. Results are saved in `cosmological_distances.csv` (data sub-directory). The script `plottingFriedmann.py` was used to prepare the data reading part and to test the `friedmannequations.py` script by creating a simple plot of all 3 distances with redshift in the horizontal axis.

The file `figures.ipynb` is a notebook that generates the visualisations of the data to be found in the `figures/` sub-directory. The figures created are the following:
1. [distances (all) vs redshift](figures/distances_vs_redshift.png)
2. [distances vs Universe time](figures/distances_vs_universetime.png)
3. [Scale factor vs light travel time](figures/scalefactor_vs_lighttraveltime.png)
4. [Time vs redshift](figures/time_vs_redshift.png)
5. [Scale factor vs Universe time (for different cosmologies)](figures/scalefactor_timesincebigbang_evolution.png)