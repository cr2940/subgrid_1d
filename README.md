# subgrid_1d
1D SWE with barrier 

This repository contains codes from Jiao Li's hbox repo, but with modified solver ideas (under function called barrier_solver inside shallow_1D_redistribute_wave.py), in the case of wall set inside a cell.

The codes feed into Pyclaw in order to solve the SWE.

sill_h_box_wave.py sets up a simulation to be fed into Pyclaw for solving.
