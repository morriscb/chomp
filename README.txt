================================================================================
CHOMP:
Cosmology and HalO Model Python code
version: Beta 3.0 (3/10/13)

Authors:
Christopher Morrison
Ryan Scranton
Michael Schneider
================================================================================

================================================================================
1. Introduction
================================================================================

CHOMP is an object oriented python code for predicting power spectra and 
correlation functions designed to be flexible accommodate new theoretical models
as the field of cosmology and large-scale structure progresses.

It currently implements several different models of the dark matter power
spectrum from Seljak et al. 00 to HaloFit using the parameterization from 
Takahashi et al. 12. Users are encouraged to implement their own preferred
models.

================================================================================
2. Installation
================================================================================

The minimum requirements for the code are:

python2.7
numpy
scipy
matplotlib is recommended for plotting but not required

Installing the code is as simple as adding the source directory to your
PYTHONPATH. Testing the code is as simple as running the command

"python unit_test.py"

================================================================================
3. Modules
================================================================================

For current details on each model see their respective .py files. For an example
on implementing the code see example_script.py.

================================================================================
3.1 cosmology
================================================================================

The base module for all calculations in chomp. Currently has enough flexibility
to implement a w(z)CDM + Curvature model. Uses a the linear power spectrum of
matter as derived in Eisenstein & Hu 99. The possibility exists to extend this
module to use the transfer function from CAMB to allow for the inclusion of
massive neutrinos, 3+ neutrino species, etc.

================================================================================
3.2 mass_function
================================================================================

Module defining the number of halos as a function fo mass and their halo bias
as a function of mass. Defaults the Sheth & Torman99 mass function but has a
mass function from Tinker et al. 2010 currently implemented. 

The Definition of the mass limits are defined dynamically as a function of
redshift and cosmology by enforcing a constant space in nu where nu from 0.1 to
50.0 where nu(M)=(delta_c/sigma(M))^2. The integrals of the halo model are in
nu space so this keeps those limits roughly constant. At z=0 for a WMAP7
cosmology, the nu limits correspond to 10^8 - 10^16 M_solar/h.


================================================================================
3.3 hod
================================================================================

This module defines the halo occupation distribution for calculating both the
galaxy-galaxy power spectrum and the galaxy-matter spectrum. Generic class
takes as input a dictionary of values for the HOD. Currently implemented are:

HODZheng: 5 parameter model from Zheng et al. 2007
HODMand: 2 parameter model from Mandelbaum et al. 2005. Definitions here are
    slightly different, but translate to the same parameters if properly
    normalized.

================================================================================
3.4 halo
================================================================================

Module defining the different halo models/fits. The default class is from
Seljak00. Extensions to this include a model with halo exclusion and a halo fit
implementation with parameters from Takahashi et al. 2012. For the default model
agreement with nbody sims is (do numbers yet. Likely a factor of 2 and better if
you change halo parameters.)

================================================================================
3.5 kernel
================================================================================

Convenience classes defining defining three separate and necessary pieces of
functionality.

The first is the definition of a redshift distribution (aka dn/dz). Functions
implemented are a boxcar, Gaussian, magnitude limited sample, and an
interpolated function from an array of redshifts and dn/dz values.

The second defines window functions to Limber's equation for both projecting the
halo function in l space and real space. Implemented functions are for both
galaxy cluster and convergence allowing for combinations of
clustering-clustering, galaxy-lenings, lensing-lensing.

Third is the kernel itself that projects the power spectrum onto the sky.
Implemented kernels are for 2-D matter clustering or differential matter
clustering al la galaxy-galaxy lensing (Put simply projections that follow the
J0 Bessel function or J2).

================================================================================
3.6 correlation
================================================================================



================================================================================
3.7 covariance
================================================================================

Refs: 
Seljak et al. 2000
Sheth & Tormen 1999
(Others I still need to write down)
