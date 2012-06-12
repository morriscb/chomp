"""
This module expresses the default values for the cosmology, halo model, and
the precision of all modules within the code.
"""
### parameters specifilying a cosmology.
default_cosmo_dict = {
    "omega_m0": 0.278, ### total matter desnity at z=0
    "omega_b0": 0.046, ### baryon density at z=0
    "omega_l0": 0.722, ### dark energy density at z=0
    "omega_r0": 4.15e-5/0.7**2, ### radiation density at z=0
    "cmb_temp": 2.726, ### temperature of the CMB in K at z=0
    "h"       : 0.7, ### Hubbles constant at z=0 normalized to 1/100 km/s/Mpc
    "sigma_8" : 0.811, ### overdensity of matter at 8.0 Mpc/h
    "n_scalar": 0.960, ### large k slope of the power spetcurm
    "w0"      : -1.0, ### dark energy equation of state at z=0
    "wa"      : 0.0 ### varrianing dark energy eqauation of state. At a=0 the 
                    ### value is w0 + wa.
    }

### Default parameters specifying a halo.
default_halo_dict = {
    "stq": 0.3,
    "st_little_a": 0.707,
    "c0": 9.,
    "beta": -0.13,
    "alpha": -1 ### Halo mass profile slope. [NFW = -1]
    }

### default precision parameters defining how many steps splines are evaluated
### for as well as the convergence of the romberg integration used.
### If a user defines new derived classes it is recommended that they test if
### these values are still relavent to their modules. As a rull of thumb for
### the module returns: if a module has a quickly varying function use more
### n_points; if a module returns values of the order of the precision increase
### this variable. For highly discontinous functions it is recommened that,
### instead of changing these variables, the integration method quad in
### scipy is used.
default_precision = {
    "corr_npoints": 25,
    "corr_precision":1.48e-8,
    "cosmo_npoints": 50,
    "cosmo_precision": 1.48e-8,
    "dNdz_precision": 1.48e-8,
    "halo_npoints": 50,
    "halo_precision": 1.48-8,
    "kernel_npoints": 100,
    "kernel_precision": 1.48e-16,
    "kernel_limit": 100, ### If the varible force_quad is set in the Kernel 
                         ### class this value sets the limit for the quad
                         ### integration
    "kernel_bessel_limit": 32, ### Defines how many zeros before cutting off the
                               ### bessel function in kernel.py
    "mass_npoints": 50,
    "mass_precision": 1.48e-8,
    "window_npoints": 50,
    "window_precision": 1.48e-8
    }
