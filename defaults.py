"""
This module expresses the default values for the cosmology, halo model, and
the precision of all modules within the code.
"""
### parameters specifying a cosmology.
default_cosmo_dict = {
    "omega_m0": 0.278 - 4.15e-5/0.7**2, ### total matter density at z=0
    "omega_b0": 0.046, ### baryon density at z=0
    "omega_l0": 0.722, ### dark energy density at z=0
    "omega_r0": 4.15e-5/0.7**2, ### radiation density at z=0
    "cmb_temp": 2.726, ### temperature of the CMB in K at z=0
    "h"       : 0.7, ### Hubble's constant at z=0 normalized to 1/100 km/s/Mpc
    "sigma_8" : 0.811, ### over-density of matter at 8.0 Mpc/h
    "n_scalar": 0.960, ### large k slope of the power spectrum
    "w0"      : -1.0, ### dark energy equation of state at z=0
    "wa"      : 0.0 ### varying dark energy equation of state. At a=0 the 
                    ### value is w0 + wa.
    }

### Default parameters specifying a halo.
default_halo_dict = {
    "stq"        :  0.3,
    "st_little_a":  0.707,
    "c0"         :  9.0,
    "beta"       : -0.13,
    "alpha"      : -1, ### Halo mass profile slope. [NFW = -1]
    "delta_v"    : -1.0 ### over-density for defining. -1 means default behavior of
                    ### redshift dependent over-density defined in NFW97
    }

### Default values for the ZhengHOD class (other models will fail when using
### this definition)
default_hod_dict = {
    "log_M_min": 12.14,
    "sigma"    :  0.15,
    "log_M_0"  : 12.14,
    "log_M_1p" : 13.43,
    "alpha":      1.0
    }

### Default global integration limits for the code.
default_limits = {
    "k_min": 0.001,
    "k_max": 100.0,
    "mass_min": -1, ### If instead of integrating the mass function over a fixed
    "mass_max": -1  ### range of nu a fixed mass range is desired, set these
                    ### limits to control the mass integration range. Setting
                    ### These to hard limits can halve the running time of the
                    ### code at the expense of less integration range consitency
                    ### as a function of redshift and cosmology.
    }

### default precision parameters defining how many steps splines are evaluated
### for as well as the convergence of the Romberg integration used.
### If a user defines new derived classes it is recommended that they test if
### these values are still relevant to their modules. As a rule of thumb for
### the module returns: if a module has a quickly varying function use more
### n_points; if a module returns values of the order of the precision increase
### this variable. For highly discontinuous functions it is recommended that,
### instead of changing these variables, the integration method quad in
### scipy is used.
default_precision = {
    "corr_npoints": 50,
    "corr_precision": 1.48e-6,
    "cosmo_npoints": 50,
    "cosmo_precision": 1.48e-8,
    "dNdz_precision": 1.48e-8,
    "halo_npoints": 50,
    "halo_precision": 1.48e-5, ### The difference between e-4 and e-5 are at the
                               ### 0.1% level. Since this is the main slow down
                               ### in the calculation e-4 can be used to speed
                               ### up the code.
    "halo_limit" : 100,
    "kernel_npoints": 50,
    "kernel_precision": 1.48e-6,
    "kernel_limit": 100, ### If the variable force_quad is set in the Kernel 
                         ### class this value sets the limit for the quad
                         ### integration
    "kernel_bessel_limit": 8, ### Defines how many zeros before cutting off
                              ### the Bessel function integration in kernel.py
    "mass_npoints": 50,
    "mass_precision": 1.48e-8,
    "window_npoints": 100,
    "window_precision": 1.48e-6,
    "global_precision": 1.48e-32, ### Since the code has large range of values
                                  ### from say 1e-10 to 1e10 we don't want to 
                                  ### use absolute tolerances, instead using 
                                  ### relative tolerances to define convergence
                                  ### of our integrands
    "divmax":20                   ### Maximum number of subdivisions for
                                  ### the romberg integration.
    }