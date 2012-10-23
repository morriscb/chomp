### This is a example python script for creating and writing the values of an 
### angular correlation function from the import of chomp down to writing of the
### correlation. This example uses the correlation induced from galaxy-galaxy
### magnification as the example, however any correlation currently available
### or created by the user (assuming it conforms to the template class) could
### be used.

### import all the modules of CHOMP. This assumes that the user has implemented
### their own inherited objects. If the default classes are desired the only
### imports required are correlation, kernel, and halo. However, if the default
### objects are used additional set_cosmology commands are required if a non
### WMAP7 cosmology is desired.
import correlation
import cosmology
import halo
import hod
import kernel
import mass_function
import numpy

degToRad = numpy.pi/180.0 # define conversion from degrees to radians

### Specify a dictionary defining the properties of a cosmology. Details on
### on each variable can be found in defaults.py
cosmo_dict = {
    "omega_m0": 0.3 - 4.15e-5/0.7**2,
    "omega_b0": 0.046,
    "omega_l0": 0.7,
    "omega_r0": 4.15e-5/0.7**2,
    "cmb_temp": 2.726,
    "h"       : 0.7,
    "sigma_8" : 0.800,
    "n_scalar": 0.960,
    "w0"      : -1.0,
    "wa"      : 0.0
    }

### Initialize the two cosmology objects with at the desired redshifts and
### with the desired cosmological values. Both types of cosmology objects are
### required as some aspects of the code (halo, mass_function) are calculated
### for a single redshift. Others (kernel, correlation) are calculated for a 
### range of redshifts. 
cosmo_single = cosmology.SingleEpoch(redshift=0.0, cosmo_dict=cosmo_dict,
                                 with_bao=False)
cosmo_multi = cosmology.MultiEpoch(z_min=0.0, z_max=5.0, cosmo_dict=cosmo_dict,
                               with_bao=False)

### Define the parameters of halos for the halo model. Change these if you have
### different, favorite values. (Note: alpha != -1 [NFW] not currently 
### implemented and if you like the code fast I wouldn't recommend changing it)
halo_dict = {
    "stq": 0.3,
    "st_little_a": 0.707,
    "c0": 9.,
    "beta": -0.13,
    "alpha": -1
    }

### Load up a Sheth & Tormen mass function object for use in halo. We set the
### redshift to 0 initially, however, later when we run the correlation it will
### be re-initialized to z!=0.
mass = mass_function.MassFunction(redshift=0.0, cosmo_single_epoch=cosmo_single,
                                  halo_dict=halo_dict)

### Initialize the hod object defining how galaxies populate halos. Values used
### in this HOD are from Zehavi et al. 2011 with parameter assumptions from 
### Wake et al. 2011.
sdss_hod = hod.HODZheng(M_min=10**12.14, sigma=0.15, M_0=10**12.14,
                        M_1p=10**13.43, alpha=1.0)

### Initialize the halo object with the mass function and single epoch 
### cosmology implementation is from Seljak2000.
halo_model = halo.Halo(redshift=0.0, input_hod=sdss_hod,
                       cosmo_single_epoch=cosmo_single)


### From this point we have fully defined our cosmology and halo model.
### The next step is defining the redshift distributions and appropriate 
### window functions.

### Below we define our foreground lenses and background source distributions 
### needed for projecting our power spectrum from halo onto the sky. We use
### the functional form of a magnitude limited sample for the lens and for the
### sources we use a Gaussian with mean z=1.0. Other options could be used here,
### see kernel.py for other implemented distributions.
lens_dist = kernel.dNdzMagLim(0.0, 2.0, 2.0, 0.3, 2.0)
source_dist = kernel.dNdzGaussian(0.0, 2.0, 1.0, 0.2)

### Now we need to create the appropriate window functions that will allow us 
### to project the power spectrum from halo. Currently these come in two
### varieties. The first is WindowFunctionGalaxy which defines the dn/dchi
### distribution of galaxies. If one is interested in the clustering of two(one)
### galaxy populations this is correlation should be used. The second is
### WindowFunctionConvergence which defines the lensing kernel weighted
### distribution. Cosmic shear and magnification studies should use 2 of 
### these. For the galaxy-galaxy magnification which we are trying to compute we
### need both a galaxy window function (for the lenses) and a convergence
### window function (for the sources).
lens_window = kernel.WindowFunctionGalaxy(lens_dist, cosmo_multi)
source_window = kernel.WindowFunctionConvergence(source_dist, cosmo_multi)
lens_window.write('test_lens_window.ascii')
source_window.write('test_source_window.ascii')

### Now we need to create the kernel object that will compute the projection
### over redshift for us and also compute the peak redshift sensitivity of the
### kernel which is used to evaluate the halo model. Vales of note for
### initializing the kernel is that we need to know what value of k*theta we
### are going to integrate over in our correlation. The safe bet is setting
### the limits to k_min*theta_min - k_max*theta_max where k_min and k_max are
### set in the code as 0.001 and 100.0 respectively.
con_kernel = kernel.Kernel(ktheta_min=0.001*0.001*degToRad,
                           ktheta_max=100.0*1.0*degToRad,
                           window_function_a=lens_window,
                           window_function_b=source_window,
                           cosmo_multi_epoch=cosmo_multi)
con_kernel.write('test_kernel.ascii')

### Finally we define and run our correlation function, writing the results out
### to test_corr.ascii. Correlation does the job of defining the k space
### integral for a given theta. It also takes responsibility for setting the
### halo model object redshift to that of the peak kernel redshift. It also
### convenient allows for the setting of both the kernel and halo model
### cosmologies through the set_cosmology method. Note that like the kernel
### module, cosmology takes an input as radians.
corr = correlation.Correlation(theta_min_deg=0.001,
                               theta_max_deg=1.0,
                               input_kernel=con_kernel,
                               input_halo=halo_model,
                               powSpec='power_gm')
corr.compute_correlation()
corr.write('test_corr.ascii')

### and done, to make this a proper magnification correlation though, the user
### will have to multiply the output wtheta by 2.

### If you want to make this a script that could MCMCed create all of the
### objects as shown here and then in the MCMC loop call corr.set_cosmology
### and corr.set_hod (in this case) to change the cosmology/HOD and recompute.
