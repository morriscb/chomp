#!/usr/bin/python
import argparse
import numpy

### chomp modules we are currently loading.
import correlation
import cosmology
import defaults
import halo
import kernel

"""
Wrapper script for computing the shear-shear power spectrum for different
redshifts, cosmologies and Fourier modes. The script uses the latest
parameterization of HaloFit from Takahashi et al. 2012 to compute the matter
power spectrum. Runs default values in 6.5 seconds, less if you request fewer
l modes.

For usage instructions run "python shear_shear_specrum.py --help".
"""

if __name__ == '__main__':

    ### Read in command line values. For a list from the command line, type
    ### python shear_shear_spectrum.py --help.
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_name', default = 'test.ascii',
                        type = str, help='File name to write the results to. '
                        'Output file is 2 column: l C_shear_shear(l)')
    parser.add_argument('-z', '--redshift', type = float, default=1.0,
                        help='Source plane redshift')
    parser.add_argument('-d', '--n_z', default = None, type = str, 
                        help='Specify a file to read dn/dz from.'
                        'Overrides --redshift option. File should be '
                        'two columns: z dn/dz')
    parser.add_argument('-n', '--n_l', default = 1000, type = int, 
                        help='Specify the number of points to calculate the '
                        'shear spectrum')
    parser.add_argument('-l', '--l_low', default = 10, type = float, 
                        help='Lowest l value at which to calculate the shear '
                        'spectrum')
    parser.add_argument('-u', '--l_high', default = 10**5, type = float, 
                        help='Highest l value at which to calculate the shear '
                        'spectrum')
    parser.add_argument('-f', '--l_file', default = None, type = str, 
                        help='File specifying the points at which to '
                        'calculate the shear spectrum. Overrides options n_l, '
                        ' l_low, l_high')
    parser.add_argument('-c', '--cosmology', type = str, default=None,
                        help='File specifying the cosmology.\n'
                        'Formating should be:\n'
                        'omega_m0 omega_b0 omega_l0 h sigma_8 n_s w0 wa')
    parser.add_argument('-v', '--verbose', default = 0,
                        type = int, )
    args = parser.parse_args()

    ### If there is not input cosmology file we use the default WMAP7 as defined
    ### in defaults.py
    if args.cosmology is None:
        c_dict = defaults.default_cosmo_dict
    else:
        cosmo_data = numpy.loadtxt(args.cosmology)
        ### Set the cosmology by overwritting the global cosmology dictionary
        ### in chomp.
        defaults.default_cosmo_dict = {
            "omega_m0": cosmo_data[0] - 4.15e-5/cosmo_data[3]**2, ### total matter density at z=0
            "omega_b0": cosmo_data[1], ### baryon density at z=0
            "omega_l0": cosmo_data[2], ### dark energy density at z=0
            "omega_r0": 4.15e-5/cosmo_data[3]**2, ### radiation density at z=0
            "cmb_temp": 2.726, ### temperature of the CMB in K at z=0
            "h"       : cosmo_data[3], ### Hubble's constant at z=0 normalized to 
                                       ### 1/100 km/s/Mpc
            "sigma_8" : cosmo_data[4], ### over-density of matter at 8.0 Mpc/h
            "n_scalar": cosmo_data[5], ### large k slope of the power spectrum
            "w0"      : cosmo_data[6], ### dark energy equation of state at z=0
            "wa"      : cosmo_data[7]  ### varying dark energy equation of state. 
                                       ### At a=0 the value is w0 + wa.
            }

    ### check to see if we have input a redshift distribution instead of a
    ### single redshift value. If a redshift distribution is input the single
    ### value is overridden
    if args.n_z is None:
        dist = kernel.dNdzGaussian(0.001, 5.0, args.redshift, 0.0001)
    else:
        dndz = numpy.loadtxt(args.n_z)
        dist = kernel.dNdzInterpolation(dndz[:,0], dndz[:,1])
        
    ### Initialize the window functions and convinence classes for integrating
    ### over the window function distribution in redshift. 
    window = kernel.WindowFunctionConvergence(dist)
    kern = kernel.Kernel(0.001*0.001*numpy.pi/180.0, 1.0*100.0*numpy.pi/180.0,
                         window, window)

    ### Initialize the halo model(fit) object. If you would like to use a
    ### different model, uncomment the line above. I recommend asking me about
    ### them as the halo model can be a bit finicky. (as you are already aware
    ### of.
    h = halo.HaloFit()
    # h = halo.HaloExclusion(cosmo_single_epoch=cosmo_single)
    
    ### Initialize our correlation object for integrating the projected power
    ### spectrum
    corr = correlation.CorrelationFourier(1, 1000000, input_kernel=kern, 
                                          input_halo=h, powSpec='power_mm')
    
    ### define the l points at which to compute the correlation values
    if args.l_file is None:
        l_array = numpy.logspace(numpy.log10(args.l_low),
                                 numpy.log10(args.l_high), args.n_l)
        power_array = numpy.empty((1000, 2))
    else:
        l_array = numpy.loadtxt(args.l_file)
        power_array = numpy.empty((len(l_array), 2))
    for idx, l in enumerate(l_array):
        power_array[idx, 0] = l
        power_array[idx, 1] = corr.correlation(l)
    
    ### output the correlation (really a power spectrum) to the file specified.
    numpy.savetxt(args.output_name, power_array)



