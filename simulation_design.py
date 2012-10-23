#!/usr/bin/env python
# encoding: utf-8
"""
simulation_design.py

Created by Michael Schneider on 2012-10-12
"""

import argparse
import sys
# import os.path
import numpy
import pandas
#import pylab as pl

# CHOMP modules
import defaults
import cosmology
import mass_function
import hod
import halo
import perturbation_spectra
import halo_trispectrum
import kernel
import covariance

import logging

# Print log messages to screen:
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
logging.basicConfig(filename='log_simulation_design.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

help_message = '''
Usage: python simulation_design.py
'''

# Default list of cosmological parameters for the simulation design.
#  Each entry gives the central and min/max values.
params_default = {
    'omega_m0': [0.278, 0.268, 0.288],
    'sigma_8': [0.811, 0.711, 0.911]
}


# ----- Helper class for error reporting
class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def random_lhs(n, k):
    """
    Random Latin Hypercube Sample (non-optimized).

    Algorithm copied from randomLHS function in the R package, 'lhs'
    Rob Carnell (2012). lhs: Latin Hypercube Samples. R package version 0.10.
      http://CRAN.R-project.org/package=lhs

    Args:
        n: The number of simulation design points
        k: The number of variables
    """
    P = numpy.zeros((n, k), dtype='float64')
    for i in xrange(k):
        P[:, i] = numpy.random.permutation(range(n))
    P = P + numpy.random.uniform(size=(n * k)).reshape((n, k))
    return P / n


def init_chomp_covariance():
    cosmo_single = cosmology.SingleEpoch(redshift=0.0,
                                         cosmo_dict=defaults.default_cosmo_dict,
                                         with_bao=False)
    cosmo_multi = cosmology.MultiEpoch(z_min=0.0,
                                       z_max=5.0,
                                       cosmo_dict=defaults.default_cosmo_dict,
                                       with_bao=False)
    mass = mass_function.MassFunction(redshift=0.0,
                                      cosmo_single_epoch=cosmo_single,
                                      halo_dict=defaults.default_halo_dict)
    sdss_hod = hod.HODZheng(M_min=10 ** 12.14,
                            sigma=0.15,
                            M_0=10 ** 12.14,
                            M_1p=10 ** 13.43,
                            alpha=1.0)
    halo_model = halo.Halo(redshift=0.0, input_hod=sdss_hod,
                           cosmo_single_epoch=cosmo_single,
                           mass_func=mass,
                           halo_dict=defaults.default_halo_dict)
    pert = perturbation_spectra.PerturbationTheory(redshift=0.0,
                                                   cosmo_single_epoch=cosmo_single)
    halo_tri = halo_trispectrum.HaloTrispectrumOneHalo(redshift=0.0,
                                                       single_epoch_cosmo=cosmo_single,
                                                       mass_func_second=mass,
                                                       perturbation=pert,
                                                       halo_dict=defaults.default_halo_dict)
    source_dist = kernel.dNdzGaussian(0.0, 2.0, 1.0, 0.2)
    source_window = kernel.WindowFunctionConvergence(source_dist, cosmo_multi)
    wa1 = source_window
    wa2 = source_window
    wb1 = source_window
    wb2 = source_window
    kern = kernel.KernelCovariance(ktheta_min=0.1,
                                   ktheta_max=10.,
                                   window_function_a1=wa1,
                                   window_function_a2=wa2,
                                   window_function_b1=wb1,
                                   window_function_b2=wb2,
                                   cosmo_multi_epoch=cosmo_multi,
                                   force_quad=False)
    cov = covariance.Covariance(theta_min_deg=0.1,
                                theta_max_deg=2.0,
                                bins_per_decade=2,
                                survey_area_deg2=40.e3,
                                nongaussian_cov=False,
                                input_kernel_covariance=kern,
                                input_halo=halo_model,
                                input_halo_trispectrum=halo_tri)
    return cov


class SimulationDesign(object):
    """
    Create and execute a simulation design for emulator construction.
    """
    def __init__(self, n_des, params=params_default):
        self.n_des = n_des
        self.params = pandas.DataFrame(params, index=['center', 'min', 'max'])

    def generate_design_points(self):
        logging.debug('Generating design for %d points' % self.n_des)
        des_points = random_lhs(self.n_des, self.params.ndim)
        self.lhs = numpy.copy(des_points)
        # Rescale the design points from the [0,1] interval
        # to the parameter ranges in self.params
        for ndx, val in enumerate(self.params):
            des_points[:, ndx] *= (self.params[val]['max'] - self.params[val]['min'])
            des_points[:, ndx] += self.params[val]['min']
        self.des_points = pandas.DataFrame(des_points)
        self.des_points.columns = self.params.columns
        return None

    def plot_design(self):
        import pylab as pl
        from pandas.tools.plotting import scatter_matrix
        scatter_matrix(self.des_points, grid=True, diagonal='hist')
        pl.show()

    def run_design(self):
        cov = init_chomp_covariance()
        for params in self.des_points.iterrows():
            # Set cosmological parameters to new values in cov
            cosmo_dict = defaults.default_cosmo_dict
            param_names = params[1].index
            for i, p in enumerate(param_names):
                cosmo_dict[p] = params[1][i]
            cov.set_cosmology(cosmo_dict)
            des_cov = cov.get_covariance()
        return des_cov


def main():
    try:
        parser = argparse.ArgumentParser(
            description='Create and execute a simulation design for emulator construction.')
        parser.add_argument('n_des', help='Number of design points')

        args = parser.parse_args()
        logging.debug('----- Program started -----')
        simdes = SimulationDesign(int(args.n_des))
        simdes.generate_design_points()
        # simdes.plot_design()
        simdes.run_design()
        logging.debug('----- Program finished -----')
    except Usage, err:
        print >> sys.stderr, sys.argv[0].split("/")[-1] + ": " + str(err.msg)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
