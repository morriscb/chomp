import cosmology
import correlation
import defaults
import halo
import halo_trispectrum
import hod
import kernel
import mass_function
import perturbation_spectra
import numpy
import unittest

deg_to_rad = numpy.pi/180.0

### In order for the unittests to work correctly, these are the assumed
### precision values of the code.
defaults.default_precision = {
    "corr_npoints": 50,
    "corr_precision":1.48e-6,
    "cosmo_npoints": 50,
    "cosmo_precision": 1.48e-8,
    "dNdz_precision": 1.48e-8,
    "halo_npoints": 50,
    "halo_precision": 1.48e-5, ### This value is mostly due to integrations over
                               ### the HOD. If you are intrested in dark matter
                               ### only, this precision can be increased with
                               ### no major hit to speed.
    "halo_limit" : 100,
    "kernel_npoints": 50,
    "kernel_precision": 1.48e-6,
    "kernel_limit": 100, ### If the variable force_quad is set in the Kernel 
                         ### class this value sets the limit for the quad
                         ### integration
    "kernel_bessel_limit": 8, ### Defines how many zeros before cutting off
                              ### the Bessel function in kernel.py
    "mass_npoints": 50,
    "mass_precision": 1.48e-6,
    "window_npoints": 50,
    "window_precision": 1.48e-6,
    "global_precision": 1.48e-32, ### Since the code has large range of values
                                  ### from say 1e-10 to 1e10 don't want to use
                                  ### absolute tolerances, instead using 
                                  ### relative tolerances to define convergence
                                  ### of our integrands
    "divmax":20
    }

p_dict = {
    "corr":5,
    "cosmo":7,
    "dndz":7,
    "halo":4,
    "kernel":5,
    "mass":7,
    "window":5
    }

### Fix cosmology used in the module in case the user changes the default
c_dict = {
    "omega_m0": 0.3 - 4.15e-5/0.7**2, ### total matter desnity at z=0
    "omega_b0": 0.046, ### baryon density at z=0
    "omega_l0": 0.7, ### dark energy density at z=0
    "omega_r0": 4.15e-5/0.7**2, ### radiation density at z=0
    "cmb_temp": 2.726, ### temperature of the CMB in K at z=0
    "h"       : 0.7, ### Hubble's constant at z=0 normalized to 1/100 km/s/Mpc
    "sigma_8" : 0.8, ### over-density of matter at 8.0 Mpc/h
    "n_scalar": 0.960, ### large k slope of the power spectrum
    "w0"      : -1.0, ### dark energy equation of state at z=0
    "wa"      : 0.0 ### varying dark energy equation of state. At a=0 the 
                    ### value is w0 + wa.
    }

c_dict_2 = {
    "omega_m0": 1.0 - 4.15e-5/0.7**2,
    "omega_b0": 0.046,
    "omega_l0": 0.0,
    "omega_r0": 4.15e-5/0.7**2,
    "cmb_temp": 2.726,
    "h"       : 0.7,
    "sigma_8" : 0.8,
    "n_scalar": 0.960,
    "w0"      : -1.0,
    "wa"      : 0.0
    }

h_dict = {
    "stq": 0.3,
    "st_little_a": 0.707,
    "c0": 9.,
    "beta": -0.13,
    "alpha": -1 ### Halo mass profile slope. [NFW = -1]
    }

h_dict_2 = {
    "stq": 0.5,
    "st_little_a": 0.5,
    "c0": 5.,
    "beta": -0.2,
    "alpha": -1 ### Halo mass profile slope. [NFW = -1]
    }

hod_dict = {
    "log_M_min":12.14,
    "sigma":     0.15,
    "log_M_0":  12.14,
    "log_M_1p": 13.43,
    "alpha":      1.0
    }

hod_dict_2 = {
    "log_M_min":14.06,
    "sigma":     0.71,
    "log_M_0":  14.06,
    "log_M_1p": 14.80,
    "alpha":      1.0
    }


degToRad = numpy.pi/180.0

### All precision values below come from running python2.7.2 in MacOS 10.7.4
### on a 1.7 GHz Intel Core i5
class CosmologyTestSingleEpoch(unittest.TestCase):

    def setUp(self):
        self.cosmo = cosmology.SingleEpoch(redshift=0.0, cosmo_dict=c_dict)
        
    def test_single_epoch(self):
        self.assertTrue(self.cosmo._flat)
        self.assertEqual(self.cosmo._redshift, 0.0)
        self.assertEqual(self.cosmo._chi, 0.0)
        self.assertEqual(numpy.log(self.cosmo._growth), 0.0)
        self.assertAlmostEqual(self.cosmo.omega_m(), 0.3 - 4.15e-5/0.7**2,
                               p_dict["cosmo"])
        self.assertAlmostEqual(self.cosmo.omega_l(), 0.7, p_dict["cosmo"])
        self.assertEqual(self.cosmo.w(self.cosmo._redshift), -1.0)
        self.assertAlmostEqual(numpy.log(self.cosmo.delta_v()), 
                               5.84412388, p_dict["cosmo"])
        self.assertAlmostEqual(numpy.log(self.cosmo.delta_c()),
                               0.51601430, p_dict["cosmo"])
        self.assertAlmostEqual(self.cosmo.sigma_r(8.0), 0.8, p_dict["cosmo"])
        
    def test_set_redshift(self):
        self.cosmo.set_redshift(1.0)
        self.assertTrue(self.cosmo._flat)
        self.assertEqual(self.cosmo._redshift, 1.0)
        self.assertAlmostEqual(numpy.log(self.cosmo._chi), 
                               7.74687924, p_dict["cosmo"])
        self.assertAlmostEqual(self.cosmo._growth, 0.61184534, p_dict["cosmo"])
        self.assertAlmostEqual(self.cosmo.omega_m(), 0.77405957,
                               p_dict["cosmo"])
        self.assertAlmostEqual(self.cosmo.omega_l(), 0.22583113,
                               p_dict["cosmo"])
        self.assertEqual(self.cosmo.w(self.cosmo._redshift), -1.0)
        self.assertAlmostEqual(numpy.log(self.cosmo.delta_v()),
                               5.8139178, p_dict["cosmo"])
        self.assertAlmostEqual(numpy.log(self.cosmo.delta_c()),
                               0.52122912, p_dict["cosmo"])
        self.assertAlmostEqual(self.cosmo.sigma_r(8.0), 0.48947627,
                               p_dict["cosmo"])

    def test_set_cosmology(self):
        self.cosmo.set_cosmology(c_dict_2, 1.0)
        self.assertTrue(self.cosmo._flat)
        self.assertEqual(self.cosmo._redshift, 1.0)
        self.assertAlmostEqual(numpy.log(self.cosmo._chi),
                               7.47157876, p_dict["cosmo"])
        self.assertAlmostEqual(self.cosmo._growth, 0.50001210, p_dict["cosmo"])
        self.assertAlmostEqual(self.cosmo.omega_m(), 0.99995765,
                               p_dict["cosmo"])
        self.assertAlmostEqual(self.cosmo.omega_l(), 0.0, p_dict["cosmo"])
        self.assertEqual(self.cosmo.w(self.cosmo._redshift), -1.0)
        self.assertAlmostEqual(numpy.log(self.cosmo.delta_v()),
                               5.87492980, p_dict["cosmo"])
        self.assertAlmostEqual(numpy.log(self.cosmo.delta_c()),
                               0.52263747, p_dict["cosmo"])
        self.assertAlmostEqual(self.cosmo.sigma_r(8.0), 0.40000968,
                               p_dict["cosmo"])

    def test_linear_power(self):
        k_array = numpy.logspace(-3, 2, 4)
        lin_power = [8.18733648, 9.49322932, 2.32587979, -7.75033120]
        for idx, k in enumerate(k_array):
            self.assertAlmostEqual(numpy.log(self.cosmo.linear_power(k)),
                                   lin_power[idx], p_dict["cosmo"])
        
        
class CosmologyTestMultiEpoch(unittest.TestCase):
    
    def setUp(self):
        self.cosmo = cosmology.MultiEpoch(0.0, 5.0, cosmo_dict=c_dict)

    def test_multi_epoch(self):
        chi_list = [0.0, 7.18763416, 7.74687926, 8.60279558] 
        growth_list = [1.0, 0.77321062, 0.61184534, 0.21356291]
        omega_m_list = [0.3 - 4.15e-5/0.7**2, 0.59110684,
                        0.77405957, 0.98926392]
        omega_l_list = [0.7, 0.40878186,
                        0.22583113,0.01068951]
        w_list = [-1.0, -1.0, -1.0, -1.0]
        delta_v_list = [5.84412389, 5.72815452, 5.81391783, 6.73154414]
        delta_c_list = [0.5160143, 0.77694983, 1.01250486, 2.06640216]
        sigma_8_list = [0.8, 0.61856849, 0.48947627, 0.17085033]
        for idx, z in enumerate([0.0, 0.5, 1.0, 5.0]):
            self.assertAlmostEqual(
                numpy.where(self.cosmo.comoving_distance(z) > 1e-16, 
                            numpy.log(self.cosmo.comoving_distance(z)), 0.0),
                chi_list[idx], p_dict["cosmo"])
            self.assertAlmostEqual(self.cosmo.growth_factor(z),
                                   growth_list[idx], p_dict["cosmo"])
            self.assertAlmostEqual(self.cosmo.omega_m(z),
                                   omega_m_list[idx], p_dict["cosmo"])
            self.assertAlmostEqual(self.cosmo.omega_l(z),
                                   omega_l_list[idx], p_dict["cosmo"])
            self.assertEqual(self.cosmo.epoch0.w(z), w_list[idx])
            self.assertAlmostEqual(numpy.log(self.cosmo.delta_v(z)), 
                                   delta_v_list[idx], p_dict["cosmo"])
            self.assertAlmostEqual(numpy.log(self.cosmo.delta_c(z)),
                                   delta_c_list[idx], p_dict["cosmo"])
            self.assertAlmostEqual(self.cosmo.sigma_r(8.0, z),
                                   sigma_8_list[idx], p_dict["cosmo"])

    def test_set_cosmology(self):
        chi_list = [0.0, 7.0040003, 7.4715789, 8.17486672] 
        growth_list = [1.0, 0.66667731, 0.50001201, 0.16667338]
        omega_m_list = [1.0 - 4.15e-5/0.7**2, 0.99994353,
                        0.99995765,  0.99998588]
        omega_l_list = [0.0, 0.0, 0.0, 0.0]
        w_list = [-1.0, -1.0, -1.0, -1.0]
        delta_v_list = [5.18183013,  5.58726375,
                        5.87493001, 6.97351045]
        delta_c_list = [0.52263724, 0.92808654, 1.21576064, 2.31435676]
        sigma_8_list = [0.8, 0.53334185, 0.40000961, 0.13333871]
        
        self.cosmo.set_cosmology(c_dict_2)
        for idx, z in enumerate([0.0, 0.5, 1.0, 5.0]):
            self.assertAlmostEqual(
                numpy.where(self.cosmo.comoving_distance(z) > 1e-16, 
                            numpy.log(self.cosmo.comoving_distance(z)), 0.0),
                chi_list[idx], p_dict["cosmo"])
            self.assertAlmostEqual(self.cosmo.growth_factor(z),
                                   growth_list[idx], p_dict["cosmo"])
            self.assertAlmostEqual(self.cosmo.omega_m(z),
                                   omega_m_list[idx], p_dict["cosmo"])
            self.assertAlmostEqual(self.cosmo.omega_l(z),
                                   omega_l_list[idx], p_dict["cosmo"])
            self.assertEqual(self.cosmo.epoch0.w(z), w_list[idx])
            self.assertAlmostEqual(numpy.log(self.cosmo.delta_v(z)), 
                                   delta_v_list[idx], p_dict["cosmo"])
            self.assertAlmostEqual(numpy.log(self.cosmo.delta_c(z)),
                                   delta_c_list[idx], p_dict["cosmo"])
            self.assertAlmostEqual(self.cosmo.sigma_r(8.0, z),
                                   sigma_8_list[idx], p_dict["cosmo"])


class MassFunctionTest(unittest.TestCase):
    
    def setUp(self):
        cosmo = cosmology.SingleEpoch(0.0, c_dict)
        self.mass = mass_function.MassFunction(cosmo_single_epoch=cosmo,
                                               halo_dict=h_dict)
        self.mass_array = numpy.logspace(9, 16, 4)
        
    def test_mass_function(self):
        nu_list = [-2.00781786, -0.84190606, 0.87876348, 3.70221194]
        f_mass_list = [0.43457575, -0.47342005, -2.30336595, -17.49871784]
        for idx, mass in enumerate(self.mass_array):
            if (mass < numpy.exp(self.mass.ln_mass_min) or 
                mass > numpy.exp(self.mass.ln_mass_max)):
                    continue
            self.assertAlmostEqual(numpy.log(self.mass.nu(mass)), nu_list[idx],
                                   p_dict["cosmo"])
            self.assertAlmostEqual(numpy.log(self.mass.f_m(mass)),
                                   f_mass_list[idx], p_dict["cosmo"])

    def test_set_redshift(self):
        nu_list = [-1.37242994, -0.34899534, 1.11477110, 3.42073448]
        f_mass_list = [-0.04868790,  -0.90112098, -2.67980091, -13.81901049]
        self.mass.set_redshift(1.0)
        for idx, mass in enumerate(self.mass_array):
            if (mass < numpy.exp(self.mass.ln_mass_min) or 
                mass > numpy.exp(self.mass.ln_mass_max)):
                    continue
            self.assertAlmostEqual(numpy.log(self.mass.nu(mass)), nu_list[idx],
                                   p_dict["cosmo"])
            self.assertAlmostEqual(numpy.log(self.mass.f_m(mass)),
                                   f_mass_list[idx], p_dict["cosmo"])

    def test_set_cosmology(self):
        nu_list = [0.0, -2.29906233, -0.08732051, 3.64626503]
        f_mass_list = [0.0, 0.63403019, -1.16671850, -16.70403410]
        self.mass.set_cosmology(c_dict_2)
        for idx, mass in enumerate(self.mass_array):
            if (mass < numpy.exp(self.mass.ln_mass_min) or 
                mass > numpy.exp(self.mass.ln_mass_max)):
                    continue
            self.assertAlmostEqual(numpy.log(self.mass.nu(mass)), nu_list[idx],
                                   p_dict["cosmo"])
            self.assertAlmostEqual(numpy.log(self.mass.f_m(mass)),
                                   f_mass_list[idx], p_dict["cosmo"])

    def test_set_halo(self):
        nu_list = [-2.00781786, -0.84190606, 0.87876348, 3.70221194]
        f_mass_list = [0.56714078, -0.52205057, -2.37763207, -13.76882516]
        self.mass.set_halo(h_dict_2)
        for idx, mass in enumerate(self.mass_array):
            if (mass < numpy.exp(self.mass.ln_mass_min) or 
                mass > numpy.exp(self.mass.ln_mass_max)):
                    continue
            self.assertAlmostEqual(numpy.log(self.mass.nu(mass)), nu_list[idx],
                                   p_dict["cosmo"])
            self.assertAlmostEqual(numpy.log(self.mass.f_m(mass)),
                                   f_mass_list[idx], p_dict["cosmo"])
            
                        
class HODTest(unittest.TestCase):
    
    def setUp(self):
        self.zheng = hod.HODZheng(hod_dict)
        self.mass_array = numpy.logspace(9, 16, 4)
        self.first_moment_list = [0.0, 0.0, 2.6732276, 372.48394295]
        self.second_moment_list = [0.0, 0.0, 6.14614597, 138743.2877621]
        self.nth_moment_list = [0.0, 0.0, 11.83175124, 51678901.92217977]
        
    def test_hod(self):
        for idx, mass in enumerate(self.mass_array):
             self.assertAlmostEqual(self.zheng.first_moment(mass),
                                    self.first_moment_list[idx])
             self.assertAlmostEqual(self.zheng.second_moment(mass),
                                    self.second_moment_list[idx])
             self.assertAlmostEqual(self.zheng.nth_moment(mass, 3),
                                    self.nth_moment_list[idx])


class HaloTest(unittest.TestCase):
    
    def setUp(self):
        cosmo = cosmology.SingleEpoch(0.0, cosmo_dict=c_dict)
        zheng = hod.HODZheng(hod_dict)
        self.h = halo.Halo(input_hod=zheng, cosmo_single_epoch=cosmo)
        self.k_array = numpy.logspace(-3, 2, 4)
        
    def test_halo(self):
        power_mm_list = [8.34442,  9.53808,
                         5.59839, -2.80500]
        power_gm_list = [8.23158,  9.46939,
                         5.18083, -0.73781]
        power_gg_list = [8.13780,  9.40684,
                         4.5788,  -0.51044]
        for idx, k in enumerate(self.k_array):
            self.assertAlmostEqual(numpy.log(self.h.power_mm(k)),
                                   power_mm_list[idx], p_dict["halo"])
            self.assertAlmostEqual(numpy.log(self.h.power_gm(k)),
                                   power_gm_list[idx], p_dict["halo"])
            self.assertAlmostEqual(numpy.log(self.h.power_gg(k)),
                                   power_gg_list[idx], p_dict["halo"])
            
    def test_set_cosmology(self):
        linear_power_list = [5.16650870,  8.11613036,
                             3.69335247, -5.84391743]
        power_mm_list = [6.61674,  8.27363,
                         5.67889, -3.03685]
        power_gm_list = [5.89280,  7.89096,
                         4.93468, -1.49355]
        power_gg_list = [5.22410,  7.53313,
                         4.17840, -1.37446]
        self.h.set_cosmology(c_dict_2)
        for idx, k in enumerate(self.k_array):
            self.assertAlmostEqual(numpy.log(self.h.linear_power(k)),
                                   linear_power_list[idx], p_dict["cosmo"])
            self.assertAlmostEqual(numpy.log(self.h.power_mm(k)),
                                   power_mm_list[idx], p_dict["halo"])
            self.assertAlmostEqual(numpy.log(self.h.power_gm(k)),
                                   power_gm_list[idx], p_dict["halo"])
            self.assertAlmostEqual(numpy.log(self.h.power_gg(k)),
                                   power_gg_list[idx], p_dict["halo"])

    def test_set_halo(self):
        power_mm_list = [8.41958,  9.56136,
                         5.76760, -2.86430]
        power_gm_list = [8.25847,  9.45965,
                         5.36000, -0.75668]
        power_gg_list = [8.12278,  9.36674,
                         4.80720, -0.45823]
        self.h.set_halo(h_dict_2)
        for idx, k in enumerate(self.k_array):
            self.assertAlmostEqual(numpy.log(self.h.power_mm(k)),
                                   power_mm_list[idx], p_dict["halo"])
            self.assertAlmostEqual(numpy.log(self.h.power_gm(k)),
                                   power_gm_list[idx], p_dict["halo"])
            self.assertAlmostEqual(numpy.log(self.h.power_gg(k)),
                                   power_gg_list[idx], p_dict["halo"])

    def test_set_hod(self):
        power_gm_list = [8.87899, 10.03277,
                         6.66003,  1.19069]
        power_gg_list = [9.26146, 10.48081,
                         6.25604, -0.15185]
        self.h.set_hod(hod_dict_2)
        for idx, k in enumerate(self.k_array):
            self.assertAlmostEqual(numpy.log(self.h.power_gm(k)),
                                   power_gm_list[idx], p_dict["halo"])
            self.assertAlmostEqual(numpy.log(self.h.power_gg(k)),
                                   power_gg_list[idx], p_dict["halo"])
            
    def test_set_redshift(self):
        linear_power_list = [7.20478501, 8.51067786,
                             1.34332833, -8.73288266]
        power_mm_list = [7.25080,  8.52330,
                         3.82709, -4.41395]
        power_gm_list = [7.33920,  8.62475,
                         3.53600, -2.47495]
        power_gg_list = [7.43307,  8.72776,
                         3.15151, -2.09887]
        self.h.set_redshift(1.0)
        for idx, k in enumerate(self.k_array):
            self.assertAlmostEqual(numpy.log(self.h.linear_power(k)),
                                   linear_power_list[idx], p_dict["cosmo"])
            self.assertAlmostEqual(numpy.log(self.h.power_mm(k)),
                                   power_mm_list[idx], p_dict["halo"])
            self.assertAlmostEqual(numpy.log(self.h.power_gm(k)),
                                   power_gm_list[idx], p_dict["halo"])
            self.assertAlmostEqual(numpy.log(self.h.power_gg(k)),
                                   power_gg_list[idx], p_dict["halo"])

### Commented out currently as it is a future feature not yet mature.
# class HaloTriSpectrumTest(unittest.TestCase):
#    
#     def setUp(self):
#        cosmo = cosmology.SingleEpoch(0.0, cosmo_dict=c_dict)
#         mass = mass_function.MassFunctionSecondOrder(cosmo_single_epoch=cosmo,
#                                                     halo_dict=h_dict)
#         pert = perturbation_spectra.PerturbationTheory(
#            cosmo_single_epoch=cosmo)
#         self.h = halo_trispectrum.HaloTrispectrum(
#             redshift=0.0, single_epoch_cosmo=cosmo,
#             mass_func_second=mass, perturbation=pert, halo_dict=h_dict)
#         self.k_array = numpy.logspace(-3, 2, 4)
#         
#     def test_trispectrum(self):
#         for idx, k in enumerate(self.k_array):
#             self.assertGreater(
#                 self.h.trispectrum_parallelogram(k, k, 0.0), 0.0)


class dNdzTest(unittest.TestCase):

    def setUp(self):
        self.lens_dist = kernel.dNdzMagLim(z_min=0.0, z_max=2.0, 
                                           a=2, z0=0.3, b=2)
        self.source_dist = kernel.dNdzGaussian(z_min=0.0, z_max=2.0,
                                               z0=1.0, sigma_z=0.2)
        self.z_array = numpy.linspace(0.0, 2.0, 4)
        self.lens_dist_list = [0.0, 0.00318532, 0.0, 0.0]
        self.source_dist_list = [3.72665317e-06, 0.24935220, 
                                 0.24935220, 3.72665317e-06]

    def test_redshift_dist(self):
        for idx, z in enumerate(self.z_array):
             self.assertAlmostEqual(self.lens_dist.dndz(z),
                                    self.lens_dist_list[idx])
             self.assertAlmostEqual(self.source_dist.dndz(z),
                                    self.source_dist_list[idx])


class WindowFunctionTest(unittest.TestCase):

    def setUp(self):
        lens_dist = kernel.dNdzMagLim(z_min=0.0, z_max=2.0, 
                                           a=1, z0=0.3, b=1)
        source_dist = kernel.dNdzGaussian(z_min=0.0, z_max=2.0,
                                               z0=1.0, sigma_z=0.2)
        cosmo = cosmology.MultiEpoch(0.0, 5.0, cosmo_dict=c_dict)
        self.lens_window = kernel.WindowFunctionGalaxy(
            lens_dist, cosmo_multi_epoch=cosmo)
        self.source_window = kernel.WindowFunctionConvergence(
            source_dist, cosmo_multi_epoch=cosmo)
        self.z_array = numpy.linspace(0.0, 2.0, 4)
    
    def test_window_function(self):
        lens_window_list = [0.0, -14.00096, -13.30840, -12.903527]
        source_window_list = [0.0, -17.217075, -16.524005, -16.118615]
        for idx, z in enumerate(self.z_array):
            self.assertAlmostEqual(
                numpy.where(self.lens_window.window_function(z) > 1e-16,
                            numpy.log(self.lens_window.window_function(z)),
                            0.0), lens_window_list[idx], p_dict["window"])
            self.assertAlmostEqual(
                numpy.where(self.source_window.window_function(z) > 0.0,
                            numpy.log(self.source_window.window_function(z)),
                            0.0), source_window_list[idx], p_dict["window"])

    def test_set_cosmology(self):
        lens_window_list = [0.0, -13.99937, -13.306473, -12.901255]
        source_window_list = [0.0, -16.013004, -15.320023, -14.914725]
        
        cosmo = cosmology.MultiEpoch(0.0, 5.0, c_dict_2)
        self.lens_window.set_cosmology_object(cosmo)
        self.source_window.set_cosmology_object(cosmo)
        for idx, z in enumerate(self.z_array):
            self.assertAlmostEqual(
                numpy.where(self.lens_window.window_function(z) > 0.0,
                            numpy.log(self.lens_window.window_function(z)),
                            0.0), lens_window_list[idx], p_dict["window"])
            self.assertAlmostEqual(
                numpy.where(self.source_window.window_function(z) > 1e-32,
                            numpy.log(self.source_window.window_function(z)),
                            0.0), source_window_list[idx], p_dict["window"])


class KenrelTest(unittest.TestCase):

    def setUp(self):
        cosmo = cosmology.MultiEpoch(0.0, 5.0, cosmo_dict=c_dict)
        lens_dist = kernel.dNdzMagLim(z_min=0.0, z_max=2.0, 
                                      a=2, z0=0.3, b=2)
        source_dist = kernel.dNdzGaussian(z_min=0.0, z_max=2.0,
                                          z0=1.0, sigma_z=0.2)
        lens_window = kernel.WindowFunctionGalaxy(
            lens_dist, cosmo_multi_epoch=cosmo)
        source_window = kernel.WindowFunctionConvergence(
            source_dist, cosmo_multi_epoch=cosmo)
        self.kern = kernel.Kernel(0.001*0.001*degToRad, 1.0*100.0*degToRad,
                               window_function_a=lens_window,
                               window_function_b=source_window,
                               cosmo_multi_epoch=cosmo)
        self.ln_ktheta_array = numpy.linspace(-15, -1, 4)
        
    def test_kernel(self):
        k_list = [-10.689493, -10.689757,
                  -12.737077, -30.120051]
        for idx, ln_ktheta in enumerate(self.ln_ktheta_array):
            kern = numpy.abs(self.kern.kernel(ln_ktheta))
            self.assertAlmostEqual(
                numpy.where(kern > 0.0, numpy.log(kern), 0.0),
                k_list[idx], p_dict["kernel"])

    def test_set_cosmology(self):
        self.kern.set_cosmology(c_dict_2)
        k_list = [ -9.946510,  -9.946688,
                  -12.957102, -27.694108]
        for idx, ln_ktheta in enumerate(self.ln_ktheta_array):
            kern = numpy.abs(self.kern.kernel(ln_ktheta))
            self.assertAlmostEqual(
                numpy.where(kern > 0.0, numpy.log(kern), 0.0),
                k_list[idx], p_dict["kernel"])
            

class CorrelationTest(unittest.TestCase):
    
    def setUp(self):
        cosmo_multi = cosmology.MultiEpoch(0.0, 5.0, cosmo_dict=c_dict)
        lens_dist = kernel.dNdzMagLim(z_min=0.0, z_max=2.0, 
                                      a=2, z0=0.3, b=2)
        source_dist = kernel.dNdzGaussian(z_min=0.0, z_max=2.0,
                                          z0=1.0, sigma_z=0.2)
        lens_window = kernel.WindowFunctionGalaxy(
            lens_dist, cosmo_multi_epoch=cosmo_multi)
        source_window = kernel.WindowFunctionConvergence(
            source_dist, cosmo_multi_epoch=cosmo_multi)
        kern = kernel.Kernel(0.001*0.001*deg_to_rad, 1.0*100.0*deg_to_rad,
                             window_function_a=lens_window,
                             window_function_b=source_window,
                             cosmo_multi_epoch=cosmo_multi)
        
        zheng = hod.HODZheng(hod_dict)
        cosmo_single = cosmology.SingleEpoch(0.0, cosmo_dict=c_dict)
        h = halo.Halo(input_hod=zheng, cosmo_single_epoch=cosmo_single)
        self.corr = correlation.Correlation(0.001, 1.0,
                                            input_kernel=kern,
                                            input_halo=h,
                                            power_spec='power_mm')
        self.theta_array = numpy.logspace(-3, 0, 4)*deg_to_rad
        
    def test_correlation(self):
        corr_list = [-4.111348, -4.724446, -6.882630, -8.900066]
        for idx, theta in enumerate(self.theta_array):
            self.assertAlmostEqual(
                numpy.log(self.corr.correlation(theta)),
                corr_list[idx], p_dict["corr"])

    def test_set_redshift(self):
        self.corr.set_redshift(0.5)
        corr_list = [-4.208979, -4.826124, -6.963666, -8.900316]
        for idx, theta in enumerate(self.theta_array):
            self.assertAlmostEqual(
                numpy.log(self.corr.correlation(theta)),
                corr_list[idx], p_dict["corr"])

    def test_set_cosmology(self):
        self.corr.set_cosmology(c_dict_2)
        corr_list = [-2.908135, -3.49623, -5.804543, -9.395502]
        for idx, theta in enumerate(self.theta_array):
            self.assertAlmostEqual(
                numpy.log(self.corr.correlation(theta)),
                corr_list[idx], p_dict["corr"])

    def test_set_hod(self):
        self.corr.set_hod(hod_dict_2)
        self.corr.set_power_spectrum('power_gm')
        corr_list = [-1.866602, -3.488535, -6.347507, -8.393573]
        for idx, theta in enumerate(self.theta_array):
            self.assertAlmostEqual(
                numpy.log(self.corr.correlation(theta)),
                corr_list[idx], p_dict["corr"])

if __name__ == "__main__":
    print "*******************************"
    print "*                             *"
    print "*      CHOMP Unit Test        *"
    print "*                             *"
    print "*******************************"
    unittest.main()
    
