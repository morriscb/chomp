import cosmology
import correlation
import defaults
import halo
import hod
import kernel
import mass_function
import numpy
import unittest

### In order for the unittests to work correctly, these are the assumed
### precision values of the code.
defaults.default_precision = {
    "corr_npoints": 25,
    "corr_precision":1.48e-8,
    "cosmo_npoints": 50,
    "cosmo_precision": 1.48e-8,
    "dNdz_precision": 1.48e-16,
    "halo_npoints": 50,
    "halo_precision": 1.48-8,
    "halo_limit" : 100,
    "kernel_npoints": 200,
    "kernel_precision": 1.48e-16,
    "kernel_limit": 200, ### If the variable force_quad is set in the Kernel 
                         ### class this value sets the limit for the quad
                         ### integration
    "kernel_bessel_limit": 32, ### Defines how many zeros before cutting off the
                               ### bessel function in kernel.py
    "mass_npoints": 50,
    "mass_precision": 1.48e-8,
    "window_npoints": 50,
    "window_precision": 1.48e-16
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

degToRad = numpy.pi/180.0


class CosmologyTestSingleEpoch(unittest.TestCase):

    def setUp(self):
        self.cosmo = cosmology.SingleEpoch(redshift=0.0, cosmo_dict=c_dict)
        
    def test_single_epoch(self):
        self.assertTrue(self.cosmo._flat)
        self.assertEqual(self.cosmo._redshift, 0.0)
        self.assertEqual(self.cosmo._chi, 0.0)
        self.assertEqual(numpy.log(self.cosmo._growth), 0.0)
        self.assertAlmostEqual(self.cosmo.omega_m(), 0.3 - 4.15e-5/0.7**2)
        self.assertAlmostEqual(self.cosmo.omega_l(), 0.7)
        self.assertEqual(self.cosmo.w(self.cosmo._redshift), -1.0)
        self.assertAlmostEqual(numpy.log(self.cosmo.delta_v()), 
                               5.84412388)
        self.assertAlmostEqual(numpy.log(self.cosmo.delta_c()),
                               0.51601430)
        self.assertAlmostEqual(self.cosmo.sigma_r(8.0), 0.8)
        
    def test_set_redshift(self):
        self.cosmo.set_redshift(1.0)
        self.assertTrue(self.cosmo._flat)
        self.assertEqual(self.cosmo._redshift, 1.0)
        self.assertAlmostEqual(numpy.log(self.cosmo._chi), 
                               7.74687924)
        self.assertAlmostEqual(self.cosmo._growth, 0.61184534)
        self.assertAlmostEqual(self.cosmo.omega_m(), 0.77405957)
        self.assertAlmostEqual(self.cosmo.omega_l(), 0.22583113)
        self.assertEqual(self.cosmo.w(self.cosmo._redshift), -1.0)
        self.assertAlmostEqual(numpy.log(self.cosmo.delta_v()),
                               5.8139178)
        self.assertAlmostEqual(numpy.log(self.cosmo.delta_c()),
                               0.52122912)
        self.assertAlmostEqual(self.cosmo.sigma_r(8.0), 0.48947627)

    def test_set_cosmology(self):
        self.cosmo.set_cosmology(c_dict_2, 1.0)
        self.assertTrue(self.cosmo._flat)
        self.assertEqual(self.cosmo._redshift, 1.0)
        self.assertAlmostEqual(numpy.log(self.cosmo._chi),
                               7.47157876)
        self.assertAlmostEqual(self.cosmo._growth, 0.50001210)
        self.assertAlmostEqual(self.cosmo.omega_m(), 0.99995765)
        self.assertAlmostEqual(self.cosmo.omega_l(), 0.0)
        self.assertEqual(self.cosmo.w(self.cosmo._redshift), -1.0)
        self.assertAlmostEqual(numpy.log(self.cosmo.delta_v()),
                               5.87492980)
        self.assertAlmostEqual(numpy.log(self.cosmo.delta_c()),
                               0.52263747)
        self.assertAlmostEqual(self.cosmo.sigma_r(8.0), 0.40000968)

    def test_linear_power(self):
        k_array = numpy.logspace(-3, 2, 4)
        lin_power = [8.18733648, 9.49322932, 2.32587979, -7.75033120]
        for idx, k in enumerate(k_array):
            self.assertAlmostEqual(numpy.log(self.cosmo.linear_power(k)),
                                   lin_power[idx])
        
        
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
                numpy.where(self.cosmo.comoving_distance(z) >0.0, 
                      numpy.log(self.cosmo.comoving_distance(z)), 0.0),
                chi_list[idx])
            self.assertAlmostEqual(self.cosmo.growth_factor(z),
                                   growth_list[idx])
            self.assertAlmostEqual(self.cosmo.omega_m(z),
                                   omega_m_list[idx])
            self.assertAlmostEqual(self.cosmo.omega_l(z),
                                   omega_l_list[idx])
            self.assertEqual(self.cosmo.epoch0.w(z), w_list[idx])
            self.assertAlmostEqual(numpy.log(self.cosmo.delta_v(z)), 
                                   delta_v_list[idx])
            self.assertAlmostEqual(numpy.log(self.cosmo.delta_c(z)),
                                   delta_c_list[idx])
            self.assertAlmostEqual(self.cosmo.sigma_r(8.0, z),
                                   sigma_8_list[idx])

    def test_set_cosmology(self):
        chi_list = [0.0, 7.0040003, 7.4715789, 8.17486672] 
        growth_list = [1.0, 0.66667731, 0.50001201, 0.1666735]
        omega_m_list = [1.0 - 4.15e-5/0.7**2, 0.99994353,
                        0.99995765,  0.99998588]
        omega_l_list = [0.0, 0.0, 0.0, 0.0]
        w_list = [-1.0, -1.0, -1.0, -1.0]
        delta_v_list = [5.18183013,  5.58726375,
                        5.87493001, 6.97350979]
        delta_c_list = [0.52263724, 0.92808654, 1.21576064, 2.3143561]
        sigma_8_list = [0.8, 0.53334185, 0.40000961, 0.1333388]
        
        self.cosmo.set_cosmology(c_dict_2)
        for idx, z in enumerate([0.0, 0.5, 1.0, 5.0]):
            self.assertAlmostEqual(
                numpy.where(self.cosmo.comoving_distance(z) >0.0, 
                      numpy.log(self.cosmo.comoving_distance(z)), 0.0),
                chi_list[idx])
            self.assertAlmostEqual(self.cosmo.growth_factor(z),
                                   growth_list[idx])
            self.assertAlmostEqual(self.cosmo.omega_m(z),
                                   omega_m_list[idx])
            self.assertAlmostEqual(self.cosmo.omega_l(z),
                                   omega_l_list[idx])
            self.assertEqual(self.cosmo.epoch0.w(z), w_list[idx])
            self.assertAlmostEqual(numpy.log(self.cosmo.delta_v(z)), 
                                   delta_v_list[idx])
            self.assertAlmostEqual(numpy.log(self.cosmo.delta_c(z)),
                                   delta_c_list[idx])
            self.assertAlmostEqual(self.cosmo.sigma_r(8.0, z),
                                   sigma_8_list[idx])


class MassFunctionTest(unittest.TestCase):
    
    def setUp(self):
        cosmo = cosmology.SingleEpoch(0.0, c_dict)
        self.mass = mass_function.MassFunction(cosmo_single_epoch=cosmo,
                                               halo_dict=h_dict)
        self.mass_array = numpy.logspace(9, 16, 4)
        
    def test_mass_function(self):
        nu_list = [-2.00781697, -0.84190606, 0.87876348, 3.70221176]
        f_mass_list = [0.43457911, -0.47341601, -2.30336188, -17.4987111]
        for idx, mass in enumerate(self.mass_array):
            if (mass < numpy.exp(self.mass.ln_mass_min) or 
                mass > numpy.exp(self.mass.ln_mass_max)):
                    continue
            self.assertAlmostEqual(numpy.log(self.mass.nu(mass)), nu_list[idx])
            self.assertAlmostEqual(numpy.log(self.mass.f_m(mass)),
                                   f_mass_list[idx])

    def test_set_redshift(self):
        nu_list = [-1.37245339, -0.34899603, 1.1147666, 3.42069564]
        f_mass_list = [-0.06740317, -0.91985385, -2.69852669, -13.83730113]
        self.mass.set_redshift(1.0)
        for idx, mass in enumerate(self.mass_array):
            if (mass < numpy.exp(self.mass.ln_mass_min) or 
                mass > numpy.exp(self.mass.ln_mass_max)):
                    continue
            self.assertAlmostEqual(numpy.log(self.mass.nu(mass)), nu_list[idx])
            self.assertAlmostEqual(numpy.log(self.mass.f_m(mass)),
                                   f_mass_list[idx])

    def test_set_cosmology(self):
        nu_list = [0.0, -2.29906247, -0.08732065, 3.64626423]
        f_mass_list = [0.0, 0.63398636, -1.1667623, -16.70406668]
        self.mass.set_cosmology(c_dict_2)
        for idx, mass in enumerate(self.mass_array):
            if (mass < numpy.exp(self.mass.ln_mass_min) or 
                mass > numpy.exp(self.mass.ln_mass_max)):
                    continue
            self.assertAlmostEqual(numpy.log(self.mass.nu(mass)), nu_list[idx])
            self.assertAlmostEqual(numpy.log(self.mass.f_m(mass)),
                                   f_mass_list[idx])

    def test_set_halo(self):
        nu_list = [-2.00781697, -0.84190606, 0.87876348, 3.70221176]
        f_mass_list = [0.56712195, -0.52206856, -2.37765005, -13.76884123]
        self.mass.set_halo(h_dict_2)
        for idx, mass in enumerate(self.mass_array):
            if (mass < numpy.exp(self.mass.ln_mass_min) or 
                mass > numpy.exp(self.mass.ln_mass_max)):
                    continue
            self.assertAlmostEqual(numpy.log(self.mass.nu(mass)), nu_list[idx])
            self.assertAlmostEqual(numpy.log(self.mass.f_m(mass)),
                                   f_mass_list[idx])
            
                        
class HODTest(unittest.TestCase):
    
    def setUp(self):
        self.zheng = hod.HODZheng(10**12, 0.15, 10**12, 10**13, 1.0)
        self.mass_array = numpy.logspace(9, 16, 4)
        self.first_moment_list = [0.0, 0.0, 5.54158883, 1000.9]
        self.second_moment_list = [0.0, 0.0, 29.70920680, 1001799.80999999]
        self.nth_moment_list = [0.0, 0.0, 153.91393832, 1002699428.0309982]
        
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
        zheng = hod.HODZheng(10**13.0, 0.15, 10**13.0, 10**14.0, 1.0)
        self.h = halo.Halo(input_hod=zheng, cosmo_single_epoch=cosmo)
        self.k_array = numpy.logspace(-3, 2, 4)
        
    def test_halo(self):
        power_mm_list = [8.34440408, 9.53785114,
                         5.10250239, -3.74386887]
        power_gm_list = [8.61662044, 9.81345489,
                         5.67772657, -0.85787968]
        power_gg_list = [8.85261271, 10.07829163,
                         5.65157175, -0.31382178]
        for idx, k in enumerate(self.k_array):
            self.assertAlmostEqual(numpy.log(self.h.power_mm(k)),
                                   power_mm_list[idx])
            self.assertAlmostEqual(numpy.log(self.h.power_gm(k)),
                                   power_gm_list[idx])
            self.assertAlmostEqual(numpy.log(self.h.power_gg(k)),
                                   power_gg_list[idx])

    def test_set_redshift(self):
        linear_power_list = [7.20478501, 8.51067786,
                             1.34332833, -8.73288266]
        power_mm_list = [7.24994929, 8.52297824,
                         3.51013974, -5.3865863]
        power_gm_list = [7.69396091, 8.9648076,
                         4.17998803, -2.61963]
        power_gg_list = [8.12609056, 9.40331179,
                         4.4452134, -1.80351356]
        self.h.set_redshift(1.0)
        for idx, k in enumerate(self.k_array):
            self.assertAlmostEqual(numpy.log(self.h.linear_power(k)),
                                   linear_power_list[idx])
            self.assertAlmostEqual(numpy.log(self.h.power_mm(k)),
                                   power_mm_list[idx])
            self.assertAlmostEqual(numpy.log(self.h.power_gm(k)),
                                   power_gm_list[idx])
            self.assertAlmostEqual(numpy.log(self.h.power_gg(k)),
                                   power_gg_list[idx])

    def test_set_cosmology(self):
        linear_power_list = [5.16650884, 8.11613050,
                             3.69335261, -5.84391729]
        power_mm_list = [6.6167008, 8.27330311,
                         5.20130153, -4.09390277]
        power_gm_list = [6.33796703, 8.14871165,
                         5.10668697, -1.52516298]
        power_gg_list = [5.96915794, 8.01681526,
                         4.70719854, -1.27002309]
        self.h.set_cosmology(c_dict_2)
        for idx, k in enumerate(self.k_array):
            self.assertAlmostEqual(numpy.log(self.h.linear_power(k)),
                                   linear_power_list[idx])
            self.assertAlmostEqual(numpy.log(self.h.power_mm(k)),
                                   power_mm_list[idx])
            self.assertAlmostEqual(numpy.log(self.h.power_gm(k)),
                                   power_gm_list[idx])
            self.assertAlmostEqual(numpy.log(self.h.power_gg(k)),
                                   power_gg_list[idx])

    def test_set_halo(self):
        power_mm_list = [8.41952649, 9.56100352,
                         5.20127689, -3.78991171]
        power_gm_list = [8.63398014, 9.78279491,
                         5.74842142, -0.92271325]
        power_gg_list = [8.81481816, 9.99415073,
                         5.77308467, -0.30805068]
        self.h.set_halo(h_dict_2)
        for idx, k in enumerate(self.k_array):
            self.assertAlmostEqual(numpy.log(self.h.power_mm(k)),
                                   power_mm_list[idx])
            self.assertAlmostEqual(numpy.log(self.h.power_gm(k)),
                                   power_gm_list[idx])
            self.assertAlmostEqual(numpy.log(self.h.power_gg(k)),
                                   power_gg_list[idx])

    def test_set_hod(self):
        power_gm_list = [8.32596635, 9.54208384,
                         5.00764145, -1.71174672]
        power_gg_list = [8.31078887, 9.547477,
                         4.80756239, -1.1721439]
        zheng = hod.HODZheng(10**12.0, 0.15, 10**12.0, 10**13.0, 1.0)
        self.h.set_hod(zheng)
        for idx, k in enumerate(self.k_array):
            self.assertAlmostEqual(numpy.log(self.h.power_gm(k)),
                                   power_gm_list[idx])
            self.assertAlmostEqual(numpy.log(self.h.power_gg(k)),
                                   power_gg_list[idx])


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
        lens_window_list = [0.0, -14.0011948, -13.3086357, -12.90375869]
        source_window_list = [0.0, -17.21707686, 
                              -16.5240056, -16.11861652]
        for idx, z in enumerate(self.z_array):
            if lens_window_list[idx] == 0.0:
                # May report and error for python version < 2.7
                self.assertLessEqual(self.lens_window.window_function(z), 1e-16)
                self.assertGreaterEqual(
                    self.lens_window.window_function(z), -1e-16)
            else:
                self.assertAlmostEqual(
                    numpy.where(self.lens_window.window_function(z) > 0.0,
                    numpy.log(self.lens_window.window_function(z)),
                    0.0), lens_window_list[idx])
            self.assertAlmostEqual(
                numpy.where(self.source_window.window_function(z) > 0.0,
                            numpy.log(self.source_window.window_function(z)),
                            0.0), source_window_list[idx])

    def test_set_cosmology(self):
        lens_window_list = [0.0, -14.00059478, -13.30768976, -12.90246701]
        source_window_list = [0.0, -16.01304191, 
                              -15.32006144, -14.91476318]
        self.lens_window.set_cosmology(c_dict_2)
        self.source_window.set_cosmology(c_dict_2)
        for idx, z in enumerate(self.z_array):
            if lens_window_list[idx] == 0.0:
                self.assertLess(self.lens_window.window_function(z), 1e-16)
                self.assertGreater(self.lens_window.window_function(z), -1e-16)
            else:
                self.assertAlmostEqual(
                    numpy.where(self.lens_window.window_function(z) > 0.0,
                    numpy.log(self.lens_window.window_function(z)),
                    0.0), lens_window_list[idx])
            self.assertAlmostEqual(
                numpy.where(self.source_window.window_function(z) > 0.0,
                            numpy.log(self.source_window.window_function(z)),
                            0.0), source_window_list[idx])

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
        k_list = [-10.68949418, -10.68975736,
                  -12.68982878, -24.02250207]
        for idx, ln_ktheta in enumerate(self.ln_ktheta_array):
            kern = numpy.abs(self.kern.kernel(ln_ktheta))
            self.assertAlmostEqual(
                numpy.where(kern > 0.0, numpy.log(kern), 0.0),
                k_list[idx])

    def test_set_cosmology(self):
        self.kern.set_cosmology(c_dict_2)
        k_list = [-9.94923904, -9.94941557,
                  -13.01832302, -21.82788877]
        for idx, ln_ktheta in enumerate(self.ln_ktheta_array):
            kern = numpy.abs(self.kern.kernel(ln_ktheta))
            self.assertAlmostEqual(
                numpy.where(kern > 0.0, numpy.log(kern), 0.0),
                k_list[idx])
            

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
        kern = kernel.Kernel(0.001*0.001*degToRad, 1.0*100.0*degToRad,
                             window_function_a=lens_window,
                             window_function_b=source_window,
                             cosmo_multi_epoch=cosmo_multi)
        
        zheng = hod.HODZheng(10**13.0, 0.15, 10**13.0, 10**14.0, 1.0)
        cosmo_single = cosmology.SingleEpoch(0.0, cosmo_dict=c_dict)
        h = halo.Halo(input_hod=zheng, cosmo_single_epoch=cosmo_single)
        self.corr = correlation.Correlation(0.001*degToRad, 1.0*degToRad,
                                            input_kernel=kern,
                                            input_halo=h,
                                            powSpec='power_mm')
        self.theta_array = numpy.logspace(-3, 0, 4)*degToRad
        
    def test_correlation(self):
        corr_list = [-4.73676329, -5.19089047, -6.84293768, -8.89937344]
        for idx, theta in enumerate(self.theta_array):
            self.assertAlmostEqual(
                numpy.log(self.corr.correlation(theta)), corr_list[idx])

    def test_set_redshift(self):
        self.corr.set_redshift(0.5)
        corr_list = [-4.82397621, -5.27612659, -6.91672309, -8.89986573]
        for idx, theta in enumerate(self.theta_array):
            self.assertAlmostEqual(
                numpy.log(self.corr.correlation(theta)), corr_list[idx])

    def test_set_cosmology(self):
        self.corr.set_cosmology(c_dict_2)
        corr_list = [-3.53377593, -3.95935867, -5.74360484, -9.39434454]
        for idx, theta in enumerate(self.theta_array):
            self.assertAlmostEqual(
                numpy.log(self.corr.correlation(theta)), corr_list[idx])

    def test_set_hod(self):
        zheng = hod.HODZheng(10**12.0, 0.15, 10**12.0, 10**13.0, 1.0)
        self.corr.set_hod(zheng)
        self.corr.set_power_spectrum('power_gm')
        corr_list = [-4.32282959, -5.15879726, -6.88146865, -8.85220884]
        for idx, theta in enumerate(self.theta_array):
            self.assertAlmostEqual(
                numpy.log(self.corr.correlation(theta)), corr_list[idx])


if __name__ == "__main__":
    print "*******************************"
    print "*                             *"
    print "*      CHOMP Unit Test        *"
    print "*                             *"
    print "*******************************"

    print "WARNING::If you have changed any of the default precision values in"
    print "\tdefaults.default_precision, one or more of these tests may fail."
    unittest.main()
    
