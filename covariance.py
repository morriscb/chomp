from copy import copy
import cosmology
import defaults
import kernel
import halo
import halo_trispectrum
import numpy
from scipy import special
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline

"""Objects for computing covariance marix given an input survey are, theta
   limits, and binning.
"""

deg_to_rad = numpy.pi/180.0
rad_to_deg = 180.0/numpy.pi
deg2_to_strad = deg_to_rad*deg_to_rad
strad_to_deg2 = rad_to_deg*rad_to_deg

__author__ = ("Chris Morrison")

class Covariance(object):
    """
    Class to compute the covariance matrix between a range of theta values,
    given two imput correlation objects, survey definition, and a halo 
    trispectrum. It is assumed that the ranges for both correlations are the
    same, if they are not the covariance will default to the binning, cosmology,
    and 2-point halo objects in input_correlation_a
    estimate the covariance between different estimators as a function.

    Attributes:
        input_correlation_a: A Correlation object from correlation.py
        input_correlation_b: A Correlation object from correlation.py
        bins_per_decade: int log spacing of the theta bins
        survey_area_deg2: float value of the survey area it square degrees
        n_a: float number of objects in correlation a
        n_b: float number of objects in correlation b
        variance: float variance per pair (e.g. shape noise)
        nongaussian_cov: bool, toggles nonguassian covariance
        input_halo: HaloTrispectrum object from halo.py

        theta_array: array of theta values for computed correlation function
        wcovar_array: array of computed covariance values at theta_array values
    """

    def __init__(self, input_correlation_a, input_correlation_b,
                 bins_per_decade=5.0, survey_area_deg2=20,
                 n_a=1.0e4, n_b=1.0e4, variance=1.0, nongaussian_cov=True,
                 input_halo_trispectrum=None, power_spec='power_mm',
                 poisson_noise=None, **kws):

        self.annular_bins = []
        self.log_theta_min = input_correlation_a.log_theta_min
        self.log_theta_max = input_correlation_a.log_theta_max
        unit_double = (numpy.floor(self.log_theta_min)*bins_per_decade)
        theta = numpy.power(10.0, unit_double/(1.0*bins_per_decade))
        self.corr_a = input_correlation_a
        self.corr_b = input_correlation_b
        if self.corr_a == self.corr_b:
            self.matching_corrs = True
        else:
            self.matching_corrs = False
        
        while theta < numpy.power(10.0, self.log_theta_max):
            if (theta >= numpy.power(10.0, self.log_theta_min) and
                theta < numpy.power(10.0, self.log_theta_max)):
                self.annular_bins.append(AnnulusBin(
                    theta, numpy.power(
                        10.0, (unit_double+1.0)/(1.0*bins_per_decade))))
            unit_double += 1.0
            theta = numpy.power(10.0, unit_double/(1.0*bins_per_decade))

        self.area = survey_area_deg2 * deg2_to_strad
        try:
            self.n_a1 = n_a[0]
            self.n_a2 = n_a[1]
        except TypeError, IndexError:
            self.n_a1 = n_a
            self.n_a2 = n_a
        try:
            self.n_b1 = n_b[0]
            self.n_b2 = n_b[1]
        except TypeError, IndexError:
            self.n_b1 = n_b
            self.n_b2 = n_b
        self.nongaussian_cov = nongaussian_cov
        self.poisson_noise = poisson_noise

        self.kernel = kernel.KernelCovariance(
            numpy.power(10.0, self.log_theta_min)*
            defaults.default_limits["k_min"],
            numpy.power(10.0, self.log_theta_max)*
            defaults.default_limits["k_max"],
            input_correlation_a.kernel.window_function_a,
            input_correlation_a.kernel.window_function_b,
            input_correlation_b.kernel.window_function_a,
            input_correlation_b.kernel.window_function_b,
            input_correlation_a.kernel.cosmo, force_quad=False)

        self.equal_windows = [
            self.kernel.window_function_a1 == self.kernel.window_function_a2,
            self.kernel.window_function_a2 == self.kernel.window_function_b1,
            self.kernel.window_function_b1 == self.kernel.window_function_b2,
            self.kernel.window_function_a1 == self.kernel.window_function_b2
        ]
        self.density = [
            self.n_a1/self.area,
            self.n_a2/self.area,
            self.n_b1/self.area,
            self.n_b2/self.area,
        ]
        self.variance = variance
        self.cosmic_shear = self._identify_cosmic_shear()

        self._z_min_a = numpy.max([self.kernel.window_function_a1.z_min,
                                   self.kernel.window_function_a2.z_min])
        self._z_max_a = numpy.min([self.kernel.window_function_a1.z_max,
                                   self.kernel.window_function_a2.z_max])
        self._z_min_b = numpy.max([self.kernel.window_function_b1.z_min,
                                   self.kernel.window_function_b2.z_min])
        self._z_max_b = numpy.min([self.kernel.window_function_b1.z_max,
                                   self.kernel.window_function_b2.z_max])
        self._chi_min_a = self.kernel.cosmo.comoving_distance(self._z_min_a)
        if self._chi_min_a < 1e-8:
            self._chi_min_a = 1e-8
        self._chi_max_a = self.kernel.cosmo.comoving_distance(self._z_max_a)
        self._chi_min_b = self.kernel.cosmo.comoving_distance(self._z_min_b)
        if self._chi_min_b < 1e-8:
            self._chi_min_b = 1e-8
        self._chi_max_b = self.kernel.cosmo.comoving_distance(self._z_max_b)

        self.D_z_NG = self.kernel.cosmo.growth_factor(self.kernel.z_bar_NG)

        self.halo_a = input_correlation_a.halo
        self.halo_b = input_correlation_b.halo
        if input_halo_trispectrum is None:
            input_halo_trispectrum = halo_trispectrum.HaloTrispectrumOneHalo()
        self.halo_tri = input_halo_trispectrum
        # self.halo.set_redshift(self.kernel.z_bar_G)
        # self.halo_tri.set_redshift(self.kernel.z_bar_NG)
        
        self._initialized_halo_splines = False
        self._ln_k_min = numpy.log(defaults.default_limits['k_min'])
        self._ln_k_max = numpy.log(defaults.default_limits['k_max'])
        self._ln_K_min = numpy.log(numpy.min(
            [defaults.default_limits['k_min']*self._chi_min_a,
             defaults.default_limits['k_min']*self._chi_min_b]))
        self._ln_K_max = numpy.log(numpy.max(
            [defaults.default_limits['k_max']*self._chi_max_a,
             defaults.default_limits['k_max']*self._chi_max_b]))
                                  
        self._ln_k_array = numpy.linspace(
            self._ln_k_min, self._ln_k_max,
            defaults.default_precision["kernel_npoints"])
        self._ln_K_array = numpy.linspace(
            self._ln_K_min, self._ln_K_max,
            defaults.default_precision["kernel_npoints"])
        
        self._int_G_norm = 1.0
        self._current_theta_a = -1.0
        self._current_theta_b = -1.0
        
        self._j0_limit = special.jn_zeros(
            0, defaults.default_precision["kernel_bessel_limit"])[-1]
            
        if power_spec==None:
            power_spec = 'linear_power'
        try:
            tmp = self.halo_a.__getattribute__(power_spec)
            self.power_spec = power_spec
        except AttributeError or TypeError:
            print "WARNING: Invalid input for power spectra variable,"
            print "\t setting to linear_power"
            self.power_spec = 'linear_power'

    def _identify_cosmic_shear(self):
        shear_a1 = isinstance(self.kernel.window_function_a1,
                              kernel.WindowFunctionConvergence)
        shear_a2 = isinstance(self.kernel.window_function_a2,
                              kernel.WindowFunctionConvergence)
        shear_b1 = isinstance(self.kernel.window_function_b1,
                              kernel.WindowFunctionConvergence)
        shear_b2 = isinstance(self.kernel.window_function_b2,
                              kernel.WindowFunctionConvergence)
        return [shear_a1*shear_a2 or shear_b1*shear_b2,
                shear_a1*shear_b2 or shear_a2*shear_b1]
               
    def _projected_halo_a(self, K):
        """
        Wrapper class for the 2-D projected power spectrum of correlation a
        """
        if not self._initialized_halo_splines:
            self._initialize_halo_splines()
        return self._halo_a_spline(numpy.log(K))
    
    def _projected_halo_b(self, K):
        """
        Wrapper class for the 2-D projected power spectrum of correlation b
        """
        if not self._initialized_halo_splines:
            self._initialize_halo_splines()
        if self.matching_corrs:
            return self._halo_a_spline(numpy.log(K))
        else:
            return self._halo_b_spline(numpy.log(K))

    def _projected_halo_ab(self, K):
        """
        Wrapper class for the 2-D projected power spectrum of the 
        cross-correlation of a and b
        """
        if not self._initialized_halo_splines:
            self._initialize_halo_splines()
        return self._halo_ab_spline(numpy.log(K))

    def _projected_halo_ba(self, K):
        """
        Wrapper class for the 2-D projected power spectrum of the 
        cross-correlation of a and b
        """
        if not self._initialized_halo_splines:
            self._initialize_halo_splines()
        return self._halo_ba_spline(numpy.log(K))

    def set_cosmology(self, cosmo_dict):
        """
        Reset the cosmology

        Args:
            cosmo_dict: dictionary of floats defining a cosmology (see
                defaults.py for details)
        """
        self.kernel.set_cosmology(cosmo_dict)
        self._chi_min_a = self.kernel.cosmo.comoving_distance(self._z_min_a)
        if self._chi_min_a < 1e-8:
            self._chi_min_a = 1e-8
        self._chi_max_a = self.kernel.cosmo.comoving_distance(self._z_max_a)
        self._chi_min_b = self.kernel.cosmo.comoving_distance(self._z_min_b)
        if self._chi_min_b < 1e-8:
            self._chi_min_b = 1e-8
        self._chi_max_b = self.kernel.cosmo.comoving_distance(self._z_max_b)

        self.D_z_NG = self.kernel.cosmo.growth_factor(self.kernel.z_bar_NG)
        self.corr_a.set_cosmology(cosmo_dict)
        self.corr_b.set_cosmology(cosmo_dict)

        self.halo_a = self.corr_a.halo
        self.halo_b = self.corr_b.halo
        self.halo_tri.set_cosmology(cosmo_dict, self.kernel.z_bar_NG)
        self._initialized_halo_splines = False
        self._initialized_halo_splines = False
        return None

    def get_cosmology(self):
        return self.kernel.get_cosmology()
    
    def get_covariance(self):
        """
        Compute all covariance compoments and bins and return a square ndarry
        of dimensions (nbins, nbins) containing the ouput values.
        
        Returns:
            a float array
        """
        self.covar = numpy.empty((len(self.annular_bins),
                                  len(self.annular_bins)), 'float128')
        for idx1 in xrange(self.covar.shape[0]):
            for idx2 in xrange(idx1, self.covar.shape[1]):
                cov = self.covariance(self.annular_bins[idx1],
                                      self.annular_bins[idx2])
                if idx1 == idx2:
                    self.covar[idx1, idx2] = cov
                else:
                    self.covar[idx1, idx2] = cov
                    self.covar[idx2, idx1] = cov
        return self.covar
        
    def covariance(self, annular_bin_a, annular_bin_b):
        """
        Compute the covariance for a given pair of angular bins.
        
        Args:
            annular_bin_a: an annular bin class object
            annular_bin_b: an annular bin class object
        Returns:
            float value of the covariance
        """
        cov_P = 0.0
        theta_a = annular_bin_a.center*deg_to_rad
        theta_b = annular_bin_b.center*deg_to_rad
        if annular_bin_a == annular_bin_b and self.matching_corrs:
            cov_P = self.covariance_P(annular_bin_a.delta, theta_a)
        res = self.covariance_G(theta_a, theta_b,
                                annular_bin_a.delta, annular_bin_b.delta)
        if self.nongaussian_cov:
            res += self.covariance_NG(theta_a, theta_b)
        return res + cov_P
    
    def covariance_P(self, delta, theta, window_1=0, window_2=1):
        """
        Poisson covariance term. Computes the pure poisson term using the input
        survey parameters.
        
        Args:
            delta: outer - inner radius of the annular bin
        Returns:
            float poisson covariance
        """
        Poiss_a = self.proj_power_poisson(window_pair=0)
        Poiss_b = self.proj_power_poisson(window_pair=2)
        shot_noise_wt = 1. + self.cosmic_shear[0]
        term1 = Poiss_a * Poiss_b * shot_noise_wt
        Poiss_a = self.proj_power_poisson(window_pair=3)
        Poiss_b = self.proj_power_poisson(window_pair=1)
        shot_noise_wt = 1. + self.cosmic_shear[1]
        term2 = Poiss_a * Poiss_b * shot_noise_wt
        return (term1 + term2) / (2.*numpy.pi*self.area*theta*delta)

    def proj_power_poisson(self, window_pair=0):
        if self.equal_windows[window_pair]:
            res = self.variance * self.variance / self.density[window_pair]
        else:
            res = 0.0
        return res
        
    def covariance_G(self, theta_a, theta_b, delta_a, delta_b):
        """
        Gaussian error term of the covariance given input theta bins
        
        Args:
            theta_a: center of bin at which to compute the covariance
            theta_b: center of bin at which to compute the covariance
        Returns:
            float gaussian covariance
        """
        
        ### We normalize the integral so that romberg will have an easier time
        ### integrating it.
        if not self._initialized_halo_splines:
            self._initialize_halo_splines()
        ln_K_max = numpy.log(numpy.max([self._j0_limit/theta_a,
                                        self._j0_limit/theta_b]))
        if ln_K_max > self._ln_K_max:
            ln_K_max = self._ln_K_max
        elif ln_K_max <= self._ln_K_min:
            return 0.0
        
        norm = 1.0/self._covariance_G_integrand(0.0, 0.0, 0.0, 1.0, 1.0)
        return integrate.romberg(
            self._covariance_G_integrand, self._ln_K_min, ln_K_max,
            args=(theta_a, theta_b, delta_a, norm), vec_func=True, 
            tol=defaults.default_precision["global_precision"],
            rtol=defaults.default_precision["corr_precision"],
            divmax=defaults.default_precision["divmax"])/(
                norm*2.*numpy.pi*self.area)
    
    def _covariance_G_J07_integrand(self, ln_K, theta_a, theta_b,
                                    delta_a, delta_b, norm=1.0):
        """
        Internal function defining the integrand for the gaussian covariance.
        Covariance terms are from Joachimi et al. 2007
        """
        K = numpy.exp(ln_K)
        dK = K

        if self.matching_corrs:
            Pa = self._projected_halo_a(K)/self._D_z_a**2
            Pb = self._projected_halo_b(K)/self._D_z_b**2
            two_point_term1 = Pa*Pb
            two_point_term2 = two_point_term1
        
            two_point_terms = two_point_term1 + two_point_term2
            return (dK*K*norm*two_point_terms*
                    special.j0(K*theta_a)*special.j0(K*theta_b))
        else:
            Pab = self._projected_halo_ab(K)
            two_point_term1 = Pab*Pab
            two_point_term2 = two_point_term1
        
            two_point_terms = two_point_term1 + two_point_term2
            return (dK*K*norm*two_point_terms*
                    special.j0(K*theta_a)*special.j0(K*theta_b))
            
    def _covariance_G_integrand(self, ln_K, theta_a, theta_b,
                                delta, norm=1.0):
        """
        Internal function defining the integrand for the gaussian covariance.

        This is general for any 2-point (cross-)correlation obtained with any
        combination of window functions in the member correlation objects.
        This CANNOT be used for cosmic shear with a nonzero B-mode (see eq. 34
        in reference [1]).

        Args:
            ln_K: natural logarithm of the projected wavenumber in
                inverse radians.
            theta_a: central value of the first angular bin in radians.
            theta_b: central value of the second angular bin in radians.
            delta: outer - inner radius of the annular bin.
            norm: multiplicative normalization of the integrand.

        Note that delta is the width of a single theta bin, NOT the area of 
        an annulus in theta, as enters the shot noise term for the covariance.
        These are related by: dOmega = 2 pi theta dtheta
        An extra factor of theta is obtained when integrating the product of 
        Bessel functions in the Bessel "closure equation".

        References:
        [1] B. Joachimi, P. Schneider, and T. Eifler,
        "Analysis of two-point statistics of cosmic shear:
        III. Covariances of shear measures made easy,"
        in arXiv astro-ph (2007) [doi:10.1051/0004-6361:20078400].
        """
        K = numpy.exp(ln_K)
        dK = K

        Pa = self._projected_halo_a(K) / (self._D_z_a ** 2)
        Pb = self._projected_halo_b(K) / (self._D_z_b ** 2)
        Poiss_a = self.proj_power_poisson(window_pair=0)
        Poiss_b = self.proj_power_poisson(window_pair=2)
        shot_noise_wt = 1. + self.cosmic_shear[0]
        # two_point_term1 = Pa * Pb
        two_point_term1 = (Pa * Pb +
            Pa * Poiss_b +
            Pb * Poiss_a)
            # Poiss_a * Poiss_b * shot_noise_wt / delta)

        # print Pa*Pb, Poiss_a * Poiss_b * shot_noise_wt / delta
        
        if self.matching_corrs:
            two_point_term2 = two_point_term1
        else:
            Pab = self._projected_halo_ab(K) / (self._D_z_a * self._D_z_b)
            Pba = self._projected_halo_ba(K) / (self._D_z_a * self._D_z_b)
            Poiss_a = self.proj_power_poisson(window_pair=3)
            Poiss_b = self.proj_power_poisson(window_pair=1)
            shot_noise_wt = 1. + self.cosmic_shear[1]
            # two_point_term2 = Pab * Pba
            two_point_term2 = (Pab * Pba +
                Pab * Poiss_b +
                Pba * Poiss_a)
                # Poiss_a * Poiss_b * shot_noise_wt / delta) 
        two_point_terms = two_point_term1 + two_point_term2
        return (dK*K*norm*two_point_terms*special.j0(K*theta_a)*
                special.j0(K*theta_b))
        
    def _initialize_halo_splines(self):
        """
        Internal method for initializing and projecting the input power spectra.
        Initializes the splines used in _projected_halo methods.
        """
        self._z_bar_G_a = self.corr_a.kernel.z_bar
        self._z_bar_G_b = self.corr_b.kernel.z_bar
        
        print "Mean Redshifts:", self._z_bar_G_a, self._z_bar_G_b
        
        self.halo_a.set_redshift(self._z_bar_G_a)
        self.halo_b.set_redshift(self._z_bar_G_b)
        
        self._D_z_a = self.kernel.cosmo.growth_factor(self._z_bar_G_a)
        self._D_z_b = self.kernel.cosmo.growth_factor(self._z_bar_G_b)
        chi_peak_a = self.kernel.cosmo.comoving_distance(self._D_z_a)
        chi_peak_b = self.kernel.cosmo.comoving_distance(self._D_z_b)
        
        _halo_a_array = numpy.empty(self._ln_K_array.shape)
        _halo_b_array = numpy.empty(self._ln_K_array.shape)
        _halo_ab_array = numpy.empty(self._ln_K_array.shape)        
        _halo_ba_array = numpy.empty(self._ln_K_array.shape)        
        
        for idx, ln_K in enumerate(self._ln_K_array):
            chi_min = numpy.exp(ln_K)/defaults.default_limits['k_max']
            if chi_min < self._chi_min_a:
                chi_min = self._chi_min_a   
            chi_max = numpy.exp(ln_K)/defaults.default_limits['k_min']
            if chi_max > self._chi_max_a:
                chi_max = self._chi_max_a

            norm_int = self._halo_a_integrand(chi_peak_a, ln_K, norm=1.0)
            norm = numpy.where(norm_int > 0., 1.0 / norm_int, 1.0)
            _halo_a_array[idx] = integrate.romberg(
                self._halo_a_integrand, chi_min, chi_max,
                args=(ln_K, norm), vec_func=True,
                tol=defaults.default_precision["global_precision"],
                rtol=defaults.default_precision["corr_precision"],
                divmax=defaults.default_precision["divmax"])/norm

            if not self.matching_corrs:
                chi_min = numpy.exp(ln_K)/defaults.default_limits['k_max']
                if chi_min < self._chi_min_b:
                    chi_min = self._chi_min_b  
                chi_max = numpy.exp(ln_K)/defaults.default_limits['k_min']
                if chi_max > self._chi_max_b:
                    chi_max = self._chi_max_b  
            
                norm_int = self._halo_b_integrand(chi_peak_b, ln_K, norm=1.0)
                norm = numpy.where(norm_int > 0., 1.0 / norm_int, 1.0)
                _halo_b_array[idx] = integrate.romberg(
                    self._halo_b_integrand, chi_min, chi_max,
                    args=(ln_K, norm), vec_func=True,
                    tol=defaults.default_precision["global_precision"],
                    rtol=defaults.default_precision["corr_precision"],
                    divmax=defaults.default_precision["divmax"])/norm

                if chi_min < self._chi_min_a:
                    chi_min = self.chi_min_a
                if chi_max > self._chi_max_a:
                    chi_max = self._chi_max_a                
                norm_int = numpy.sqrt(
                    self._halo_a_integrand(chi_peak_a, ln_K, norm=1.0) *
                    self._halo_b_integrand(chi_peak_b, ln_K, norm=1.0))
                norm = numpy.where(norm_int > 0., 1.0 / norm_int, 1.0)
                _halo_ab_array[idx] = integrate.romberg(
                    self._halo_ab_integrand, chi_min, chi_max,
                    args=(ln_K, norm), vec_func=True,
                    tol=defaults.default_precision["global_precision"],
                    rtol=defaults.default_precision["corr_precision"],
                    divmax=defaults.default_precision["divmax"])/norm
                _halo_ba_array[idx] = integrate.romberg(
                    self._halo_ba_integrand, chi_min, chi_max,
                    args=(ln_K, norm), vec_func=True,
                    tol=defaults.default_precision["global_precision"],
                    rtol=defaults.default_precision["corr_precision"],
                    divmax=defaults.default_precision["divmax"])/norm
            
        self._halo_a_spline = InterpolatedUnivariateSpline(
            self._ln_K_array, _halo_a_array)
        if not self.matching_corrs:
            self._halo_b_spline = InterpolatedUnivariateSpline(
                self._ln_K_array, _halo_b_array)
            self._halo_ab_spline = InterpolatedUnivariateSpline(
                self._ln_K_array, _halo_ab_array)
            self._halo_ba_spline = InterpolatedUnivariateSpline(
                self._ln_K_array, _halo_ba_array)
        
        self._initialized_halo_splines = True
    
    def _halo_a_integrand(self, chi, ln_K, norm=1.0):
        """
        Integrand for the halo from input_correlation_a using it's window
        functions as the projection in redshift.
        """
        K = numpy.exp(ln_K)
        return (norm*self.halo_a.__getattribute__(self.power_spec)(K/chi)*
                self.kernel._kernel_G_a_integrand(chi))
    
    def _halo_b_integrand(self, chi, ln_K, norm=1.0):
        """
        Integrand for the halo from input_correlation_b using it's window
        functions as the projection in redshift.
        """
        K = numpy.exp(ln_K)
        return (norm*self.halo_b.__getattribute__(self.power_spec)(K/chi)*
                self.kernel._kernel_G_b_integrand(chi))

    def _halo_ab_integrand(self, chi, ln_K, norm=1.0):
        """
        Integrand for the projected power spectrum of the cross-correlation 
        of a and b samples using the first window of the a sample and
        second window of the b sample.
        """
        K = numpy.exp(ln_K)
        Pa = self.halo_a.__getattribute__(self.power_spec)(K/chi)
        Pb = self.halo_b.__getattribute__(self.power_spec)(K/chi)
        # For lack of a cross-power implementation in Halo class,
        # use the geometric mean
        power = numpy.sqrt(Pa * Pb)
        return (norm*power*
                self.kernel._kernel_G_ab_integrand(chi))

    def _halo_ba_integrand(self, chi, ln_K, norm=1.0):
        """
        Integrand for the projected power spectrum of the cross-correlation 
        of a and b samples using the second window of the a sample and
        first window of the b sample.
        """
        K = numpy.exp(ln_K)
        Pa = self.halo_a.__getattribute__(self.power_spec)(K/chi)
        Pb = self.halo_b.__getattribute__(self.power_spec)(K/chi)
        # For lack of a cross-power implementation in Halo class,
        # use the geometric mean
        power = numpy.sqrt(Pa * Pb)
        return (norm*power*
                self.kernel._kernel_G_ba_integrand(chi))
        
    def covariance_NG(self, theta_a_rad, theta_b_rad):
        """
        Compute the non-gaussian covariance from the halo model trispectrum.
        
        Args:
            theta_a_rad: Input annular bin center in radians
            theta_b_rad: Input annular bin center in radians
        Returns:
            float value of non-gaussian covariance
        """
        self._initialize_kb_spline(theta_a_rad, theta_b_rad)
        
        norm = 1.0/self._ka_integrand(0.0, 1.0)
        
        return integrate.romberg(
            self._ka_integrand, self._ln_k_min, self._ln_k_max, vec_func=True,
            args=(norm,),
            tol=defaults.default_precision["global_precision"],
            rtol=defaults.default_precision["corr_precision"],
            divmax=defaults.default_precision["divmax"])/(
                4.0*numpy.pi*numpy.pi*norm*self.area)
        
    def _ka_integrand(self, ln_ka, norm=1.0):
        """
        Internal function defining the integrand over one of the k compoments.
        """
        dln_ka = 1.0
        ka = numpy.exp(ln_ka)
        dka = ka*dln_ka
        return dka*ka*self._kb_spline(ln_ka)*norm
    
    def _initialize_kb_spline(self, theta_a, theta_b):
        """
        Internal function initializing the spline in ka by integrating over kb.
        """
        if (self._current_theta_a == theta_a and
            self._current_theta_b == theta_b):
            return None
        
        _kb_int_array = numpy.empty(self._ln_k_array.shape, 'float128')
        
        for idx, ln_k in enumerate(self._ln_k_array):
            _kb_int_array[idx] = self._kb_integral(ln_k, theta_a, theta_b)
            
        self._kb_min = numpy.min(_kb_int_array)
        self._kb_spline = InterpolatedUnivariateSpline(
            self._ln_k_array, _kb_int_array)
    
    def _kb_integral(self, ln_k, theta_a, theta_b):
        """
        Internal function defining the integral over kb for different vector and
        non-vector input cases.
        """
        if type(ln_k) == numpy.ndarray:
            kb_int = numpy.empty(ln_k.shape)
            
            inv_norm = self._kb_integrand(0.0, ln_k, theta_a, 
                                          theta_b, 1.0)
            norm = 1.0
            for idx, ln_k in enumerate(ln_k):
                kb_int[idx] = integrate.romberg(
                    self._kb_integrand, self._ln_k_min, self._ln_k_max,
                    args=(ln_k, theta_a, theta_b, norm), vec_func=True,
                    tol=defaults.default_precision["global_precision"],
                    rtol=defaults.default_precision["corr_precision"],
                    divmax=defaults.default_precision["divmax"])/(
                        norm*self.D_z_NG*self.D_z_NG*self.D_z_NG*self.D_z_NG)
            return kb_int

        inv_norm = self._kb_integrand(0.0, ln_k, theta_a, 
                                          theta_b, 1.0)
        norm = 1.0
        return integrate.romberg(
           self._kb_integrand, self._ln_k_min, self._ln_k_max,
           args=(ln_k, theta_a, theta_b, norm), vec_func=True,
           tol=defaults.default_precision["global_precision"],
           rtol=defaults.default_precision["corr_precision"],
           divmax=defaults.default_precision["divmax"])/(
               norm*self.D_z_NG*self.D_z_NG*self.D_z_NG*self.D_z_NG)
    
    def _kb_integrand(self, ln_kb, ln_ka, theta_a, theta_b, norm=1.0):
        """
        Internal function defining the integrand over kb
        """
        dln_kb = 1.0
        ka = numpy.exp(ln_ka)
        kb = numpy.exp(ln_kb)
        dkb = kb*dln_kb
        return (dkb*kb*norm*self.halo_tri.trispectrum_parallelogram(ka, kb)*
            self.kernel.kernel(numpy.log(ka*theta_a),
                               numpy.log(kb*theta_b))[0])
    
    def write(self, file_name):
        """
        Write the computed covariance to a file.
        
        Args:
            file_name: string name of file to output to.
        """
        f = open(file_name, 'w')
        f.write("#ttype1 = theta_a [deg]\n#ttype2 = theta_b [deg]\n" + 
                "#ttype3 = covariance\n")
        for idx_a, bin_a in enumerate(self.annular_bins):
            for idx_b, bin_b in enumerate(self.annular_bins):
                f.writelines('%1.16f %1.16f %1.16f\n' % (
                             bin_a.center*rad_to_deg, bin_b.center*rad_to_deg,
                             self.covar[idx_a, idx_b]))
        f.close()
        
        
class CovarianceMulti(Covariance):
    """
    Wrapper class for Covariance allowing for computation of both
    auto-covaraince and the cross covariance between different correlatoins.
    Takes as input a list of correlation objects and returns creates an output
    covariance respresenting all the auto and cross terms.
    
    Attributes:
        correlation_object_list: a list of correlation objects from
            correlation.py
        bins_per_decade: int log spacing of the theta bins
        survey_area_deg2: float value of the survey area it square degrees
        n_pairs: float number of pairs to compute the poisson term
        variance: float variance per pair
        nongaussian_cov: bool, toggles nonguassian covariance
        input_halo_trispectrum: HaloTrispectrum object from halo_trispectrum.py
        
        covariance_list: list of covariance objects using all non-repeating
            pairs of correlation_objects and the input survey parameters.
        theta_bins: number of annular bins
        wcovar: ouput numpy array. has dimensions
            (theta_bins*len(correlation_list), theta_bins*len(correlation_list))
    """
    
    def __init__(self, correlation_object_list, bins_per_decade=5,
                 survey_area_deg2=4*numpy.pi*strad_to_deg2,
                 n_pairs=1e6*1e6, variance=1.0, nongaussian_cov=True,
                 input_halo_trispectrum=None, **kws):
        self.covariance_list = []
        n_covars = 0
        for idx1 in xrange(len(correlation_object_list)):
            tmp_list = []
            for idx2 in xrange(idx1, len(correlation_object_list)):
                tmp_list.append(Covariance(
                    correlation_object_list[idx1],
                    correlation_object_list[idx2], bins_per_decade,
                    survey_area_deg2, n_pairs, variance, nongaussian_cov,
                    input_halo_trispectrum))
                tmp_list.append
                n_covars += 1
            self.covariance_list.append(tmp_list)
        self.theta_bins = len(self.covariance_list[0][0].annular_bins)
        n_corrs = 1
        self.wcovar = numpy.empty((
            self.theta_bins*len(correlation_object_list),
            self.theta_bins*len(correlation_object_list)))
        
    def get_covariance(self):
        """
        Wrapper class for looping over each of the different covariance blocks.
        """
        for idx1, row in enumerate(self.covariance_list):
            for idx2, cov in enumerate(row):
                print idx1, idx2
                cov.get_covariance()
                print cov.covar
                self.wcovar[idx1*self.theta_bins:(idx1+1)*self.theta_bins,
                            (idx1+idx2)*self.theta_bins:(idx1+idx2+1)*
                            self.theta_bins] = cov.covar
                self.wcovar[(idx1+idx2)*self.theta_bins:(idx1+idx2+1)*
                            self.theta_bins, idx1*self.theta_bins:(idx1+1)*
                            self.theta_bins] = cov.covar
        print self.wcovar
                
class CovarianceFourier(object):
    
    def __init__(self, l_min, l_max, input_kernel_covariance=None,
                 input_halo=None, input_halo_trispectrum=None, **kws):
        
        self._ln_l_min = numpy.log(l_min)
        self._ln_l_max = numpy.log(l_max)
        self._ln_l_array = numpy.linspace(
            self._ln_l_min, self._ln_l_max,
            defaults.default_precision["corr_npoints"])
        
        self.kernel = input_kernel_covariance
        
        self._z_min_a1a2 = numpy.max([self.kernel.window_function_a1.z_min,
                                      self.kernel.window_function_a2.z_min])
        self._z_min_b1b2 = numpy.max([self.kernel.window_function_b1.z_min,
                                      self.kernel.window_function_b2.z_min])
        self._z_min_a1b2 = numpy.max([self.kernel.window_function_a1.z_min,
                                      self.kernel.window_function_b2.z_min])
        self._z_min_b1a2 = numpy.max([self.kernel.window_function_b1.z_min,
                                      self.kernel.window_function_a2.z_min])
        
        self._z_max_a1a2 = numpy.min([self.kernel.window_function_a1.z_max,
                                      self.kernel.window_function_a2.z_max])
        self._z_max_b1b2 = numpy.min([self.kernel.window_function_b1.z_max,
                                      self.kernel.window_function_b2.z_max])
        self._z_max_a1b2 = numpy.min([self.kernel.window_function_a1.z_max,
                                      self.kernel.window_function_b2.z_max])
        self._z_max_b1a2 = numpy.min([self.kernel.window_function_b1.z_max,
                                      self.kernel.window_function_a2.z_max])
        
        self._z_array = numpy.linspace(
            numpy.min([self._z_min_a1a2, self._z_min_b1b2,
                       self._z_min_a1b2, self._z_min_b1a2]),
            numpy.max([self._z_max_a1a2, self._z_max_b1b2,
                       self._z_max_a1b2, self._z_max_b1a2]),
            defaults.default_precision['kernel_npoints'])
        
        self.window_a1 = self.kernel.window_function_a1.window_function
        self.window_a2 = self.kernel.window_function_a2.window_function
        self.window_b1 = self.kernel.window_function_b1.window_function
        self.window_b2 = self.kernel.window_function_b2.window_function
        
        self.halo_a1a2 = input_halo
        self.halo_b1b2 = copy(input_halo)
        self.halo_a1b2 = copy(input_halo)
        self.halo_b1a2 = copy(input_halo)
        self.halo_tri = input_halo_trispectrum
        
        self._initialized_pl = False
        
    def covariance(self, l_a, l_b):
        pass
        
    def covariance_G(self, l):
        if not self._initialized_pl:
            self._initialize_pl()
        return 1.0/(2.0*l + 1.0)*(self._pl_a1a2(l)*self._pl_b1b2(l)+
                self._pl_a1b2(l)*self._pl_b1a2(l))
                
    def _pl_a1a2(self, l):
        ln_l = numpy.log(l)
        return numpy.where(
            numpy.logical_and(ln_l >= self._ln_l_min, ln_l <= self._ln_l_max),
            numpy.exp(self._a1a2_spline(ln_l))/self._norm_G_a1a2, 0.0)
   
    def _pl_b1b2(self, l):
        ln_l = numpy.log(l)
        return numpy.where(
            numpy.logical_and(ln_l >= self._ln_l_min, ln_l <= self._ln_l_max),
            numpy.exp(self._b1b2_spline(ln_l))/self._norm_G_b1b2, 0.0)
            
    def _pl_a1b2(self, l):
        ln_l = numpy.log(l)
        return numpy.where(
            numpy.logical_and(ln_l >= self._ln_l_min, ln_l <= self._ln_l_max),
            numpy.exp(self._a1b2_spline(ln_l))/self._norm_G_a1b2, 0.0)
            
    def _pl_b1a2(self, l):
        ln_l = numpy.log(l)
        return numpy.where(
            numpy.logical_and(ln_l >= self._ln_l_min, ln_l <= self._ln_l_max),
            numpy.exp(self._b1a2_spline(ln_l))/self._norm_G_b1a2, 0.0)
            
    def _initialize_pl(self):
        self._z_bar_G_a1a2 = self._calculate_zbar(self.window_a1,
                                                  self.window_a2)
        self._z_bar_G_b1b2 = self._calculate_zbar(self.window_b1,
                                                  self.window_b2)
        self._z_bar_G_a1b2 = self._calculate_zbar(self.window_a1,
                                                  self.window_b2)
        self._z_bar_G_b1a2 = self._calculate_zbar(self.window_b1,
                                                  self.window_a2)
        
        self.halo_a1a2.set_redshift(self._z_bar_G_a1a2)
        self.halo_b1b2.set_redshift(self._z_bar_G_b1b2)
        self.halo_a1b2.set_redshift(self._z_bar_G_a1b2)
        self.halo_b1a2.set_redshift(self._z_bar_G_b1a2)
        chi_a1a2 = self.kernel.cosmo.comoving_distance(self._z_bar_G_a1a2)
        chi_b1b2 = self.kernel.cosmo.comoving_distance(self._z_bar_G_b1b2)
        chi_a1b2 = self.kernel.cosmo.comoving_distance(self._z_bar_G_a1b2)
        chi_b1a2 = self.kernel.cosmo.comoving_distance(self._z_bar_G_b1a2)
            
        chi_a1a2_min = self.kernel.cosmo.comoving_distance(self._z_min_a1a2)
        chi_b1b2_min = self.kernel.cosmo.comoving_distance(self._z_min_b1b2)
        chi_a1b2_min = self.kernel.cosmo.comoving_distance(self._z_min_a1b2)
        chi_b1a2_min = self.kernel.cosmo.comoving_distance(self._z_min_b1a2)
        
        chi_a1a2_max = self.kernel.cosmo.comoving_distance(self._z_max_a1a2)
        chi_b1b2_max = self.kernel.cosmo.comoving_distance(self._z_max_b1b2)
        chi_a1b2_max = self.kernel.cosmo.comoving_distance(self._z_max_a1b2)
        chi_b1a2_max = self.kernel.cosmo.comoving_distance(self._z_max_b1a2)
            
        self._norm_G_a1a2= 1.0/self._pl_integrand(chi_a1a2,
                                                  numpy.log(chi_a1a2),
                                                  self.halo_a1a2,
                                                  self.window_a1,
                                                  self.window_a2, 1.0)
        self._norm_G_b1b2= 1.0/self._pl_integrand(chi_b1b2,
                                                  numpy.log(chi_b1b2),
                                                  self.halo_b1b2,
                                                  self.window_a1,
                                                  self.window_a2, 1.0)
        self._norm_G_a1b2= 1.0/self._pl_integrand(chi_a1b2,
                                                  numpy.log(chi_a1b2),
                                                  self.halo_a1a2,
                                                  self.window_a1,
                                                  self.window_a2, 1.0)
        self._norm_G_b1a2= 1.0/self._pl_integrand(chi_b1a2,
                                                  numpy.log(chi_b1a2),
                                                  self.halo_b1a2,
                                                  self.window_a1,
                                                  self.window_a2, 1.0)
        
        _pl_a1a2 = numpy.empty(self._ln_l_array.shape)
        _pl_b1b2 = numpy.empty(self._ln_l_array.shape)
        _pl_a1b2 = numpy.empty(self._ln_l_array.shape)
        _pl_b1a2 = numpy.empty(self._ln_l_array.shape)
        for idx, ln_l in enumerate(self._ln_l_array):
            _pl_a1a2[idx] = integrate.romberg(
                self._pl_integrand, chi_a1a2_min, chi_a1a2_max,
                args=(ln_l, self.halo_a1a2, self.window_a1, self.window_a2,
                      self._norm_G_a1a2),
                vec_func=True,
                tol=defaults.default_precision["global_precision"],
                rtol=defaults.default_precision["corr_precision"],
                divmax=defaults.default_precision["divmax"])
            _pl_b1b2[idx] = integrate.romberg(
                self._pl_integrand, chi_b1b2_min, chi_b1b2_max,
                args=(ln_l, self.halo_b1b2, self.window_b1, self.window_b2,
                      self._norm_G_b1b2),
                vec_func=True,
                tol=defaults.default_precision["global_precision"],
                rtol=defaults.default_precision["corr_precision"],
                divmax=defaults.default_precision["divmax"])
            _pl_a1b2[idx] = integrate.romberg(
                self._pl_integrand, chi_a1b2_min, chi_a1b2_max,
                args=(ln_l, self.halo_a1b2, self.window_a1, self.window_b2,
                      self._norm_G_a1b2),
                vec_func=True,
                tol=defaults.default_precision["global_precision"],
                rtol=defaults.default_precision["corr_precision"],
                divmax=defaults.default_precision["divmax"])
            _pl_b1a2[idx] = integrate.romberg(
                self._pl_integrand, chi_b1a2_min, chi_b1a2_max,
                args=(ln_l, self.halo_b1a2, self.window_b1, self.window_a2,
                      self._norm_G_b1a2),
                vec_func=True,
                tol=defaults.default_precision["global_precision"],
                rtol=defaults.default_precision["corr_precision"],
                divmax=defaults.default_precision["divmax"])
            
        print _pl_a1a2
                      
        self._a1a2_spline = InterpolatedUnivariateSpline(
            self._ln_l_array,
            numpy.log(_pl_a1a2/
                      self.kernel.cosmo.growth_factor(self._z_bar_G_a1a2)**2))
        self._b1b2_spline = InterpolatedUnivariateSpline(
            self._ln_l_array,
            numpy.log(_pl_b1b2/
                      self.kernel.cosmo.growth_factor(self._z_bar_G_b1b2)**2))
        self._a1b2_spline = InterpolatedUnivariateSpline(
            self._ln_l_array,
            numpy.log(_pl_a1b2/
                      self.kernel.cosmo.growth_factor(self._z_bar_G_a1b2)**2))
        self._b1a2_spline = InterpolatedUnivariateSpline(
            self._ln_l_array,
            numpy.log(_pl_b1a2/
                      self.kernel.cosmo.growth_factor(self._z_bar_G_b1a2)**2))
        
        self._initialized_pl = True      
        
    def _calculate_zbar(self, window1, window2):
        func = lambda chi: (window1(chi)*window2(chi)/(chi*chi)*
                            self.kernel.cosmo.growth_factor(
                                self.kernel.cosmo.redshift(chi))*
                            self.kernel.cosmo.growth_factor(
                                self.kernel.cosmo.redshift(chi)))
        print func(self.kernel.cosmo.comoving_distance(self._z_array))
        return self._z_array[numpy.argmax(
            func(self.kernel.cosmo.comoving_distance(self._z_array)))]
        
    def _pl_integrand(self, chi, ln_l, halo, window1, window2, norm):
        l = numpy.exp(ln_l)
        k = l/chi
        D_z = self.kernel.cosmo.growth_factor(
            self.kernel.cosmo.redshift(chi))
        return (norm*window1(chi)*window2(chi)*D_z*D_z/(chi*chi)*
                halo.power_mm(k))
        
class AnnulusBin(object):
    """
    Container class for storing information on the inner and outer radii of an
    annular bin as well as the center value of the bin. The class also stores
    the delta r value from the inner to outer radii.
    
    Attributes:
        inner: inner radius of the annulus
        outer: outer radius of the annulus
        center: derived log mean distance
        diff: outer - inner difference
    """
    
    def __init__(self, inner, outer):
        self.inner = inner
        self.outer = outer
        self.center = numpy.power(10.0, 0.5*(numpy.log10(inner)+
                                             numpy.log10(outer)))
        self.delta = outer - inner