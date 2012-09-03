from copy import copy
import cosmology
import defaults
import halo
import hod
import kernel
import numpy
from numpy import vectorize
from scipy import special
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline

"""
Classes describing different correlation functions.

Each correlation function can be defined as

w(theta) = int(z, g_1(z)*g_2(z)*int(k, k/(2*pi) P_x(k, z) J(k*theta*chi(z))))

where P_x is a power spectrum from halo, J is the Bessel function, and g_1/g_2
are window functions with redshift. This class is the final wrapper method uses
each all of the combined classes of the code base together to predict observable
correlations functions.
"""

speed_of_light = 3*10**5
degToRad = numpy.pi/180.0

__author__ = ("Chris Morrison <morrison.chrisb@gmail.com>, "+
              "Ryan Scranton <ryan.scranton@gmail.com>")


class Correlation(object):
    """
    Bass class for correlation functions.

    Given a maximum and minimum angular extent in radians, two window functions
    from kernel.py, dictionaries defining the cosmology and halo properties,
    an input HOD from hod.py, and a requested power spectrum type, 
    returns the predicted correlation function.
    
    Derived classes should return an array of the projected variable (in this
    case theta in radians) and return the value of the correlation w.

    Attributes:
        theta_min: minimum angular extent in radians
        theta_max: maximum angular extent in radians
        input_kernel: Kernel object from kernel.py
        input_halo: Halo object from halo.py
        input_hod: HOD object from hod.py
        powSpec: string defining a power spectrum
        
        theta_array: array of theta values for computed correlation function
        wtheta_array: array of computed correlation values at theta_array values
    """

    def __init__(self, theta_min, theta_max, input_kernel,
                 input_halo=None, powSpec=None, **kws):

        self.log_theta_min = numpy.log10(theta_min)
        self.log_theta_max = numpy.log10(theta_max)
        self.theta_array = numpy.logspace(
            self.log_theta_min, self.log_theta_max,
            defaults.default_precision["corr_npoints"])
        if theta_min==theta_max:
            self.log_theta_min = numpy.log10(theta_min)
            self.log_theta_max = numpy.log10(theta_min)
            self.theta_array = numpy.array([theta_min])
        self.wtheta_array = numpy.zeros(self.theta_array.size)

        # Hard coded, but we shouldn't expect halos outside of this range.

        self.kernel = input_kernel

        self.D_z = self.kernel.cosmo.growth_factor(self.kernel.z_bar)
                      
        if input_halo is None:
            input_halo = halo.Halo(self.kernel.z_bar)
        self.halo = input_halo
        self.halo.set_redshift(self.kernel.z_bar)

        if powSpec==None:
            powSpec = 'linear_power'
        try:
            self.power_spec = self.halo.__getattribute__(powSpec)
        except AttributeError or TypeError:
            print "WARNING: Invalid input for power spectra variable,"
            print "\t setting to linear_power"
            self.power_spec = self.halo.__getattribute__('linear_power')

    def set_redshift(self, redshift):
        """
        Force redshift of all objects to input value.

        Args:
            redshift: float value of redshift
        """
        self.kernel.z_bar = redshift
        self.D_z = self.kernel.cosmo.growth_factor(self.kernel.z_bar)
        self.halo.set_redshift(self.kernel.z_bar)
            
    def set_cosmology(self, cosmo_dict):
        """
        Set all objects to the cosmology of cosmo_dict

        Args:
            cosmo_dict: dictionary of float values defining a cosmology (see
                defaults.py for details)
        """
        self.kernel.set_cosmology(cosmo_dict)
        self.D_z = self.kernel.cosmo.growth_factor(self.kernel.z_bar)
        self.halo.set_cosmology(cosmo_dict, self.kernel.z_bar)

    def set_power_spectrum(self, powSpec):
        """
        Set power spectrum to type specified in powSpec. Of powSpec is not a
        member of the halo object return the linear power spectrum.

        Args:
            powSpec: string name of power spectrum to use from halo.py object.
        """
        try:
            self.power_spec = self.halo.__getattribute__(powSpec)
        except AttributeError or TypeError:
            print "WARNING: Invalid input for power spectra variable,"
            print "\t setting to 'linear_power'"
            self.power_spec = self.halo.__getattribute__('linear_power')

    def set_halo(self, halo_dict):
        """
        Reset halo parameters to halo_dict

        Args:
            halo_dict: dictionary of floats defining halos (see defaults.py
                for details)
        """
        self.halo.set_halo(halo_dict)

    def set_hod(self, input_hod):
        """
        Reset hod object to input_hod
        cosmo_dict: dictionary of floats defining a cosmology (see defaults.py
            for details)
        Args:
            input_hod: an HOD object from hod.py
        """
        self.halo.set_hod(input_hod)

    def compute_correlation(self):
        """
        Compute the value of the correlation over the range
        theta_min - theta_max
        """
        for idx,theta in enumerate(self.theta_array):
            self.wtheta_array[idx] = self.correlation(theta)

    def correlation(self, theta):
        """
        Compute the value of the correlation at array values theta

        Args:
            theta: float array of angular values in radians to compute the
                correlation
        """
        ln_kmin = numpy.log(self.halo._k_min)
        ln_kmax = numpy.log(self.halo._k_max)
        wtheta = integrate.romberg(
            self._correlation_integrand, 
            ln_kmin, ln_kmax, args=(theta,), vec_func=True,
            tol=defaults.default_precision["global_precision"],
            rtol=defaults.default_precision["corr_precision"],
            divmax=defaults.default_precision["divmax"])
        return wtheta

    def _correlation_integrand(self, ln_k, theta):
        dln_k = 1.0
        k = numpy.exp(ln_k)
        dk = k*dln_k
        return (dk*k/(2.0*numpy.pi)*self.power_spec(k)/(self.D_z*self.D_z)*
                self.kernel.kernel(numpy.log(k*theta)))

    def write(self, output_file_name):
        """
        Write out current values of the correlation object.

        Args:
            output_file_name: string name of file to output
        """
        f = open(output_file_name, "w")
        f.write("#ttype1 = theta [deg]\n#ttype2 = wtheta\n")
        for theta, wtheta in zip(
            self.theta_array, self.wtheta_array):
            f.write("%1.10f %1.10f\n" % (theta/degToRad, wtheta))
        f.close()
        
class Covariance(object):
    """
    Inherited class to compute the covariance matrix between theta_a and thata_b
    given input kernel and halo trispectrum objects. This class can be used to
    estimate the covariance between different estimators as a function.

    Attributes:
        theta_min: minimum angular extent in radians
        theta_max: maximum angular extent in radians
        input_kernel: KernelTrispectrum object from kernel.py
        input_halo: HaloTrispectrum object from halo.py
        input_hod: HOD object from hod.py
        
        theta_array: array of theta values for computed correlation function
        wcovar_array: array of computed covariance values at theta_array values
    """

    def __init__(self, theta_min, theta_max, 
                 input_kernel_covariance=None,
                 input_halo=None,
                 input_halo_trispectrum=None, **kws):

        self.log_theta_min = numpy.log10(theta_min)
        self.log_theta_max = numpy.log10(theta_max)
        self.theta_array = numpy.logspace(
            self.log_theta_min, self.log_theta_max,
            defaults.default_precision["corr_npoints"])
        if theta_min==theta_max:
            self.log_theta_min = numpy.log10(theta_min)
            self.log_theta_max = numpy.log10(theta_min)
            self.theta_array = numpy.array([theta_min])
        self.wcovar_array = numpy.zeros(self.theta_array.size)

        self.kernel = input_kernel_covariance
        
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
        
        self.halo_a = input_halo
        self.halo_b = copy(input_halo)
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
        
    def _initialize_halo_splines(self):
        z_array_a = numpy.linspace(self._z_min_a, self._z_max_a,
                                   defaults.default_precision['kernel_npoints'])
        z_array_b = numpy.linspace(self._z_min_b, self._z_max_b,
                                   defaults.default_precision['kernel_npoints'])
        self._z_bar_G_a = z_array_a[numpy.argmax(
            self.kernel._kernel_G_a_integrand(
                self.kernel.cosmo.comoving_distance(z_array_a)))]
        self._z_bar_G_b = z_array_a[numpy.argmax(
            self.kernel._kernel_G_b_integrand(
                self.kernel.cosmo.comoving_distance(z_array_a)))]
        
        self.halo_a.set_redshift(self._z_bar_G_a)
        self.halo_b.set_redshift(self._z_bar_G_b)
        
        self._D_z_a = self.kernel.cosmo.growth_factor(self._z_bar_G_a)
        self._D_z_b = self.kernel.cosmo.growth_factor(self._z_bar_G_b)
        chi_max_a = self.kernel.cosmo.comoving_distance(self._D_z_a)
        chi_max_b = self.kernel.cosmo.comoving_distance(self._D_z_b)
        
        self._halo_a_norm = 1.0
        self._halo_a_norm = 1.0/self._halo_a_integrand(chi_max_a,
                                                       numpy.log(1.0*chi_max_a))
        self._halo_b_norm = 1.0
        self._halo_b_norm = 1.0/self._halo_b_integrand(chi_max_b,
                                                       numpy.log(1.0*chi_max_b))
        
        _halo_a_array = numpy.empty(self._ln_K_array.shape)
        _halo_b_array = numpy.empty(self._ln_K_array.shape)
        
        for idx, ln_K in enumerate(self._ln_K_array):
            _halo_a_array[idx] = integrate.romberg(
                self._halo_a_integrand, self._chi_min_a, self._chi_max_a,
                args=(ln_K,), vec_func=True,
                tol=defaults.default_precision["global_precision"],
                rtol=defaults.default_precision["corr_precision"],
                divmax=defaults.default_precision["divmax"])
            _halo_b_array[idx] = integrate.romberg(
                self._halo_b_integrand, self._chi_min_b, self._chi_max_b,
                args=(ln_K,), vec_func=True,
                tol=defaults.default_precision["global_precision"],
                rtol=defaults.default_precision["corr_precision"],
                divmax=defaults.default_precision["divmax"])
            
        self._halo_a_spline = InterpolatedUnivariateSpline(
            self._ln_K_array, numpy.log(_halo_a_array))
        self._halo_b_spline = InterpolatedUnivariateSpline(
            self._ln_K_array, numpy.log(_halo_b_array))
        
        self._initialized_halo_splines = True
        
    def _projected_halo_a(self, K):
        if not self._initialized_halo_splines:
            self._initialize_halo_splines()
        return numpy.exp(self._halo_a_spline(numpy.log(K)))/self._halo_a_norm
    
    def _projected_halo_b(self, K):
        if not self._initialized_halo_splines:
            self._initialize_halo_splines()
        return numpy.exp(self._halo_b_spline(numpy.log(K)))/self._halo_b_norm
    
    def get_covariance(self):
        pass
        
    def covariance(self, theta_a, theta_b):
        cov_P = 0.0
        if theta_a == theta_b:
            cov_P = self.covariance_P(theta_a)
        return (cov_P +
                self.covariance_G(theta_a, theta_b) +
                self.covariance_NG(theta_a, theta_b))
    
    def covariance_P(self, theta_min, theta_max):
        return 1.0/(numpy.pi*(theta_max*theta_max - theta_min*theta_min))
        
    def covariance_G(self, theta_a, theta_b):
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
        
        self._norm_G = 1.0
        self._norm_G = 1.0/self._covariance_G_integrand(0, 0.0, 0.0)
            
        value = 1.0/(self._D_z_a*self._D_z_a*self._D_z_b*self._D_z_b*numpy.pi)*(
            integrate.romberg(
                self._covariance_G_integrand, self._ln_K_min, ln_K_max,
                args=(theta_a, theta_b), vec_func=True, 
                tol=defaults.default_precision["global_precision"],
                rtol=defaults.default_precision["corr_precision"],
                divmax=defaults.default_precision["divmax"]))
        return value/self._norm_G
    
    def _covariance_G_integrand(self, ln_K, theta_a, theta_b):
        K = numpy.exp(ln_K)
        dK = K
        return (dK*K*self._projected_halo_a(K)*self._projected_halo_b(K)*
                special.j0(K*theta_a)*special.j0(K*theta_b)*self._norm_G)
    
    def _halo_a_integrand(self, chi, ln_K):
        K = numpy.exp(ln_K)
        
        return (self.halo_a.power_mm(K/chi)*self._halo_a_norm*
                self.kernel._kernel_G_a_integrand(chi))
    
    def _halo_b_integrand(self, chi, ln_K):
        K = numpy.exp(ln_K)
        return (self.halo_a.power_mm(K/chi)*self._halo_b_norm*
                self.kernel._kernel_G_b_integrand(chi))
        
    def _chi_integrand_a(self, chi, k):
        return (self.window_function_a1.window_function(chi)*
                self.window_function_a2.window_function(chi)/(chi*chi))
        
    def covariance_NG(self, theta_a, theta_b):
        self._initialize_kb_spline(theta_a, theta_b)
        
        self._ka_norm = 1.0
        self._ka_norm = 1.0/self._ka_integrand(0.0)
        
        return 1.0/(4.0*numpy.pi*numpy.pi*self._ka_norm)*integrate.romberg(
            self._ka_integrand, self._ln_k_min, self._ln_k_max, vec_func=True,
            tol=defaults.default_precision["global_precision"],
            rtol=defaults.default_precision["corr_precision"],
            divmax=defaults.default_precision["divmax"])
        
    def _ka_integrand(self, ln_ka):
        dln_ka = 1.0
        ka = numpy.exp(ln_ka)
        dka = ka*dln_ka
        return dka*ka*self._kb_spline(ln_ka)*self._ka_norm/self._kb_norm
    
    def _initialize_kb_spline(self, theta_a, theta_b):
        if (self._current_theta_a == theta_a and
            self._current_theta_a == theta_b):
            return None
        
        _kb_int_array = numpy.empty(self._ln_k_array.shape)
        self._kb_norm = 1.0
        self._kb_norm = 1.0/self._kb_integrand(0.0, 1.0, theta_a, theta_b)
        
        for idx, ln_k in enumerate(self._ln_k_array):
            _kb_int_array[idx] = self._kb_integral(numpy.exp(ln_k),
                                                   theta_a, theta_b)
            
        self._kb_spline = InterpolatedUnivariateSpline(
            self._ln_k_array, _kb_int_array)
    
    def _kb_integral(self, ka, theta_a, theta_b):
        if type(ka) == numpy.ndarray:
            kb_int = numpy.empty(ka.shape)
            for idx, k in enumerate(ka):
                kb_int[idx] = integrate.romberg(
                    self._kb_integrand, self._ln_k_min, self._ln_k_max,
                    args=(ka, theta_a, theta_b), vec_func=True,
                    tol=defaults.default_precision["global_precision"],
                    rtol=defaults.default_precision["corr_precision"],
                    divmax=defaults.default_precision["divmax"])
            return kb_int
        return integrate.romberg(
           self._kb_integrand, self._ln_k_min, self._ln_k_max,
           args=(ka, theta_a, theta_b), vec_func=True,
           tol=defaults.default_precision["global_precision"],
           rtol=defaults.default_precision["corr_precision"],
           divmax=defaults.default_precision["divmax"])
    
    def _kb_integrand(self, ln_kb, ka, theta_a, theta_b):
        dln_kb = 1.0
        kb = numpy.exp(ln_kb)
        dkb = kb*dln_kb
        return (dkb*kb*self.halo_tri.trispectrum_parallelogram(ka, kb)*
            self._kb_norm/(self.D_z_NG*self.D_z_NG*self.D_z_NG*self.D_z_NG)*
            self.kernel.kernel(numpy.log(ka*theta_a),
                               numpy.log(kb*theta_b))[0])
        
    def correlation(self, theta):
        pass

    def _correlation_integrand(self, ln_k, theta):
        pass
    
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