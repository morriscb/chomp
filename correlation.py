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

__author__ = "Chris Morrison <morrison.chrisb@gmail.com>"


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
            tol=defaults.default_precision["corr_precision"],
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
        
class Covariance(Correlation):
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

    def __init__(self, theta_min, theta_max, input_kernel_trispectrum,
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

        self.kernel = input_kernel_trispectrum

        self.D_z = self.kernel.cosmo.growth_factor(self.kernel.z_bar)
                      
        self.halo = input_halo_trispectrum
        self.halo.set_redshift(self.kernel.z_bar)
        
        self.ln_k_min = self.halo._k_min
        self.ln_k_max = self.halo._k_max
        
        self._current_theta_a = -1.0
        self._current_theta_b = -1.0
        
    def covariance(self, theta_a, theta_b):
        self._initialize_kb_spline(theta_a, theta_b)
        return 1.0/(2.0*numpy.pi)*integrate.romberg(
           self._ka_integrand, self.ln_k_min, self.ln_k_max,
           args=(theta_a, theta_b), vec_func=True,
            tol=defaults.default_precision["corr_precision"],
            divmax=defaults.default_precision["divmax"])
        
    def _ka_integrand(self, ln_ka, theta_a, theta_b):
        dln_ka = 1.0
        ka = numpy.exp(ln_ka)
        dka = ka*dln_ka
        return dka*ka*self._kb_integral(ka, theta_a, theta_b)
    
    def _initialize_kb_spline(self, theta_a, theta_b):
        if (self._current_theta_a == theta_a and
            self._current_theta_a == theta_b):
            return None
        k_array = numpy.logspace(
            -3, 2,defaults.default_precision["corr_npoints"])
        _kb_int_array = numpy.empty(k_array.shape)
        
        for idx, k in enumerate(k_array):
            _kb_int_array[idx] = self._kb_integral(k, theta_a, theta_b)
            
        self._kb_spline = InterpolatedUnivariateSpline(
            numpy.log(k_array), _kb_int_array)
    
    def _kb_integral(self, ka, theta_a, theta_b):
        if type(ka) == numpy.ndarray:
            kb_int = numpy.empty(ka.shape)
            for idx, k in enumerate(ka):
                kb_int[idx] = 1.0/(2.0*numpy.pi)*integrate.romberg(
                    self._kb_integrand, self.ln_k_min, self.ln_k_max,
                    args=(ka, theta_a, theta_b), vec_func=True,
                    tol=defaults.default_precision["corr_precision"],
                    divmax=defaults.default_precision["divmax"])
            return kb_int
        return 1.0/(2.0*numpy.pi)*integrate.romberg(
           self._kb_integrand, self.ln_k_min, self.ln_k_max,
           args=(ka, theta_a, theta_b), vec_func=True,
           tol=defaults.default_precision["corr_precision"],
           divmax=defaults.default_precision["divmax"])
    
    def _kb_integrand(self, ln_kb, ka, theta_a, theta_b):
        dln_kb = 1.0
        kb = numpy.exp(ln_kb)
        dkb = kb*dln_kb
        return (dkb*kb*self.halo.trispectrum_projected(ka, kb)/(
            self.D_z*self.D_z*self.D_z*self.D_z)*
            self.kernel.kernel(numpy.log(ka*theta_a),
                               numpy.log(kb*theta_b))[0])
        
    def correlation(self, theta):
        pass

    def _correlation_integrand(self, ln_k, theta):
        pass
