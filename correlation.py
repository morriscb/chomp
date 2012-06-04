import camb
import cosmology
import defaults
import halo
import hod
import kernel
import numpy
from scipy import special
from scipy import integrate

"""
Classes describing different correlation functions.

Each correlation function can be defined as

w(theta) = int(z, g_1(z)*g_2(z)*int(k, k/(2*pi) P_x(k, z) J(k*theta*chi(z))))

where P_x is a power spectrum from halo, J is the bessel function, and g_1/g_2
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
        window_function_a: a window function object from kernel.py
        window_function_b: a window function object from kernel.py
        cosmo_dict: dictionary of floats defining a cosmology (see defaults.py
            for details)
        input_hod: an HOD object from hod.py
        halo_dict: dictionary of floats defining halos (see defaults.py
            for details)
        powSpec: string defining a power spectrum
        theta_array: array of theta values for computed correlation function
        wtheta_array: array of computed correlation values at theta_array values
    """

    def __init__(self, theta_min, theta_max, input_kernel=None,
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
            print "WARNING: Invalid input for power spectra varriable,"
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
        member of the halo object return the linear powerspectrum.

        Args:
            powSpec: string name of power spectrum to use from halo.py object.
        """
        try:
            self.power_spec = self.halo.__getattribute__(powSpec)
        except AttributeError or TypeError:
            print "WARNING: Invalid input for power spectra varriable,"
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
            tol=defaults.default_precision["corr_precision"])
        return wtheta

    def _correlation_integrand(self, ln_k, theta):
        dln_k = 1.0
        k = numpy.exp(ln_k)
        dk = k*dln_k
        return 1.0/(numpy.pi)*(
            dk*k*self.power_spec(k)/(self.D_z*self.D_z)*
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
