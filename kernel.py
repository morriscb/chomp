import cosmology
import defaults
import numpy
from scipy import integrate
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import RectBivariateSpline
import copy

"""This is a set of classes for constructing an angular correlation kernel.

To calculate the Limber's approximation of an angular correlation function,
we need a kernel object that integrates over our window functions and translates
between physical and angular scale.  These window functions can be simply the
redshift distribution of our galaxies, a lensing function, an ISW potential
decay function and so on.  The idea here is that our kernel is a generic object
that takes two window function objects and performs all of the necessary
integrals.  It can also return the peak in the redshift sensitivity, so we can
make the best approximation as to the appropriate power spectrum.
"""

__author__ = ("Chris Morrison <morrison.chrisb@gmail.com",
              "Ryan Scranton <ryan.scranton@gmail.com>")

class dNdz(object):
    """Base class for a simple redshift distribution.

    This class handles all of the details of normalization and interpolation.
    Derived classes should be used for specific redshift distributions.

    Attributes:
        z_min: float minimum redshift
        z_max: float maximum redshift
    """
    def __init__(self, z_min, z_max):
        self.z_min = z_min
        self.z_max = z_max
        self.norm = 1.0

    def normalize(self):
        """
        Compute the normalized PDF for the redshift distribution for the range
        z_min - z_max.
        """
        norm = integrate.romberg(
            self.dndz, self.z_min, self.z_max, vec_func=True,
            tol=defaults.default_precision["global_precision"],
            rtol=defaults.default_precision["dNdz_precision"],
            divmax=defaults.default_precision["divmax"])

        self.norm = 1.0/norm

    def raw_dndz(self, redshift):
        """
        Raw definition of the redshift distribution.

        Args:
            redshift: float array of redshift values
        Returns:
            float array of unnormalized dn/dz.
        """
        return 1.0

    def dndz(self, redshift):
        """
        Normalized dn/dz PDF

        Args:
            redshift: float array of redshift values
        Returns:
            float array redshift PDF
        """
        return numpy.where(numpy.logical_and(redshift <= self.z_max, 
                                             redshift >= self.z_min),
                           self.norm*self.raw_dndz(redshift), 0.0)
    
    def set_limits(self, z_min=None, z_max=None, calc_norm=False):
        if z_min is not None:
            self.z_min = z_min
        if z_max is not None:
            self.z_max = z_max
        if calc_norm:
            self.normalize()


class dNdzGaussian(dNdz):
    """Derived class for a Gaussian-shaped redshift distribution.

    dNdz ~ exp(-(z-z0)^2/sigma_z^2)

    Attributes:
        z_min: float minimum redshift
        z_max: float maximum redshift
        z0: float mean redshift of gausian
        sigma_z: float standard deviation of Gaussian
    """
    def __init__(self, z_min, z_max, z0, sigma_z):
        dNdz.__init__(self, z_min, z_max)
        self.z0 = z0
        self.sigma_z = sigma_z

    def raw_dndz(self, redshift):
        return numpy.exp(-1.0*(redshift-self.z0)*(redshift-self.z0)/
                          (2.0*self.sigma_z*self.sigma_z))


class dNdChiGaussian(dNdz):
    """Derived class for a Gaussian-shaped comoving distance distribution.

    dNdz ~ exp(-(chi-chi0)^2/sigma_chi^2)*dchi/dz

    Make sure that chi_min and chi_max do not correspond to redshifts outside 
    the range z=0.0-5.0

    Attributes:
        chi_min: float minimum comoving distance
        chi_max: float maximum comoving distance
        chi0: float mean comoving distance of Gaussian
        sigma_chi: float standard deviation of Gaussian
        cosmo_multi: MultiEpoch object from cosmology.py
    """
    def __init__(self, chi_min, chi_max, chi0, sigma_chi,
                 cosmo_multi_epoch=None):
        if cosmo_multi_epoch is None:
            cosmo_multi_epoch = cosmology.MultiEpoch(0.0, 5.0)
        self.cosmo = cosmo_multi_epoch
        z_min = self.cosmo.redshift(chi_min)
        z_max = self.cosmo.redshift(chi_max)
        dNdz.__init__(self, chi_min, chi_max)
        self.chi0 = chi0 
        self.sigma_chi = sigma_chi

    def raw_dndz(self, redshift):
        chi = self.cosmo.comoving_distance(redshift)
        return (numpy.exp(-1.0*(chi-self.chi0)*(chi-self.chi0)/
                           (2.0*self.sigma_chi*self.sigma_chi)))


class dNdzMagLim(dNdz):
    """Derived class for a magnitude-limited redshift distribution.

    dNdz ~ z^a*exp(-(z/z0)^b)

    Attributes:
        z_min: float minimum redshift
        z_max: float maximum redshift
        a: float power law slope
        z0: float "mean" redshift of distribution
        b: float exponential decay slope
    """
    def __init__(self, z_min, z_max, a, z0, b):
        dNdz.__init__(self, z_min, z_max)
        self.a = a
        self.z0 = z0
        self.b = b

    def raw_dndz(self, redshift):
        return (numpy.power(redshift, self.a)*
                numpy.exp(-1.0*numpy.power(redshift/self.z0, self.b)))


class dNdzInterpolation(dNdz):
    """Derived class for a p(z) derived from real data assuming an array
    of redshifts with a corresponding array of probabilities for each
    redshift.

    Attributes:
        z_array: float array of redshifts
        p_array: float array of weights
        interpolation_order: order of spline interpolation
    """

    def __init__(self, z_array, p_array, interpolation_order=2):
        ## Need to impliment a test that throws out data at the begining or
        ## end of the z_array that has a value of zero for p_array
        dNdz.__init__(self, z_array[0], z_array[-1])
        norm = numpy.trapz(p_array, z_array)
        self._p_of_z = InterpolatedUnivariateSpline(z_array, p_array/norm,
                                                    k=interpolation_order)

    def raw_dndz(self, redshift):
        return self._p_of_z(redshift)
    

class WindowFunction(object):
    """Base class for an angular correlation window function.

    This object represents the window function for one of the two fields going
    into a correlation measurement, expressed as a function of comoving
    distance.  The details of the window function depends on the field
    involved (galaxy distribution, lensing potential, ISW potential, etc.), but
    the base class defines the API which is necessary in order to integrate the
    window function over comoving distance.

    In general, a proper calculation of the window function may be expensive,
    involving integrals of its own, so we want to evaluate the window function
    at a set number of points and then spline over them when it comes time to
    integrate over the wave function.  The former is done via the
    raw_window_function() method, which will be re-implemented by each of the
    derived classes and then the kernel object will sample the spline via the
    window_function() method.

    Attributes:
        z_min: mimimum redshift to define window function over
        z_max: maximum redshift to define window function over
        cosmo_multi_epoch: MultiEpoch cosmology object from cosmology.py
    """
    def __init__(self, z_min, z_max, cosmo_multi_epoch=None, **kws):
        self.initialized_spline = False

        if z_min < 0.0:
            z_min = 0.0
        self.z_min = z_min
        self.z_max = z_max

        if cosmo_multi_epoch is None:
            cosmo_multi_epoch = cosmology.MultiEpoch(z_min, z_max)
        self.set_cosmology_object(cosmo_multi_epoch)

        self._wf_array = numpy.zeros_like(self._chi_array, dtype='float128')
        
    def get_cosmology(self):
        """
        Return the internal dictionary defining a cosmology
        """
        return self.cosmo.get_cosmology()

    def set_cosmology(self, cosmo_dict, z_min=None, z_max=None):
        """
        Reset cosmology to values in cosmo_dict

        Args:
            cosmo_dict: dictionary of floats defining a cosmology. (see 
                defaults.py for details)
        """
        if z_min is not None:
            self.z_min = z_min
        if z_max is not None:
            self.z_max = z_max
        self.cosmo.set_cosmology(cosmo_dict, z_min, z_max)
        self.chi_min = self.cosmo.comoving_distance(self.z_min)
        if self.chi_min < defaults.default_precision["window_precision"]:
            self.chi_min = defaults.default_precision["window_precision"]
        self.chi_max = self.cosmo.comoving_distance(self.z_max)
        self._chi_array = numpy.linspace(
            self.chi_min, self.chi_max,
            defaults.default_precision["window_npoints"])

        self.initialized_spline = False

    def set_cosmology_object(self, cosmo_multi_epoch, z_min=None, z_max=None):
        """
        Reset cosmology to values in cosmo_dict

        Args:
            cosmo_multi_epoch: a MultiEpoch cosmology object from cosmology.py
        """
        #self.cosmo = cosmology.MultiEpoch(self.z_min, self.z_max, cosmo_dict 
        # if z_min is not None:
        #     self.z_min = z_min
        # if z_max is not None:
        #     self.z_max = z_max            
        self.cosmo = cosmo_multi_epoch
        self.chi_min = self.cosmo.comoving_distance(self.z_min)
        if self.chi_min < defaults.default_precision["window_precision"]:
            self.chi_min = defaults.default_precision["window_precision"]
        self.chi_max = self.cosmo.comoving_distance(self.z_max)
        self._chi_array = numpy.linspace(
            self.chi_min, self.chi_max,
            defaults.default_precision["window_npoints"])
        
        self.initialized_spline = False

    def _initialize_spline(self):
        for idx in xrange(self._chi_array.size):
            self._wf_array[idx] = self.raw_window_function(self._chi_array[idx])
        self._wf_spline = InterpolatedUnivariateSpline(self._chi_array,
                                                       self._wf_array)
        self.initialized_spline = True

    def raw_window_function(self, chi):
        """
        Raw, possibly computationally intensive, window function.

        Args:
            chi: float array comoving distance
        Returns:
            float array window function values
        """
        return 1.0

    def window_function(self, chi):
        """
        Wrapper for splined window function.

        Args:
            chi: float array of comoving distance
        Returns:
            float array of window function values
        """
        if not self.initialized_spline:
            self._initialize_spline()

        return numpy.where(numpy.logical_and(chi >= self.chi_min,
                                             chi <= self.chi_max),
                           self._wf_spline(chi), 0.0)

    def write(self, output_file_name):
        """
        Output current values of the window function

        Args:
            output_file_name: string file name
        """
        if not self.initialized_spline:
            self._initialize_spline()
        f = open(output_file_name, "w")
        f.write("#ttype1 = chi [Mpc/h]/n#ttype2 = window function value\n")
        for chi, wf in zip(self._chi_array, self._wf_array):
            f.write("%1.10f %1.10f\n" % (chi, wf))
        f.close()


class WindowFunctionGalaxy(WindowFunction):
    """WindowFunction class for a galaxy distribution.

    This derived class takes the standard WindowFunction arguments along with
    a redshift distribution and turns it into a proper WindowFunction for
    kernel integration:

    W(chi) = dN/dz dz/dchi

    for comoving distance chi.

    Attributes:
        z_min: minimum redshift to define window function over
        z_max: maximum redshift to define window function over
        cosmo_multi_epoch: MultiEpoch object from cosmology.py
    """
    def __init__(self, redshift_dist,
                 cosmo_multi_epoch=None, **kws):
        self._redshift_dist = redshift_dist
        self._redshift_dist.normalize()

        WindowFunction.__init__(self, redshift_dist.z_min, redshift_dist.z_max,
                                cosmo_multi_epoch)

    def raw_window_function(self, chi):
        z = self.cosmo.redshift(chi)

        dzdchi = 1.0/self.cosmo.E(z)

        return dzdchi*self._redshift_dist.dndz(z)


class WindowFunctionGalaxyDelta(WindowFunction):
    """
    WindowFunction class for a delta-function galaxy distribution.
    """
    def __init__(self, redshift, cosmo_multi_epoch=None, **kws):
        WindowFunction.__init__(self, redshift, redshift,
                                cosmo_multi_epoch)
        self._delta_width = 0.001
        ### Reset z range since this distribution is not integrated in Kernel.
        self.z_min = 0.0
        self.z_max = 10.0

    def raw_window_function(self, chi):
        return numpy.where(numpy.abs(chi - self.chi_min) < self._delta_width,
                           1.0, 0.0)

    def window_function(self, chi):
        return self.raw_window_function(chi)


class WindowFunctionConvergence(WindowFunction):
    """WindowFunction class for convergence of a background sample.

    This derived class calculates the convergence effect on a background
    sample as a function of comoving distance chi.  In essence, given a sample
    with redshift distribution dN/dz, what is the weighted fraction of that
    sample that is beyond chi:

    g(chi) = chi*int(chi, inf, dN/dz dz/dchi' (1.0 - chi/chi'))

    and the window function is

    W(chi) = 3/2*omega_m*g(chi)/a

    Attributes:
        z_min: mimimum redshift to define window function over
        z_max: maximum redshift to define window function over
        cosmo_multi_epoch: MultiEpoch cosmology object from cosmology.py
    """
    def __init__(self, redshift_dist, cosmo_multi_epoch=None, **kws):
        self._redshift_dist = redshift_dist
        self._redshift_dist.normalize()

        self._g_chi_min = defaults.default_precision["window_precision"]
        # Even though the input distribution may only extend between some bounds
        # in redshift, the lensing kernel will extend across z = [0, z_max)
        WindowFunction.__init__(self, 0.0, redshift_dist.z_max,
                                cosmo_multi_epoch, **kws)
        self._g_chi_min = self.cosmo.comoving_distance(
            self._redshift_dist.z_min)
        if self._g_chi_min < defaults.default_precision["window_precision"]:
            self._g_chi_min = defaults.default_precision["window_precision"]

    def raw_window_function(self, chi):
        a = 1.0/(1.0 + self.cosmo.redshift(chi))

        try:
            g_chi = numpy.empty(len(chi))
            for idx,value in enumerate(chi):
                chi_bound = value
                if chi_bound < self._g_chi_min: chi_bound = self._g_chi_min

                if chi_bound <= defaults.default_precision["window_precision"]:
                    g_chi[idx] = 0.0;
                else:
                    g_chi[idx] = integrate.romberg(
                        self._lensing_integrand, chi_bound,
                        self.chi_max, args=(value,), vec_func=True,
                        tol=defaults.default_precision["global_precision"],
                        rtol=defaults.default_precision["window_precision"],
                        divmax=defaults.default_precision["divmax"])
        except TypeError:
            chi_bound = chi
            if chi_bound < self._g_chi_min: chi_bound = self._g_chi_min

            if chi_bound <= defaults.default_precision["window_precision"]:
                g_chi = 0.0;
            else:
                g_chi = integrate.romberg(
                    self._lensing_integrand, chi_bound,
                    self.chi_max, args=(chi,), vec_func=True,
                    tol=defaults.default_precision["global_precision"],
                    rtol=defaults.default_precision["window_precision"],
                    divmax=defaults.default_precision["divmax"])

        g_chi *= self.cosmo.H0*self.cosmo.H0*chi

        return 3.0/2.0*self.cosmo._omega_m0*g_chi/a

    def _lensing_integrand(self, chi, chi0):
        z = self.cosmo.redshift(chi)

        dzdchi = 1.0/self.cosmo.E(z)

        return dzdchi*self._redshift_dist.dndz(z)*(chi - chi0)/chi

class WindowFunctionFlatConvergence(WindowFunction):
    """WindowFunction class for magnification of a background sample.

    This derived class calculates the magnification effect of a background
    sample as a function of comoving distance chi.  In essence, given a sample
    with redshift distribution dN/dz, what is the weighted fraction of that
    sample that is beyond chi:

    g(chi) = chi*int(chi, inf, dN/dz dz/dchi' (1.0 - chi/chi'))

    and the window function is

    W(chi) = 3/2*omega_m*g(chi)/a

    
    """
    def __init__(self, z_min, z_max, cosmo_multi_epoch=None, **kws):
        # Even though the input distribution may only extend between some bounds
        # in redshift, the lensing kernel will extend across z = [0, z_max)
        WindowFunction.__init__(self, z_min, z_max,
                                cosmo_multi_epoch, **kws)

    def raw_window_function(self, chi):
        a = 1.0/(1.0 + self.cosmo.redshift(chi))

        g_chi = 1.0

        g_chi *= self.cosmo.H0*self.cosmo.H0*1907.71

        return 3.0/2.0*self.cosmo._omega_m0*g_chi

class WindowFunctionConvergenceDelta(WindowFunction):
    """WindowFunction class for convergence of a background sample.

    This derived class calculates the convergence effect of a background
    sample as a function of comoving distance chi.  In essence, given a sample
    with redshift distribution dN/dz, what is the weighted fraction of that
    sample that is beyond chi:

    g(chi) = chi*int(chi, inf, dN/dz dz/dchi' (1.0 - chi/chi'))

    and the window function is

    W(chi) = 3/2*omega_m*g(chi)/a
    """
    def __init__(self, redshift, cosmo_multi_epoch=None, **kws):
        self._redshift = redshift
        #self._redshift_dist.normalize()

        self._g_chi_min = 0.0
        # Even though the input distribution may only extend between some bounds
        # in redshift, the lensing kernel will extend across z = [0, z_max)
        WindowFunction.__init__(self, 0.0, redshift,
                                cosmo_multi_epoch, **kws)

    def raw_window_function(self, chi):
        a = 1.0/(1.0 + self._redshift)

        chi_bound = numpy.min(chi)
        if chi_bound < self._g_chi_min: chi_bound = self._g_chi_min

        g_chi = self._lensing_integrand(chi)

        g_chi *= self.cosmo.H0*self.cosmo.H0*chi

        return 3.0/2.0*self.cosmo._omega_m0*g_chi/a

    def _lensing_integrand(self, chi0):
        if chi0 > self.chi_max:
            return 0.0

        return (self.chi_max - chi0)/self.chi_max

class Kernel(object):
    """Container class for calculating correlation function kernels.

    A kernel is an integtral over the product of two window functions
    representing the spatial extent of two fields (or one field for the case of
    an autocorrelation).  In addition, there is a Bessel function which
    incorporates the projected angular dependence of the correlation function.
    This means that the kernel is a function of k*theta, where k is in h/Mpc
    and theta is in radians:

    K(k, theta) = 4pi^2*int(0, inf, D^2(chi)*W_a(chi)*W_b(chi)*J_0(k*theta*chi))

    In addition to providing the kernel function, a kernel object also
    calculates z_bar, the peak in the kernel redshift sensitivity.

    Args:
        ktheta_min: float k*theta minimum value for the kernel
        ktheta_min: float k*theta maximum value for the kernel
        window_function_a: first window function for kernel
        window_function_b: second window function for kernel
        cosmo_multi_epoch: MultiEpoch cosmology object from cosmology.py
        force_quad: If the romberg integration is giving too much numerical
            noise at large ktheta set this flag to True to use quad integration 
            for more accuracy at the cost of speed.
    """
    def __init__(self, ktheta_min, ktheta_max,
                 window_function_a, window_function_b,
                 cosmo_multi_epoch=None, force_quad=False, **kws):
        self.initialized_spline = False

        self.ln_ktheta_min = numpy.log(ktheta_min)
        self.ln_ktheta_max = numpy.log(ktheta_max)

        self.window_function_a = copy.copy(window_function_a)
        self.window_function_b = copy.copy(window_function_b)

        self.z_min = numpy.max([self.window_function_a.z_min,
                                self.window_function_b.z_min])
        self.z_max = numpy.min([self.window_function_a.z_max,
                                self.window_function_b.z_max])

        if cosmo_multi_epoch is None:
            cosmo_multi_epoch = cosmology.MultiEpoch(
                self.z_min, self.z_max)
        self.cosmo = cosmo_multi_epoch

        self.window_function_a.set_cosmology_object(self.cosmo, 
                                                    self.z_min, self.z_max)
        self.window_function_b.set_cosmology_object(self.cosmo, 
                                                    self.z_min, self.z_max)

        self.chi_min = numpy.max([defaults.default_precision["window_precision"],
                                  self.cosmo.comoving_distance(self.z_min)])
        self.chi_max = self.cosmo.comoving_distance(self.z_max)

        self._window_norm = integrate.romberg(
            lambda chi: (self.window_function_a.window_function(chi)*
                         self.window_function_b.window_function(chi)),
            self.chi_min, self.chi_max, vec_func=True,
            tol=defaults.default_precision["global_precision"],
            rtol=defaults.default_precision["kernel_precision"],
            divmax=defaults.default_precision["divmax"])
        
        self._ln_ktheta_array = numpy.linspace(
            self.ln_ktheta_min, self.ln_ktheta_max,
            defaults.default_precision["kernel_npoints"])
        self._kernel_array = numpy.zeros_like(self._ln_ktheta_array,
                                              dtype='float128')

        self._j0_limit = special.jn_zeros(
            0, defaults.default_precision["kernel_bessel_limit"])[-1]

        self._force_quad = force_quad

        self._find_z_bar()

    def _find_z_bar(self):
        z_array = numpy.linspace(self.z_min, self.z_max,
                                 defaults.default_precision["kernel_npoints"])
        self.z_bar = z_array[numpy.argmax(
            self._kernel_integrand(self.cosmo.comoving_distance(z_array), 0.0))]

    def _initialize_spline(self):
        for idx in xrange(self._ln_ktheta_array.size):
            kernel = self.raw_kernel(self._ln_ktheta_array[idx])
            self._kernel_array[idx] = kernel

        self._kernel_spline = InterpolatedUnivariateSpline(
            self._ln_ktheta_array, self._kernel_array)

        self.initialized_spline = True
        
    def get_cosmology(self):
        """
        Return the internal dictionary defining a cosmology
        """
        return self.cosmo.get_cosmology()

    def set_cosmology(self, cosmo_dict):
        """
        Reset the cosmology

        Args:
            cosmo_dict: dictionary of floats defining a cosmology (see
                defaults.py for details)
        """
        self.initialized_spline = False

        self.cosmo.set_cosmology(cosmo_dict)
        self.window_function_a.set_cosmology_object(self.cosmo)
        self.window_function_b.set_cosmology_object(self.cosmo)
        
        self.chi_min = numpy.max([1e-8,
                                  self.cosmo.comoving_distance(self.z_min)])
        self.chi_max = self.cosmo.comoving_distance(self.z_max)

        self._find_z_bar()  

    def raw_kernel(self, ln_ktheta):
        """
        Raw kernel function. Projected power as a function of chi.

        Args:
            ln_ktheta: float array natural logathim of k*theta
        Returns:
            float array kernel value
        """
        ktheta = numpy.exp(ln_ktheta)

        chi_max = self._j0_limit/ktheta
        if chi_max >= self.chi_max:
            chi_max = self.chi_max
        if self._force_quad:
            kernel = integrate.quad(
                self._kernel_integrand, self.chi_min,
                chi_max, args=(ktheta,),
                limit=defaults.default_precision["kernel_limit"])[0]
            return kernel
        else:
            kernel = integrate.romberg(
                self._kernel_integrand, self.chi_min,
                self.chi_max, args=(ktheta,), vec_func=True,
                tol=defaults.default_precision["global_precision"],
                rtol=defaults.default_precision["kernel_precision"],
                divmax=defaults.default_precision["divmax"])
            return kernel

    def _kernel_integrand(self, chi, ktheta):
        D_z = self.cosmo.growth_factor(self.cosmo.redshift(chi))
        z = self.cosmo.redshift(chi)
        
        return (self.window_function_a.window_function(chi)*
                self.window_function_b.window_function(chi)*
                D_z*D_z*special.j0(ktheta*chi))

    def kernel(self, ln_ktheta):
        """
        Wrapper function for the splined kernel function.

        Args:
            ln_ktheta: float array natural logathim of k*theta
        Returns:
            float array kernel value
        """
        if not self.initialized_spline:
            self._initialize_spline()

        return numpy.where(ln_ktheta < self.ln_ktheta_min,
                           self._kernel_spline(self.ln_ktheta_min),
                           numpy.where(ln_ktheta <= self.ln_ktheta_max,
                                       self._kernel_spline(ln_ktheta), 0.0))
                                
    def kernel_weighted_mean(self, function):
        """
        Given an input function of redshift, compute the mean value of the 
        function weighted by the kernel. The function must be defined between
        kernel.z_min and kernel.z_max

        Args:
            function_obj: input redshift dependent function
        Returns:
            float weighted mean value of function
        """
        chi_fun = lambda chi: function(self.cosmo.redshift(chi))

        mean = integrate.romberg(
            lambda chi: (chi_fun(chi)*
                         self.window_function_a.window_function(chi)*
                         self.window_function_b.window_function(chi)),
            self.chi_min, self.chi_max, vec_func=True,
            tol=defaults.default_precision["global_precision"],
            rtol=defaults.default_precision["kernel_precision"],
            divmax=defaults.default_precision["divmax"])

        return mean/self._window_norm


    def write(self, output_file_name):
        """
        Output current values of the kernel

        Args:
            output_file_name: string file name
        """
        if not self.initialized_spline:
            self._initialize_spline()

        f = open(output_file_name, "w")
        f.write("#ttype1 = k*theta [h/Mpc*Radians]\n"
                "#ttype2 = kernel [(h/Mpc)^2]\n")
        for ln_ktheta, kernel in zip(
            self._ln_ktheta_array, self._kernel_array):
            f.write("%1.10g %1.10g\n" % (numpy.exp(ln_ktheta), kernel))
        f.close()
        
        
class GalaxyGalaxyLensingKernel(Kernel):
    """Derived class for Galaxy-Galaxy lensing. The galaxy-galaxy lensing kernel
    differes slightly from the standard kernel in that the Bessel function is
    J_2 instead of J_0. Hence Delta_Sigma instead of Sigma for the measured
    mass profile.

    K(k, theta) = 4pi^2*int(0, inf, D^2(chi)*W_a(chi)*W_b(chi)*J_2(k*theta*chi))

    Args:
        ktheta_min: float k*theta minimum value for the kernel
        ktheta_min: float k*theta maximum value for the kernel
        window_function_a: first window function for kernel
        window_function_b: second window function for kernel
        cosmo_multi_epoch: MultiEpoch object from cosmology.py
        force_quad: If the romberg integration is giving too much numerical
            noise at large ktheta set this flag to True to use quad integration 
            for more accuracy at the cost of speed.
    """

    def __init__(self, ktheta_min, ktheta_max,
                 window_function_a, window_function_b,
                 cosmo_multi_epoch=None, force_quad=False, **kws):
        self._j2_limit = special.jn_zeros(
            2, defaults.default_precision["kernel_bessel_limit"])[-1]
        Kernel.__init__(self, ktheta_min, ktheta_max,
                        window_function_a, window_function_b,
                        cosmo_multi_epoch, force_quad, **kws)

    def raw_kernel(self, ln_ktheta):
        ktheta = numpy.exp(ln_ktheta)

        chi_max = self._j2_limit/ktheta
        if chi_max >= self.chi_max:
            chi_max = self.chi_max
        if self._force_quad:
            kernel = integrate.quad(
                self._kernel_integrand_j2, self.chi_min,
                chi_max, args=(ktheta,),
                limit=defaults.default_precision["kernel_limit"])[0]
            return kernel
        else:
            kernel = integrate.romberg(
                self._kernel_integrand_j2, self.chi_min,
                chi_max, args=(ktheta,), vec_func=True,
                tol=defaults.default_precision["global_precision"],
                rtol=defaults.default_precision["kernel_precision"],
                divmax=defaults.default_precision["divmax"])
            return kernel

    def _kernel_integrand_j2(self, chi, ktheta):
        D_z = self.cosmo.growth_factor(self.cosmo.redshift(chi))
        z = self.cosmo.redshift(chi)

        return (self.window_function_a.window_function(chi)*
                self.window_function_b.window_function(chi)*
                D_z*D_z*special.jn(2, ktheta*chi))     


class KernelGalaxyDelta(Kernel):
    def __init__(self, ktheta_min, ktheta_max,
                 window_function_a, window_function_b,
                 cosmo_multi_epoch=None, force_quad=False, **kws):
        if window_function_a.__class__.__name__ != "WindowFunctionGalaxyDelta":
            print "WARNING: KernelGalaxyDelta expects window_function_a" \
                  "of class WindowFunctionGalaxyDelta"
        Kernel.__init__(self, ktheta_min, ktheta_max,
                        window_function_a, window_function_b,
                        cosmo_multi_epoch, force_quad, **kws)
        ### Set the distance obtained by integrating over the delta-function
        ### foreground distribution
        self.chi_g = window_function_a.chi_min

    def raw_kernel(self, ln_ktheta):
        D_z = self.cosmo.growth_factor(self.cosmo.redshift(self.chi_g))
        ktheta = numpy.exp(ln_ktheta)

        return (self.window_function_b.window_function(self.chi_g) *
                D_z*D_z*special.j0(ktheta*self.chi_g))


class KernelCovariance(Kernel):
    """
    Inherited class from Kernel defining the redshift integral over the various
    redshift dependent window functions. We also include in this integral the
    Bessel functions. The of the kernel gives the fouier transform of the
    coveriance matrix between two probes at two k scales. The probes
    (clustering, cosmic shear, magnification, etc.) are refered a and b and
    their kernels are a1 a2.
    
    K(k_a, k_b, theta_a, theta_b) = 4pi^2*int(0, inf, D^4(chi)*
        W_a1(chi)*W_a2(chi)*W_b1(chi)*W_b2*
        J_0(k_a*theta_a*chi)*J_0(k_b*theta_b*chi))
                            
    In addition to providing the kernel function, a kernel object also
    calculates z_bar, the peak in the kernel redshift sensitivity.

    Args:
        ktheta_min: float k*theta minimum value for the kernel
        ktheta_min: float k*theta maximum value for the kernel
        window_function_a1: first window function for kernel a
        window_function_a2: second window function for kernel a
        window_function_b1: first window function for kernel b
        window_function_b2: second window function for kernel b
        cosmo_multi_epoch: MultiEpoch cosmology object from cosmology.py
        force_quad: If the romberg integration is giving too much numerical
            noise at large ktheta set this flag to True to use quad integration 
            for more accuracy at the cost of speed.  
    """

    def __init__(self, ktheta_min, ktheta_max,
                 window_function_a1, window_function_a2,
                 window_function_b1, window_function_b2,
                 cosmo_multi_epoch, force_quad=False):
        self.initialized_G_spline = False
        self.initialized_NG_spline = False

        self.ln_ktheta_min = numpy.log(ktheta_min)
        self.ln_ktheta_max = numpy.log(ktheta_max)

        self.window_function_a1 = window_function_a1
        self.window_function_a2 = window_function_a2
        self.window_function_b1 = window_function_b1
        self.window_function_b2 = window_function_b2

        self.z_min = numpy.max(
            [self.window_function_a1.z_min, self.window_function_a2.z_min,
             self.window_function_b1.z_min, self.window_function_b2.z_min])

        self.z_max = numpy.min(
            [self.window_function_a1.z_max, self.window_function_a2.z_max,
             self.window_function_b1.z_max, self.window_function_b2.z_max])

        if cosmo_multi_epoch is None:
            cosmo_multi_epoch = cosmology.MultiEpoch(
                self.z_min, self.z_max)
        self.cosmo = cosmo_multi_epoch

        self.window_function_a1.set_cosmology_object(self.cosmo)
        self.window_function_a2.set_cosmology_object(self.cosmo)
        self.window_function_b1.set_cosmology_object(self.cosmo)
        self.window_function_b2.set_cosmology_object(self.cosmo)

        self.chi_min = numpy.max([defaults.default_precision["window_precision"],
                                  self.cosmo.comoving_distance(self.z_min)])
        self.chi_max = self.cosmo.comoving_distance(self.z_max)
        
        self._ln_ktheta_array = numpy.linspace(
            self.ln_ktheta_min, self.ln_ktheta_max,
            defaults.default_precision["kernel_npoints"])
        self._kernel_array = numpy.empty(
            (defaults.default_precision["kernel_npoints"],
             defaults.default_precision["kernel_npoints"]), 'float128')

        self._j0_limit = special.jn_zeros(
            0, defaults.default_precision["kernel_bessel_limit"])[-1]

        self._force_quad = force_quad

        ### Forward declaration of our splines so we can call get/settatribute
        self._kernel_G_spline = None
        self._kernel_NG_spline = None
        
        self._initialized_NG_spline = False
        
        self._find_z_bar()
        
    def _find_z_bar(self):
        self._int_G_norm = 1.0
        self._int_NG_norm = 1.0
        z_array = numpy.linspace(self.z_min, self.z_max,
                                 defaults.default_precision["kernel_npoints"])
        chi_array = self.cosmo.comoving_distance(z_array)
        self.z_bar_NG = z_array[numpy.argmax(self._kernel_NG_integrand(
            numpy.where(chi_array > 
                        defaults.default_precision["window_precision"], 
                        chi_array,
                        defaults.default_precision["window_precision"]),
                        0.0, 0.0))]

    def set_cosmology(self, cosmo_dict):
        """
        Reset the cosmology

        Args:
            cosmo_dict: dictionary of floats defining a cosmology (see
                defaults.py for details)
        """
        self.initialized_G_spline = False
        self.initialized_NG_spline = False

        self.cosmo.set_cosmology(cosmo_dict)
        self.window_function_a1.set_cosmology_object(self.cosmo)
        self.window_function_a2.set_cosmology_object(self.cosmo)
        self.window_function_b1.set_cosmology_object(self.cosmo)
        self.window_function_b2.set_cosmology_object(self.cosmo)
        
        self.chi_min = self.cosmo.comoving_distance(self.z_min)
        self.chi_max = self.cosmo.comoving_distance(self.z_max)

        self._find_z_bar()
        
    def kernel(self, ln_ktheta_a, ln_ktheta_b):
        return self.kernel_NG(ln_ktheta_a, ln_ktheta_b)
        
    def kernel_NG(self, ln_ktheta_a, ln_ktheta_b):
        if not self._initialized_NG_spline:
            self._initialize_NG_spline()
            
        ln_ktheta_a = numpy.where(ln_ktheta_a < self.ln_ktheta_min,
                                  self.ln_ktheta_min,
                                  ln_ktheta_a)
        ln_ktheta_b = numpy.where(ln_ktheta_b < self.ln_ktheta_min,
                                  self.ln_ktheta_min,
                                  ln_ktheta_b)
        
        return numpy.where(
            numpy.logical_and(ln_ktheta_a <= self.ln_ktheta_max,
                              ln_ktheta_b <= self.ln_ktheta_max),
            numpy.exp(self._kernel_NG_spline(ln_ktheta_a, ln_ktheta_b)) + 
            self._kernel_NG_min*10.0, 0.0)

    def _initialize_NG_spline(self):
        for idx1 in xrange(len(self._ln_ktheta_array)):
            for idx2 in xrange(idx1, len(self._ln_ktheta_array)):
                kern = self.raw_kernel(self._ln_ktheta_array[idx1],
                                       self._ln_ktheta_array[idx2])
                if idx1 == idx2:
                    self._kernel_array[idx1, idx2] = kern
                else:
                    self._kernel_array[idx1, idx2] = kern
                    self._kernel_array[idx2, idx1] = kern
        self._kernel_NG_min = numpy.min(self._kernel_array)
        self._kernel_NG_spline = RectBivariateSpline(
            self._ln_ktheta_array, self._ln_ktheta_array,
            numpy.log(self._kernel_array - self._kernel_NG_min*10.0))
        self._initialized_NG_spline = True
        
    def raw_kernel(self, ln_ktheta_a, ln_ktheta_b):
        return self.raw_kernel_NG(ln_ktheta_a, ln_ktheta_b)

    def raw_kernel_NG(self, ln_ktheta_a, ln_ktheta_b):
        """
        Raw kernel function. Projected power as a function of chi.

        Args:
            ln_ktheta: float array natural logathim of k*theta
        Returns:
            float array kernel value
        """
        ktheta_a = numpy.exp(ln_ktheta_a)
        ktheta_b = numpy.exp(ln_ktheta_b)

        chi_max = numpy.max([self._j0_limit/ktheta_a,
                             self._j0_limit/ktheta_b])
        if chi_max >= self.chi_max:
            chi_max = self.chi_max
        elif chi_max <= self.chi_min:
            return 0.0
        
        peak_NG_chi = self.cosmo.comoving_distance(self.z_bar_NG)
        inv_norm = self._kernel_NG_integrand(
            peak_NG_chi, ln_ktheta_a, ln_ktheta_a, 1.0)
        norm = 1.0
        if inv_norm > 1e-16 or inv_norm < -1e-16:
            norm = 1.0/inv_norm
        if self._force_quad:
            kernel = integrate.quad(
                self._kernel_NG_integrand, self.chi_min, chi_max,
                args=(ktheta_a, ktheta_b, norm),
                limit=defaults.default_precision["kernel_limit"])[0]
            return kernel/norm
        else:
            kernel = integrate.romberg(
                self._kernel_NG_integrand, self.chi_min, chi_max,
                args=(ktheta_a, ktheta_b, norm), vec_func=True,
                tol=defaults.default_precision["global_precision"],
                rtol=defaults.default_precision["kernel_precision"],
                divmax=defaults.default_precision["divmax"])
            return kernel/norm

    def _kernel_G_a_integrand(self, chi):
        D_z = self.cosmo.growth_factor(self.cosmo.redshift(chi))
        
        return (self.window_function_a1.window_function(chi)*
                self.window_function_a2.window_function(chi)*
                D_z*D_z/(chi*chi))
        
    def _kernel_G_b_integrand(self, chi):
        D_z = self.cosmo.growth_factor(self.cosmo.redshift(chi))
        
        return (self.window_function_b1.window_function(chi)*
                self.window_function_b2.window_function(chi)*
                D_z*D_z/(chi*chi))

    def _kernel_G_ab_integrand(self, chi):
        D_z = self.cosmo.growth_factor(self.cosmo.redshift(chi))
        
        return (self.window_function_a1.window_function(chi)*
                self.window_function_b2.window_function(chi)*
                D_z*D_z/(chi*chi))    

    def _kernel_G_ba_integrand(self, chi):
        D_z = self.cosmo.growth_factor(self.cosmo.redshift(chi))
        
        return (self.window_function_a2.window_function(chi)*
                self.window_function_b1.window_function(chi)*
                D_z*D_z/(chi*chi))    
        
    def _kernel_NG_integrand(self, chi, ktheta_a, ktheta_b, norm=1.0):
        D_z = self.cosmo.growth_factor(self.cosmo.redshift(chi))
        
        return (norm*self.window_function_a1.window_function(chi)*
                self.window_function_a2.window_function(chi)*
                self.window_function_b1.window_function(chi)*
                self.window_function_b2.window_function(chi)*
                D_z*D_z*D_z*D_z/(chi*chi)*
                special.j0(ktheta_a*chi)*special.j0(ktheta_b*chi))

    def kernel_weighted_mean(self, function):
        pass

    def write(self, output_file_name):
        """
        Output current values of the kernel

        Args:
            output_file_name: string file name
        """
        if not self.initialized_spline:
            self._initialize_spline()

        f = open(output_file_name, "w")
        f.write("#type1 = k_a*theta_a [h/Mpc*Radians]\n"
                "#type2 = k_b*theta_b"
                "#ttype2 = kernel\n")
        for ln_ktheta_a in self._ln_ktheta_array:
            for ln_ktheta_b in self._ln_ktheta_array:
                f.write("%1.10f %1.10f %1.10f\n" % 
                        (numpy.exp(ln_ktheta_a), numpy.exp(ln_ktheta_b),
                         self.kernel(ln_ktheta_a, ln_ktheta_b)))
        f.close()
        
