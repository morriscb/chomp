import cosmology
import defaults
import numpy
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline

"""Classes for encoding basic cosmological parameters and quantities.

Given a cosmology, we can derive a mass function object.  This should be able
to return the abundance of halo at a given mass as well as the mean halo bias.
Likewise, these quantities should be available as a function of nu:

nu = (delta_c/sigma(M))^2

This is the ratio of the over-density necessary for linear, spherical
collapse and the RMS density fluctuations as a function of mass.  M*, the mass
at which nu == 1, represents the break between halos that are mostly collapsed
and those where fluctuations large enough for collapse are exponentially
suppressed.
"""

__author__ = "Ryan Scranton <ryan.scranton@gmail.com>"

class MassFunction(object):
    """Object representing a mass function for a given input cosmology.

    A MassFunction object can return a properly normalized halo abundance or
    halo bias as a function of halo mass or as a function of nu, as well as
    translate between mass and nu. Current definition is from Sheth & Torman

    Attributes:
        redshift: float redshift at which to compute the mass function
        cosmo_single_epoch: SingleEpoch cosmology object from cosmology.py
        halo_dict: dictionary of floats defining halo and mass function 
            parameters (see defualts.py for details)
    """
    #def __init__(self, redshift=0.0, cosmo_dict=None, halo_dict=None, **kws):
    def __init__(self, redshift=0.0, cosmo_single_epoch=None, 
                 halo_dict=None, **kws):
        self._redshift = redshift
        #self.cosmo = cosmology.SingleEpoch(self._redshift, cosmo_dict)
        if cosmo_single_epoch is None:
            cosmo_single_epoch = cosmology.SingleEpoch(self._redshift)
        self.cosmo = cosmo_single_epoch
        self.cosmo.set_redshift(self._redshift)
        self.delta_c = self.cosmo.delta_c()
        self.delta_v = self.cosmo.delta_v()

        if halo_dict is None:
            halo_dict = defaults.default_halo_dict
        self.halo_dict = halo_dict

        self.stq = halo_dict["stq"]
        self.st_little_a = halo_dict["st_little_a"]
        self.c0 = halo_dict["c0"]/(1.0 + redshift)

        self._set_mass_limits()
        self._initialize_splines()
        self._normalize()

    def set_redshift(self, redshift):
        """
        Reset mass function parameters at redshift.

        Args:
            redshift: float value of redshift
            cosmo_dict: dictionary of floats defining a cosmology (see
                defaults.py for details)
        """
        self._redshift = redshift

        self.cosmo.set_redshift(redshift)

        self.delta_c = self.cosmo.delta_c()
        self.c0 = halo_dict["c0"]/(1.0 + redshift)

        self._set_mass_limits()
        self._initialize_splines()
        self._normalize()

    def set_cosmology(self, cosmo_dict, redshift = None):
        """
        Reset mass function parameters for cosmology cosmo_dict.

        Args:
            cosmo_dict: dictionary of floats defining a cosmology (see
                defaults.py for details)
            redshift: float value of redshift
        """
        if redshift is None:
            redshift = self._redshift
        self.cosmo.set_cosmology(redshift, cosmo_dict)

        self.delta_c = self.cosmo.delta_c()
        self.delta_v = self.cosmo.delta_v()
        self.c0 = self.halo_dict["c0"]/(1.0 + redshift)

        self._set_mass_limits()
        self._initialize_splines()
        self._normalize()

    def set_cosmology_object(self, cosmo_single_epoch):
        self._redshift = cosmo_single_epoch.redshift()
        self.cosmo = cosmo_single_epoch

        self.delta_c = self.cosmo.delta_c()
        self.c0 = self.halo_dict["c0"]/(1.0 + self._redshift)

        self._set_mass_limits()
        self._initialize_splines()
        self._normalize()

    def set_halo(self, halo_dict):
        """
        Reset mass function parameters for halo_dict.

        Args:
            halo_dict: dictionary of floats defining halos (see
                defaults.py for details)
        """
        self.halo_dict = halo_dict

        self.stq = self.halo_dict["stq"]
        self.st_little_a = self.halo_dict["st_little_a"]
        self.c0 = self.halo_dict["c0"]/(1.0 + self._redshift)

        self._normalize()

    def _set_mass_limits(self):
        mass_min = 1.0e9
        mass_max = 1.0e16

        mass_limit_not_set = True
        while mass_limit_not_set:
            if 0.1*(1.0+0.05) < self.cosmo.nu_m(mass_min):
                #print "Min mass", mass_min,"too high..."
                mass_min = mass_min/1.05
                #print "\tSetting to",mass_min,"..."
                continue
            elif 0.1*(1.0-0.05) > self.cosmo.nu_m(mass_min):
                #print "Min mass", mass_min,"too low..."
                mass_min = mass_min*1.05
                #print "\tSetting to",mass_min,"..."
                continue
            if  50.0*(1.0-0.05) > self.cosmo.nu_m(mass_max):
                #print "Max mass", mass_max,"too low..."
                mass_max = mass_max*1.05
                #print "\tSetting to",mass_max,"..."
                continue
            elif 50.0*(1.0+0.05) < self.cosmo.nu_m(mass_max):
                #print "Max mass", mass_max,"too high..."
                mass_max = mass_max/1.05
                #print "\tSetting to",mass_max,"..."
                continue
            mass_limit_not_set = False

        #print "Mass Limits:",mass_min*(0.95),"-",mass_max*(1.05)

        self.ln_mass_max = numpy.log(mass_max*(0.95))
        self.ln_mass_min = numpy.log(mass_min*(1.05))

        self._ln_mass_array = numpy.linspace(
            self.ln_mass_min, self.ln_mass_max,
            defaults.default_precision["mass_npoints"])

    def _initialize_splines(self):
        self._nu_array = numpy.zeros_like(self._ln_mass_array)

        for idx in xrange(self._ln_mass_array.size):
            mass = numpy.exp(self._ln_mass_array[idx])
            self._nu_array[idx] = self.cosmo.nu_m(mass)

        self.nu_min = 1.001*self._nu_array[0]
        self.nu_max = 0.999*self._nu_array[-1]

        #print "nu_min:",self.nu_min,"nu_max:",self.nu_max

        self._nu_spline = InterpolatedUnivariateSpline(
            self._ln_mass_array, self._nu_array)
        self._ln_mass_spline = InterpolatedUnivariateSpline(
            self._nu_array, self._ln_mass_array)

        # Set M_star, the mass for which nu == 1
        self.m_star = self.mass(1.0)

    def _normalize(self):
        self.f_norm = 1.0
        norm = integrate.romberg(
            self.f_nu, self.nu_min, self.nu_max, vec_func=True,
            tol=defaults.default_precision["mass_precision"])
        self.f_norm = 1.0/norm

        self.bias_norm = 1.0
        norm = integrate.romberg(
            lambda x: self.f_nu(x)*self.bias_nu(x),
            self.nu_min, self.nu_max, vec_func=True,
            tol=defaults.default_precision["mass_precision"])
        self.bias_norm = 1.0/norm

    def f_nu(self, nu):
        """
        Halo mass function as a function of normalized mass overdensity nu

        Args:
            nu: float array normalized mass overdensity nu
        Returns:
            float array number of halos
        """
        nu_prime = nu*self.st_little_a
        return (
            self.f_norm*(1.0 + nu_prime**(-1.0*self.stq))*
            numpy.sqrt(nu_prime)*numpy.exp(-0.5*nu_prime)/nu)

    def f_m(self, mass):
        """
        Halo mass function as a function of halo mass

        Args:
            mass: float array halo mass
        Returns:
            float array number of halos
        """
        return self.f_nu(self.nu(mass))

    def bias_nu(self, nu):
        """
        Halo bias as a function of nu.

        Args:
            mass: float array mass overdensity mu
        Returns:
            float array halo bias
        """
        nu_prime = nu*self.st_little_a
        return self.bias_norm*(
            1.0 + (nu - 1.0)/self.delta_c +
            2.0*self.stq/(self.delta_c*(1.0 + nu_prime**self.stq)))

    def bias_m(self, mass):
        """
        Halo bias as a function of mass.

        Args:
            mass: float array halo mass
        Returns:
            float array halo bias
        """
        return self.bias_nu(self.nu(mass))

    def nu(self, mass):
        """
        nu as a function of halo mass.

        Args:
            nu: float array mass M [M_Solar]
        Returns:
            float array 
        """
        return self._nu_spline(numpy.log(mass))

    def ln_mass(self, nu):
        """
        Natural log of halo mass as a function of nu.

        Args:
            nu: float array normalized mass overdensity
        Returns:
            float array natrual log mass [M_Solar]
        """
        return self._ln_mass_spline(nu)

    def mass(self, nu):
        """
        Halo mass as a function of nu.

        Args:
            nu: float array normalized mass overdensity
        Returns:
            float array halo mass [M_Solar]
        """
        return numpy.exp(self.ln_mass(nu))

    def write(self, output_file_name):
        """
        Write current mass function values

        Args:
            output_file_name: string file name to write mass function parameters
        """
        print "M* = 10^%1.4f M_sun" % numpy.log10(self.m_star)
        output_file = open(output_file_name, "w")
        output_file.write("#ttype1 = mass [M_solar/h]\n#ttype2 = nu\n"
                          "#ttype3 = f(nu)\n#ttype4 = bias(nu)\n")
        for ln_mass, nu, in zip(self._ln_mass_array, self._nu_array):
            output_file.write("%1.10f %1.10f %1.10f %1.10f\n" % (
                numpy.exp(ln_mass), nu, self.f_nu(nu), self.bias_nu(nu)))
        output_file.close()

class TinkerMassFunction(MassFunction):
    """
    Derived mass function expressing the functional form from Tinker et al. 
    2008. (Currently gives junk results)

    Attributes:
        redshift: float redshift at which to compute the mass function
        cosmo_single_epoch: SingleEpoch cosmology object from cosmology.py
        halo_dict: dictionary of floats defining halo and mass function 
            parameters (see defualts.py for details)
    """

    def __init__(self, redshift=0.0, cosmo_single_epoch=None, 
                 halo_dict=None, **kws):
        delta_list = [200, 300, 400, 600, 800, 1200, 1600, 2400, 3200]
        beta_list = [0.589, 0.585, 0.544, 0.543, 
                     0.564, 0.632, 0.637, 0.673, 0.702]
        gamma_list = [0.864, 0.922, 0.987, 1.09, 1.20, 1.34, 1.50, 1.68, 1.81]
        phi_list = [-0.729, -0.789, -0.910, -1.05,
                     -1.20, -1.26, -1.45, -1.50, -1.49]
        eta_list = [-0.243, -0.261, -0.261, -0.273,
                     -0.278, -0.301, -0.301, -0.319, -0.336]
        
        self._beta0_spline = InterpolatedUnivariateSpline(
            numpy.log(delta_list), beta_list)
        self._gamma0_spline = InterpolatedUnivariateSpline(
            numpy.log(delta_list), gamma_list)
        self._phi0_spline = InterpolatedUnivariateSpline(
            numpy.log(delta_list), phi_list)
        self._eta0_spline = InterpolatedUnivariateSpline(
            numpy.log(delta_list), eta_list)

        self._k_min = 0.001
        self._k_max = 100.0

        self._redshift = redshift
        #self.cosmo = cosmology.SingleEpoch(self._redshift, cosmo_dict)
        if cosmo_single_epoch is None:
            cosmo_single_epoch = cosmology.SingleEpoch(self._redshift)
        self.cosmo = cosmo_single_epoch
        self.cosmo.set_redshift(self._redshift)
        self.delta_c = self.cosmo.delta_c()
        self.delta_v = cosmo_single_epoch.delta_v()

        if halo_dict is None:
            halo_dict = defaults.default_halo_dict
        self.halo_dict = halo_dict

        self._set_mass_limits()
        self._initialize_splines()
        self._normalize()

    def f_m(self, mass):
        nu = self.nu(mass)
        return f_nu(nu)

    def _normalize(self):

        self.f_norm = 1.0
        norm = integrate.romberg(
            lambda x: self.f_nu(x)*self.bias_nu(x),
            self.nu_min, self.nu_max, vec_func=True,
            tol=defaults.default_precision["mass_precision"])
        self.f_norm = 1.0/norm

    def f_nu(self, nu):
        return (self.f_norm*(
                1 + numpy.power(self._beta()*nu,-2*self._phi()))*
                numpy.power(nu, 2*self._eta())*
                numpy.exp(-self._gamma()*nu*nu/2.0))

    def bias_nu(self, nu):
        y = numpy.log10(self.delta_v)
        A = 1 + 0.24*y*numpy.exp(-(4.0/y)**4)
        a = 0.44*y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107*y + 0.19*numpy.exp(-(4.0/y)**4)
        c = 2.4
        return (1 - A*nu**a/(nu**a + self.delta_c**a) + B*nu**b + C*nu**c)

    def _beta(self):
        return self._beta0_spline(numpy.log(self.delta_v))*numpy.power(
            1 + self._redshift, 0.20)
    
    def _phi(self):
        return self._phi0_spline(numpy.log(self.delta_v))*numpy.power(
            1 + self._redshift, -0.08)

    def _eta(self):
        return self._eta0_spline(numpy.log(self.delta_v))*numpy.power(
            1 + self._redshift, 0.27)

    def _gamma(self):
        return self._gamma0_spline(numpy.log(self.delta_v))*numpy.power(
            1 + self._redshift, -0.01)
