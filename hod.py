import numpy
from scipy import special

"""Classes for implementing a halo occupation distribution.

A halo occupation distribution describes the population of a given dark matter
halo.  In principle, this distribution can be a function of any number of halo
and galaxy variables.  For the purposes of this implementation, we allow for
the HOD to vary as a function of halo mass and redshift.  Any HOD-derived
object should be capable of returning the first and second moments of the
distribution as a function of mass and redshift.
"""

__author__ = "Chris Morrison <morrison.chrisb@gmail.com>"

class HOD(object):
    """Base class for a halo occupation distribution.

    The exact form of an HOD will vary depending on the galaxy type and
    redshift.  Our base class defines the API that all of the derived class
    instances need to implement.
    """
    def __init__(self):
        self._hod = {}
        self._hod[1] = self.first_moment
        self._hod[2] = self.second_moment

    def first_moment(self, mass, redshift=None):
        """
        Expected number of galaxies per halo, <N> as a function of mass and 
        redshift.
        
        Args:
            mass: float array Halo mass in M_Solar/h^2
            redshift: float redshift to evalute the first moment if redshift
                dependent
        Returns:
            float array of <N>
        """
        return 1.0

    def second_moment(self, mass, z=None):
        """
        Expected number of galaxy pairs per halo, <N(N-1)> as a function of mass
        and redshift.
        
        Args:
            mass: float array Halo mass in M_Solar/h^2
            redshift: float redshift to evalute the first moment if redshift
                dependent
        Returns:
            float array of <N(N-1)>
        """
        return 1.0

    def nth_moment(self, mass, z=None, n=3):
        """
        Expected number galaxy moment, <N(N-1)...(N-n+1)> as a function of mass
        and redshift.
        
        Args:
            mass: float array Halo mass in M_Solar/h^2
            redshift: float redshift to evalute the first moment if redshift
                dependent
            n: integer moment to compute
        Returns:
            float array of <N(N-1)...(N-n+1)>
        """
        exp_nth = 0.0
        if n in self._hod:
            exp_nth = self._hod[n](mass, z)
        else:
            exp_nth = self.first_moment(mass, z)**n
            alpha_m2 = numpy.where(
                exp_nth > 0.0,
                self.second_moment(mass, z)/self.first_moment(mass, z)**2, 0.0)
            for j in xrange(n):
                exp_nth *= (j*alpha_m2 - j + 1)
        return exp_nth

    def set_halo(self, halo_dict):
        pass

    def write(self, output_file_name):
        mass_max = 1.0e16
        mass_min = 1.0e9

        dln_mass = (numpy.log(mass_max) - numpy.log(mass_min))/200
        ln_mass_max = numpy.log(mass_max) + dln_mass
        ln_mass_min = numpy.log(mass_min) - dln_mass

        ln_mass_array = numpy.arange(
            ln_mass_min, ln_mass_max + dln_mass, dln_mass)

        f = open(output_file_name, "w")
        for ln_mass in ln_mass_array:
            mass = numpy.exp(ln_mass)
            f.write("%1.10f %1.10f %1.10f %1.10f\n" % (
                mass, self.first_moment(mass), self.second_moment(mass),
                self.nth_moment(mass,None,3)))
        f.close()

class HODPoisson(HOD):

    def __init__(self):
        pass

class HODBinomial(HOD):

    def __init__(self, n_max, min_mass, mass_max, p_m_spline):
        pass

class HODKravtsov(HOD):
    def __init__(self, halo_param=None, **kws):
        HOD.__init__(self)

        if halo_param is None:
            halo_param = param.HaloModelParams(**kws)
        self.set_halo(halo_param)

    def set_halo(self, halo_param):
        self.min_mass = halo_param.m11_msunh
        self.mass_one = halo_param.m13_msunh
        self.mass_cut = halo_param.mcut_msunh
        self.alpha = halo_param.alphaexp

    def first_moment(self, mass, z=None):
        return (self.central_first_moment(mass) +
                self.satellite_first_moment(mass))

    def second_moment(self, mass, z=None):
        n_sat = self.satellite_first_moment(mass)
        return (2 + n_sat)*n_sat

    def central_first_moment(self, mass):
        """
        Expected number of central galaxies in a halo, <N> as a function of 
        mass and redshift.
        
        Args:
            mass: float array Halo mass in M_Solar/h^2
            redshift: float redshift to evalute the first moment if redshift
                dependent
        Returns:
            float array of <N>
        """
        if mass >= self.min_mass:
            return 1.0
        else:
            return 0.0

    def satellite_first_moment(self, mass):
        """
        Expected number of satellite galaxies in a halo, <N> as a function of 
        mass and redshift.
        
        Args:
            mass: float array Halo mass in M_Solar/h^2
            redshift: float redshift to evalute the first moment if redshift
                dependent
        Returns:
            float array of <N>
        """
        return mass/self.mass_one*numpy.exp(-self.mass_cut/mass)

class HODZheng(HOD):
    """
    HOD object describing the model from Zheng2007.

    Attributes:
        M_min: Minimum mass for a halo to to contain a central galaxy
        sigma: with of central the central galaxy turn on
        M_0: minimum mass for a halo to contain satellite galaxies. Note: 
            Wake et al. 2011 and Zehavi et al. 2011 show show that M_0~M_min
        M_1p: Mass differential at which a halo contains one satellite (M-M_0).
        alpha: slope of the satellite number counts. Note: This is motivated to
            be 1. (see references above)
    """

    def __init__(self, M_min=10**13.0, sigma=0.15, 
                 M_0=10**13.0, M_1p=10**14.00, alpha=1.0):
        self.M_min = M_min
        self.sigma = sigma
        self.M_0 = M_0
        self.M_1p = M_1p
        self.alpha = alpha
        HOD.__init__(self)

    def first_moment(self, mass, z=None):
        return (self.central_first_moment(mass)+
                self.satellite_first_moment(mass))

    def second_moment(self, mass, z=None):
        n_sat = self.satellite_first_moment(mass)
        return (2 + n_sat)*n_sat

    def central_first_moment(self, mass):
        """
        Expected number of central galaxies in a halo, <N> as a function of 
        mass and redshift.
        
        Args:
            mass: float array Halo mass in M_Solar/h^2
            redshift: float redshift to evalute the first moment if redshift
                dependent
        Returns:
            float array of <N>
        """
        return 0.5*(1+special.erf((numpy.log10(mass) - numpy.log10(self.M_min))/
                                  self.sigma))
    
    def satellite_first_moment(self, mass):
        """
        Expected number of satellite galaxies in a halo, <N> as a function of 
        mass and redshift.
        
        Args:
            mass: float array Halo mass in M_Solar/h^2
            redshift: float redshift to evalute the first moment if redshift
                dependent
        Returns:
            float array of <N>
        """
        diff = mass - self.M_0
        return numpy.where(diff >= 0.0,
                           self.central_first_moment(mass)*
                           numpy.power(diff/self.M_1p, self.alpha),
                           0.0)

class HODMandelbaum(HOD):
    """
    HOD object describing the model from Mandelbaum2005.

    Attributes:
        M_0: Minimum mass for a halo to contain a central galaxy. Also defines
            the transition of the satellite galaxy profile from ~M**2 at <3*M_0
            to ~M at >3*M_0
        w: Normalization of the satellite galaxy profile. Halos will contain
            1 satellite galaxy at a mass of 3*M0/w. This quantity is related
            to the satellite fraction alpha of Mand2005 where alpha is
            M_total_sat/M_total where M_total mass of the halos studied and
            M_total_sat is the total mass in satellites.
    """

    def __init__(self, M_0=1.0e13, w=1.0):
        HOD.__init__(self)

        self.M_0 = M_0
        self.M_min = 3.0*M_0
        self.w = w

        self._hod[1] = self.first_moment
        self._hod[2] = self.second_moment

    def first_moment(self, mass, z=None):
        return (self.central_first_moment(mass) + 
                self.satellite_first_moment(mass))

    def second_moment(self, mass, z=None):
        n_sat = self.satellite_first_moment(mass)
        return (2 + n_sat)*n_sat

    def central_first_moment(self, mass, z=None):
        """
        Expected number of central galaxies in a halo, <N> as a function of 
        mass and redshift.
        
        Args:
            mass: float array Halo mass in M_Solar/h^2
            redshift: float redshift to evalute the first moment if redshift
                dependent
        Returns:
            float array of <N>
        """
        return numpy.where(mass >= self.M_0,
                           1.0, 0.0)

    def satellite_first_moment(self, mass, z=None):
        """
        Expected number of satellite galaxies in a halo, <N> as a function of 
        mass and redshift.
        
        Args:
            mass: float array Halo mass in M_Solar/h^2
            redshift: float redshift to evalute the first moment if redshift
                dependent
        Returns:
            float array of <N>
        """
        return numpy.where(mass < self.M_min,
                           (mass/self.M_min)**2*self.w,
                           mass/self.M_min*self.w)
