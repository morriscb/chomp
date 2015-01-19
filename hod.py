import defaults
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
    def __init__(self, hod_dict):
        self.hod_dict = hod_dict
        self._hod = {}
        self._hod[1] = self.first_moment
        self._hod[2] = self.second_moment
        
        ### These variables are useful for focusing the halo model mass integral
        ### on non-zero ranges of the integrand. Default -1 forces code to 
        ### integrate over the whole mass range.
        self.first_moment_zero = -1
        self.second_moment_zero = -1
        
        ### Optional value to declare at which mass value the hod is guaranteed
        ### to evaluate to positive definite.
        self._safe_norm = -1

    def first_moment(self, mass, z=None):
        """
        Expected number of galaxies per halo, <N> as a function of mass and 
        redshift.
        
        Args:
            mass: float array Halo mass in M_Solar/h^2
            z: float redshift to evalute the first moment if redshift
                dependent
        Returns:
            float array of <N>
        """
        return 1.0

    def second_moment(self, mass, z=None):
        """
        Expected number of galaxy pairs per halo, <N(N-1)> as a function of
        mass and redshift.
        
        Args:
            mass: float array Halo mass in M_Solar/h^2
            redshift: float redshift to evalute the first moment if redshift
                dependent
        Returns:
            float array of <N(N-1)>
        """
        return 1.0

    def nth_moment(self, mass, n=3, z=None):
        """
        Expected number galaxy moment, <N(N-1)...(N-n+1)> as a function of
        mass and redshift.
        
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
            first_mom = self.first_moment(mass, z)
            exp_nth = first_mom**n
            alpha_m2 = numpy.where(
                 first_mom != 0.0,
                self.second_moment(mass, z)/first_mom**2, 0.0)
            for j in xrange(n):
                exp_nth *= (j*alpha_m2 - j + 1)
        return exp_nth
    
    def get_hod(self):
        """
        Return the interanl dictionary defining an HOD.
        """
        return self.hod_dict
    
    def set_hod(self, hod_dict):
        """
        Set the internal valeus of the hod.
        
        Args:
            hod_dict: a dictionary defining an HOD.
        """
        self.__init__(hod_dict)

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

class HODZheng(HOD):
    """
    HOD object describing the model from Zheng2007. Input is a dictionary
    of variable names and values. Names for these variables are listed below.

    Attributes:
        M_min: Minimum mass for a halo to to contain a central galaxy
        sigma: with of central the central galaxy turn on
        M_0: minimum mass for a halo to contain satellite galaxies. Note: 
            Wake et al. 2011 and Zehavi et al. 2011 show show that M_0~M_min
        M_1p: Mass differential at which a halo contains one satellite (M-M_0).
        alpha: slope of the satellite number counts. Note: This is motivated to
            be 1. (see references above)
    """

    def __init__(self, hod_dict=None):
        if hod_dict is None:
            ### HOD parameters from Zehavi et al. 2011 with parameter 
            ### assumptions from Wake et al. 2011.
            self.log_M_min = 12.14
            self.sigma = 0.15
            self.log_M_0 = 12.14
            self.log_M_1p = 13.43
            self.alpha = 1.0
        else:
            self.log_M_min = hod_dict['log_M_min']
            self.sigma = hod_dict['sigma']
            self.log_M_0 = hod_dict['log_M_0']
            self.log_M_1p = hod_dict['log_M_1p']
            self.alpha = hod_dict['alpha']
        HOD.__init__(self, hod_dict)
        
        ### These variables are useful for focusing the halo model mass integral
        ### on non-zero ranges of the integrand. Default -1 forces code to 
        ### integrate over the whole mass range.
        self.first_moment_zero = numpy.power(
            10.0, self.log_M_min +      
            self.sigma*special.erfinv(
                2.*defaults.default_precision['halo_precision'] - 1.0))
        # self.first_moment_zero = numpy.power(10, self.log_M_min - 
        #                                      6*self.sigma)
        self.second_moment_zero = 10.0**self.log_M_0
        if self.second_moment_zero < self.first_moment_zero:
            self.secon_moment_zero = self.first_moment_zero
        self._safe_norm = 10.0**(self.log_M_min + 1.0*self.sigma)
        
        
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
            float array of <N_c>
        """
        if self.sigma <= 0.0:
            log_mass = numpy.log10(mass)
            return numpy.where(log_mass > self.log_M_min, 1.0, 0.0)
        return 0.5*(1+special.erf((numpy.log10(mass) - 
                                   self.log_M_min)/self.sigma))
    
    def satellite_first_moment(self, mass):
        """
        Expected number of satellite galaxies in a halo, <N> as a function of 
        mass and redshift.
        
        Args:
            mass: float array Halo mass in M_Solar/h^2
            redshift: float redshift to evalute the first moment if redshift
                dependent
        Returns:
            float array of <N_s>
        """
        diff = mass - numpy.power(10, self.log_M_0)
        return numpy.where(diff > 0.0,
                           self.central_first_moment(mass)*
                           numpy.power(diff/(10**self.log_M_1p), self.alpha),
                           0.0)

class HODMandelbaum(HOD):
    """
    HOD object describing the model from Mandelbaum2005. Input is a dictionary
    of variable names and values. Names for these variables are listed below.

    Attributes:
        log_M_0: Minimum mass for a halo to contain a central galaxy. Also
            defines the transition of the satellite galaxy profile from ~M**2 at
            <3*M_0 to ~M at >3*M_0
        w: Normalization of the satellite galaxy profile. Halos will contain
            1 satellite galaxy at a mass of 3*M0/w. This quantity is related
            to the satellite fraction alpha of Mand2005 where alpha is
            M_total_sat/M_total where M_total mass of the halos studied and
            M_total_sat is the total mass in satellites.
    """

    def __init__(self, hod_dict = None):

        if hod_dict is None:
            self.log_M_0 = 12.14
            self.log_M_min = numpy.log10(3.0) + 12.14
            self.w = 1.0
        else:
            self.log_M_0 = hod_dict["log_M_0"]
            self.log_M_min = numpy.log10(3.0) + hod_dict["log_M_0"]
            self.w = hod_dict["w"]
        
            HOD.__init__(self, hod_dict)


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
        return numpy.where(numpy.log10(mass) >= self.log_M_0,
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
        return numpy.where(numpy.log10(mass) < self.log_M_min,
                           (mass/10**self.log_M_min)**2*self.w,
                           mass/10**self.log_M_min*self.w)
