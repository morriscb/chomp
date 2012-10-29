import defaults
import numpy
from numpy import vectorize
from scipy import integrate
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline

"""Classes for encoding basic cosmological parameters and quantities.

This is inspired by the CosmoPy package.  The intent is to have a singleton
object that contains all of the basic cosmological parameters as well as those
expected by the CAMB software.  In addition, we want a single class object
that can take those parameters and return standard cosmological quantities
(comoving distance, linear growth factor, etc.) as a function of redshift.
"""

__author__ = ("Chris Morrison <morrison.chrisb@gmail.com>, "+
              "Ryan Scranton <ryan.scranton@gmail.com>")


class SingleEpoch(object):
    """Container class for calculating cosmological values at a given redshift.

    Given a redshift and a set of parameters, SingleEpoch can return the
    comoving distance, angular diameter distance or luminosity distance for.
    Likewise, it can return a redshift, given an input comoving distance, as
    well as the growth factor D(z) and other cosmological parameters.

    Attributes:
        redshift: float redshift at which to compute all cosmological values
        cosmo_dict: dictionary of float values that define the cosmology 
            (see default.py)
    """

    def __init__(self, redshift, cosmo_dict=None, with_bao=False, **kws):
        if redshift < 0.0:
            redshift = 0.0
        self._redshift = redshift

        if cosmo_dict is None:
            cosmo_dict = defaults.default_cosmo_dict

        self.cosmo_dict = cosmo_dict

        self._omega_m0 = cosmo_dict["omega_m0"]
        self._omega_b0 = cosmo_dict["omega_b0"]
        self._omega_l0 = cosmo_dict["omega_l0"]
        self._omega_r0 = cosmo_dict["omega_r0"]
        self._cmb_temp = cosmo_dict["cmb_temp"]
        self._h = cosmo_dict["h"]
        self._sigma_8 = cosmo_dict["sigma_8"]
        self._n = cosmo_dict["n_scalar"]
        self._w0 = cosmo_dict["w0"]
        self._wa = cosmo_dict["wa"]
        self.H0 = 100.0/(3.0*10**5)

        self._flat = True
        self._open = False
        self._closed = False

        omega_total = self._omega_m0 + self._omega_l0
        if omega_total <= 1.0001 and omega_total >= 0.9999:
            self._flat = True
        else:
            self._flat = False
        if omega_total <= 0.9999:
            self._open = True
        else:
            self._open = False
        if omega_total > 1.0001:
            self._closed = True
        else:
            self._closed = False

        self._k_min = defaults.default_limits['k_min']
        self._k_max = defaults.default_limits['k_max']
        self.delta_H = (
            1.94e-5*self._omega_m0**(-0.785 - 0.05*numpy.log(self._omega_m0))*
            numpy.exp(-0.95*(self._n - 1) - 0.169*(self._n - 1)**2))

        self._with_bao = with_bao

        self._initialize_defaults()

    def _initialize_defaults(self):
        if self._w0 != -1.0 or self._wa != 0.0:
            a_array = numpy.logspace(-4, 0,
                defaults.default_precision["cosmo_npoints"])
            self._de_pressure_array = self._de_pressure(1/a_array - 1.0)
            self._de_pressure_spline = InterpolatedUnivariateSpline(
                numpy.log(a_array), self._de_pressure_array)

        self._chi = integrate.romberg(
            self.E, 0.0, self._redshift, vec_func=True,
            tol=defaults.default_precision["global_precision"],
            rtol=defaults.default_precision["cosmo_precision"],
            divmax=defaults.default_precision["divmax"])

        self.growth_norm = integrate.romberg(
            self._growth_integrand, 1e-16, 1.0, vec_func=True,
            tol=defaults.default_precision["global_precision"],
            rtol=defaults.default_precision["cosmo_precision"],
            divmax=defaults.default_precision["divmax"])
        self.growth_norm *= 2.5*self._omega_m0*numpy.sqrt(self.E0(0.0))

        a = 1.0/(1.0 + self._redshift)
        growth = integrate.romberg(
            self._growth_integrand, 1e-16, a, vec_func=True,
            tol=defaults.default_precision["global_precision"],
            rtol=defaults.default_precision["cosmo_precision"],
            divmax=defaults.default_precision["divmax"])
        growth *= 2.5*self._omega_m0*numpy.sqrt(self.E0(self._redshift))
        self._growth = growth/self.growth_norm

        self._sigma_norm = 1.0
        self._sigma_norm = self._sigma_8*self._growth/self.sigma_r(8.0)

    def set_redshift(self, redshift):
        """
        Sets internal variable _redshift to new value and re-initializes splines

        Args:
            redshift: redshift value at which to compute all cosmology values
        Returns:
            None
        """
        if redshift != self._redshift:
            self._redshift = redshift
            self._initialize_defaults()

    def set_cosmology(self, cosmo_dict, redshift=None):
        """
        Resets cosmology dictionary and internal values to new cosmology

        Args:
            cosmo_dict: a dictionary of floats containing cosmological 
                information (see defaults.py for details)
        """

        if redshift is None:
            redshift = self.redshift
        self.__init__(redshift, cosmo_dict)

    def E(self, redshift):
        """
        1/H(z). Used to compute cosmological distances.

        Args:
            redshift: redshift at which to compute E.
        Returns:
            Float value of 1/H at z=redshift.
            1/H(redshift)
        """
        return 1.0/(self.H0*numpy.sqrt(self.E0(redshift)))

    def E0(self, redshift):
        """
        (H(z)/H0)^2 aka Friedmen's equation. Used to compute various 
        redshift dependent cosmological factors.

        Args:
            redshift: redshift at which to compute E0.
        Returns:
            Float value of (H(z)/H0)^2 at z=redshift
        """
        a = 1.0/(1.0 + redshift)
        if self._w0 == -1.0 and self._wa == 0.0:
            return (self._omega_l0 + 
                    self._omega_m0/(a*a*a) + self._omega_r0/(a*a))
        else:
            return (self._omega_l0*numpy.exp(
                    self._de_pressure_spline(numpy.log(a)))
                    + self._omega_m0/(a*a*a) + self._omega_r0/(a*a))
 
    def w(self, redshift):
        """
        Redshift dependent Dark Energy w

        Args:
            redshift: float array redshift
        Returns:
            float array Dark Energy w
        """
        a = 1.0/(1 + redshift)
        return self._w0 + self._wa*(1 - a)

    def _de_pressure(self, redshift):
        dpressuredz = lambda z: (1. + self.w(z))/(1. + z)

        try:
            pressure = numpy.empty(len(redshift))
            for idx, z in enumerate(redshift):
                pressure[idx] = 3.0*integrate.romberg(
                    dpressuredz, 0, z, vec_func=True,
                    tol=defaults.default_precision["global_precision"],
                    rtol=defaults.default_precision["cosmo_precision"],
                    divmax=defaults.default_precision["divmax"])
        except TypeError:
            pressure = 3.0*integrate.romberg(
                dpressuredz, 0, redshift, vec_func=True,
                tol=defaults.default_precision["global_precision"],
                rtol=defaults.default_precision["cosmo_precision"],
                divmax=defaults.default_precision["divmax"])
        return pressure

    def _growth_integrand(self, a):
        """
        Integrand for growth factor as a function of scale

        Args:
            a: float array scale factor
        Returns:
            float array dgrowth/da
        """
        redshift = 1.0/a - 1.0
        return ((1.0 + redshift)/numpy.sqrt(self.E0(redshift)))**3

    @staticmethod
    def _growth_integrand_dynde(G, a, cosmo):
        """
        Derivatives of normalized linear growth function G := D/a and dG/da.
        For dynamical dark energy models, there is no closed integral
        expression for the growth function and the defining differential
        equation must be solved numerically.

        This function is intended as input to scipy.integrate.odeint.
        
        Args:
            G: vector of the state variables (G,G')
            a: scale factor
            cosmo: Instance of cosmology.SingleEpoch

        References:
            Linder E. V., Jenkins A., 2003, MNRAS, 346, 573. (eq. 11)
        """
        z = 1. / a - 1.
        g1, g2 = G
        X = numpy.exp(-cosmo._de_pressure(numpy.log(a)))
        X *= cosmo._omega_m0 / cosmo._omega_l0
        w = cosmo.w(z)
        # create f = (x1', x2')
        f = [(-(3.5 - 1.5 * w / (1. + X)) * g1/a -
              1.5 * ((1. - w) / (1. + X)) * g2 / a ** 2),
             g1]
        return f

    def D1_odesol(self, z):
        """
        Linear growth function obtained by solving the ODE
        (see Linder & Jenkins 2003, eq. 11)
        """
        # ----- Initial conditions
        G0 = [0., 1.] # (dG/da(a<<1), G(a<<1))
        # ----- ODE solver parameters
        abserr = 1.0e-8
        relerr = 1.0e-6
        a = (1. / (1. + z))[::-1]
        gsol = integrate.odeint(self._growth_integrand_dynde, G0, a,
                                args=(self,),
                                atol=abserr, rtol=relerr)
        return gsol[:,1] / (1.+z)    

    def comoving_distance(self):
        """
        Comoving distance at redshift self._redshif in units Mpc/h.
        
        Returns:
            Float comoving distance
        """
        return self._chi

    def luminosity_distance(self):
        """
        Luminosity distance at redshift self._redshif in units Mpc/h.

        Returns:
            Float Luminosity Distance
        """
        return (1.0 + self._redshift)*self._chi

    def angular_diameter_distance(self):
        """
        Angular diameter distance at redshift self._redshif in 
        units Mpc/h.

        Returns:
            Float Angular diameter Distance
        """
        return self._chi/(1.0 + self._redshift)

    def redshift(self):
        """
        Getter for internal value _redshift

        Returns:
            float redshift
        """
        return self._redshift

    def growth_factor(self):
        """
        Linear growth factor, normalized to unity at z = 0.

        Returns:
            float growth factor
        """
        return self._growth

    def omega_m(self):
        """
        Total mass denisty at redshift z = _redshift.

        Returns:
            float total mass density relative to critical
        """
        return self._omega_m0*(1.0 + self._redshift)**3/self.E0(self._redshift)

    def omega_l(self):
        """
        Dark Energy denisty at redshift z = _redshift.

        Returns:
            float dark energy density relative to critical
        """
        return self._omega_l0/self.E0(self._redshift)

    def delta_c(self):
        """
        Over-density threshold for linear, spherical collapse a z=_redshift.

        Returns:
            Float threshold
        """
        # Fitting function taken from NFW97
        delta_c = 0.15*(12.0*numpy.pi)**(2.0/3.0)
        if self._open:
            delta_c *= self.omega_m()**0.0185
        if self._flat and self._omega_m0 < 1.0001:
            delta_c *= self.omega_m()**0.0055

        return delta_c

    def delta_v(self):
        """
        Over-density for a collapsed, virialized halo.

        Returns:
            Float over-density for virialized halo.
        """
        # Fitting function taken from NFW97
        delta_v = 178.0
        if self._open:
            delta_v /= self.omega_m()**0.7
        if self._flat and self._omega_m0 < 1.0001:
            delta_v /= self.omega_m()**0.55

        return delta_v/self._growth

    def rho_crit(self):
        """
        Critical density in h^2 solar masses per cubic Mpc.

        Returns:
            float critical density
        """
        return 1.0e-29*1.0e-33*2.937999e+73*self.E0(self._redshift)
        #return (1.879e-29/(1.989e33)*numpy.power(3.086e24,3)*
        #        self.E0(self._redshift))

    def rho_bar(self):
        """
        Matter density in h^2 solar masses per cubic Mpc.

        Returns:
            float average matter desnity
        """
        return self.rho_crit()*self.omega_m()

    def _eh_transfer(self, k):
        """
        Eisenstein & Hu (1998) fitting function without BAO wiggles
        (see eqs. 26,28-31 in their paper)

        Args:
            k: float array wave number at which to compute power spectrum.
        Returns:
            float array Transfer function T(k).
        """

        theta = self._cmb_temp/2.7 # Temperature of CMB_2.7
        Omh2 = self._omega_m0*self._h**2
        Omb2 = self._omega_b0*self._h**2
        omega_ratio = self._omega_b0/self._omega_m0
        s = 44.5*numpy.log(9.83/Omh2)/numpy.sqrt(1+10.0*(Omb2)**(3/4))
        alpha = (1 - 0.328*numpy.log(431.0*Omh2)*omega_ratio +
                 0.38*numpy.log(22.3*Omh2)*omega_ratio**2)
        Gamma_eff = self._omega_m0*self._h*(
            alpha + (1-alpha)/(1+0.43*k*s)**4)
        q = k*theta/Gamma_eff
        L0 = numpy.log(2*numpy.e + 1.8*q)
        C0 = 14.2 + 731.0/(1+62.5*q)
        return L0/(L0 + C0*q*q)

    def _eh_bao_transfer(self, k):
        """
        Eisenstein & Hu (1998) fitting function with BAO wiggles
        (see eqs. 26,28-31 in their paper)

        Args:
            k: float array wave number at which to compute power spectrum.
        Returns:
            float array Transfer function T(k).
        """
        theta = self._cmb_temp/2.7
        Ob = self._omega_b0
        Om = self._omega_m0
        Oc = Om-Ob
        O  = Om
        h = self._h
        Omh2 = Om*h**2
        Obh2 = Ob*h**2
        Oh2  = O*h**2
        ObO  = Ob/O
        
        zeq = 2.5e4*Oh2*theta**(-4)
        keq = 7.46e-2*Oh2*theta**(-2)
        b1  = 0.313*Oh2**(-0.419)*(1.+0.607*Oh2**0.674)
        b2  = 0.238*Oh2**0.223
        zd  = 1291.*(Oh2**0.251/(1.+0.659*Oh2**0.828))*(1.+b1*Obh2**b2)
        
        R = lambda z: 31.5*Obh2*theta**(-4)*(1000./z)
        Req = R(zeq)
        Rd  = R(zd)
        
        s = (2./(3.*keq))*numpy.sqrt(6./Req)*numpy.log(
            (numpy.sqrt(1.+Rd)+numpy.sqrt(Rd+Req))/(1.+numpy.sqrt(Req)))
        ks = k*h*s
        
        kSilk = 1.6*Obh2**0.52*Oh2**0.73*(1.+(10.4*Oh2)**(-0.95))
        q = lambda k: k*h/(13.41*keq)
        
        G = lambda y: y*(
            -6.*numpy.sqrt(1.+y)+(2+3*y)*numpy.log(
            (numpy.sqrt(1.+y)+1.)/(numpy.sqrt(1.+y)-1.)))
        alpha_b = 2.07*keq*s*(1.+Rd)**(-3./4.)*G((1.+zeq)/(1.+zd))
        beta_b = 0.5 + (ObO) + (3.-2.*ObO)*numpy.sqrt((17.2*Oh2)**2+1.)
        
        C   = lambda x,a: (14.2/a) + 386./(1.+69.9*q(x)**1.08)
        T0t = lambda x,a,b: numpy.log(numpy.e+1.8*b*q(x))/(
            numpy.log(numpy.e+1.8*b*q(k))+C(x,a)*q(x)**2)
        
        a1 = (46.9*Oh2)**0.670*(1.+(32.1*Oh2)**(-0.532))
        a2 = (12.*Oh2)**0.424*(1.+(45.*Oh2)**(-0.582))
        alpha_c = a1**(-ObO)*a2**(-ObO**3)
        b1 = 0.944*(1.+(458.*Oh2)**(-0.708))**(-1)
        b2 = (0.395*Oh2)**(-0.0266)
        beta_c = 1./(1.+b1*((Oc/O)**b2-1))
        
        f = 1./(1.+(ks/5.4)**4)
        Tc = f*T0t(k,1,beta_c) + (1.-f)*T0t(k,alpha_c,beta_c)
        
        beta_node = 8.41*(Oh2**0.435)
        stilde = s/(1.+(beta_node/(ks))**3)**(1./3.)
        
        Tb1 = T0t(k,1.,1.)/(1.+(ks/5.2)**2)
        Tb2 = (alpha_b/(1.+(beta_b/ks)**3))*numpy.exp(-(k*h/kSilk)**1.4)
        Tb = numpy.sinc(k*stilde/numpy.pi)*(Tb1+Tb2)
        return ObO*Tb + (Oc/O)*Tc

    def _bbks_Transfer(self, k):
        """
        BBKS transfer function.

        Args:
            k: float array wave number at which to compute power spectrum.
        Returns:
            float array Transfer function T(k).
        """
        Gamma = self._omega_m0*self._h
        #q = k/Gamma
        q = k/Gamma*numpy.exp(
            self._omega_b0 + 
            numpy.sqrt(2*self._h)*self._omega_b0/self._omega_m0)
        return (numpy.log(1.0 + 2.34*q)/(2.34*q)*
                (1 + 3.89*q + (16.1*q)**2 + (5.47*q)**3 + (6.71*q)**4)**(-0.25))

    def transfer_function(self, k):
        """
        Function for returning the CMB transfer function. Class variable 
        with_bao determines if the transfer function is the E+H98 fitting
        function with or without BAO wiggles.

        Args:
            k [Mpc/h]: float array wave number at which to compute the transfer
            function
        Returns:
            float array CMB transfer function
        """
        if self._with_bao:
            return self._eh_bao_transfer(k)
        else:
            return self._eh_transfer(k)
    
    def delta_k(self, k):
        """
        k^3*P(k)/2*pi^2: dimensionless linear power spectrum normalized to by
        sigma_8

        Args:
            k: float array Wave number at which to compute power spectrum.
        Returns:
            float array dimensionless linear power spectrum k^3*P(k)/2*pi^2
        """
        delta_k = (self.delta_H**2*(k/self.H0)**(3 + self._n)*
                   self.transfer_function(k)**2)/self._h
        return delta_k*(
            self._growth*self._growth*self._sigma_norm*self._sigma_norm)

    def linear_power(self, k):
        """
        Linear power spectrum P(k) in units Mpc^3/h^3 normalized to by sigma_8.

        Args:
            k: float array Wave number at which to compute power spectrum.
        Returns:
            float array linear power spectrum P(k)
        """
        return numpy.where(
            k > 1e-16,
            2.0*numpy.pi*numpy.pi*self.delta_k(k)/(k*k*k), 1e-16)

    def sigma_r(self, scale):
        """
        RMS power on scale in Mpc/h. sigma_8 is defined as sigma_r(8.0).
        
        Args:
            scale: length scale on which to compute RMS power
        Returns:
            float RMS power at scale
        """
        k_min = self._k_min
        k_max = self._k_max
        needed_k_min = 1.0/scale/10.0
        needed_k_max = 1.0/scale*14.0662 ### 4 zeros of the window function
        if (needed_k_min <= k_min and
            needed_k_min > self._k_min/100.0):
            k_min = needed_k_min
        elif (needed_k_min <= k_min and
              needed_k_min <= self._k_min/100.0):
            k_min = self._k_min/100.0
            # print "In cosmology.SingleEpoch.sigma_r:"
            # print "\tWARNING: Requesting scale greater than k_min."
            # print "\tExtrapolating to k_min=",k_min
        elif (needed_k_max >= k_max and
              needed_k_max < self._k_max*100.0):
            k_max = needed_k_max
        elif (needed_k_max >= k_max and
              needed_k_max >= self._k_max*100.0):
            k_max = self._k_max*100.0
            # print "In cosmology.SingleEpoch.sigma_r:"
            # print "\tWARNING: Requesting scale greater than k_max."
            # print "\tExtrapolating to k_max=",k_max

        sigma2 = integrate.romberg(
            self._sigma_integrand, numpy.log(k_min),
            numpy.log(k_max), args=(scale,), vec_func=True,
            tol=defaults.default_precision["global_precision"],
            rtol=defaults.default_precision["cosmo_precision"],
            divmax=defaults.default_precision["divmax"])
        sigma2 /= 2.0*numpy.pi*numpy.pi

        return numpy.sqrt(sigma2)

    def _sigma_integrand(self, ln_k, scale):
        """
        Integrand to compute sigma_r

        Args:
            ln_k: float array natural log of wave number
            scale: float scale in Mpc/h
        Returns:
            float array dsigma_r/dln_k_(scale)
        """
        k = numpy.exp(ln_k)
        dk = 1.0*k
        kR = scale*k

        W = 3.0*(
            numpy.sin(kR)/kR**3-numpy.cos(kR)/kR**2)

        return dk*self.linear_power(k)*W*W*k*k

    def sigma_m(self, mass):
        """
        RMS power on scale subtended by total mean mass in solar masses/h.

        Args:
            mass: float mean mass at which to compute RMS power
        Returns:
            float fluctuation for at mass
        """
        scale = (3.0*mass/(4.0*numpy.pi*self.rho_bar()))**(1.0/3.0)

        return self.sigma_r(scale)

    def nu_r(self, scale):
        """
        Ratio of (delta_c/sigma(R))^2.

        Args:
            scale: float length scale on which to compute nu
        Returns:
            float normalized fluctuation
        """
        sqrt_nu = self.delta_c()/self.sigma_r(scale)
        return sqrt_nu*sqrt_nu

    def nu_m(self, mass):
        """
        Ratio of (delta_c/sigma(M))^2. Used as the integration variable for
        halo.py and mass.py. Determains at which mean mass a halo has
        collapsed.

        Args:
            mass: float mean mass at which to compute nu
        Returns:
            float normalized fluctuation
        """
        sqrt_nu = self.delta_c()/self.sigma_m(mass)
        return sqrt_nu*sqrt_nu

    def write(self, output_power_file_name=None):
        """
        Output current derived values of cosmology from cosmo_dict and redshift

        Args:
            output_power_file_name: string of output power spectra name
        """
        print "z = %1.4f" % self._redshift
        print "Comoving distance = %1.4f" % self._chi
        print "Growth factor = %1.4f" % self._growth
        print "Omega_m(z) = %1.4f" % self.omega_m()
        print "Omega_l(z) = %1.4f" % self.omega_l()
        print "DE w(z)    = %1.4f" % self.w(self._redshift)
        print "Delta_V(z) = %1.4f" % self.delta_v()
        print "delta_c(z) = %1.4f" % self.delta_c()
        print "sigma_8(z) = %1.4f" % self.sigma_r(8.0)

        if not output_power_file_name is None:
            dln_k = (numpy.log(self._k_max) - numpy.log(self._k_min))/200
            ln_k_max = numpy.log(self._k_max) + dln_k
            ln_k_min = numpy.log(self._k_min) - dln_k

            f = open(output_power_file_name, "w")
            f.write("#ttype1 = k [Mpc/h]\n#ttype2 = P(k) [(Mpc/h)^3]\n")
            for ln_k in numpy.arange(ln_k_min, ln_k_max + dln_k, dln_k):
                k = numpy.exp(ln_k)
                f.write("%1.10f %1.10f\n" % (k, self.linear_power(k)))
            f.close()


class MultiEpoch(object):
    """Class for calculating cosmological values across a range of redshifts.

    Given an initial set of redshift bounds and a set of parameters, MultiEpoch
    can return the comoving distance, angular diameter distance or luminosity
    distance for any redshift over the input interval.  Likewise, it can return
    a redshift, given an input comoving distance, as well as the growth factor
    D(z) and a number of other cosmological parameters.

    Atributes:
        z_min: float minimum redshift to compute cosmology
        z_max: float maximum redshift to comput cosmology
        cosmo_dict: dictionary of float values that define the cosmology 
            (see default.py for details)
    """

    def __init__(self, z_min, z_max, cosmo_dict=None, with_bao=False, **kws):
        self.z_min = z_min
        self.z_max = z_max
        if self.z_min < 0.0: self.z_min = 0.0

        self._z_array = numpy.linspace(
            self.z_min, self.z_max,
            defaults.default_precision["cosmo_npoints"])
        self._chi_array = numpy.zeros_like(self._z_array)
        self._growth_array = numpy.zeros_like(self._z_array)

        if cosmo_dict is None:
            cosmo_dict = defaults.default_cosmo_dict

        self.epoch0 = SingleEpoch(0.0, cosmo_dict, with_bao, **kws)

        self._omega_m0 = self.epoch0._omega_m0
        self._omega_b0 = self.epoch0._omega_b0
        self._omega_l0 = self.epoch0._omega_l0
        self._h = self.epoch0._h
        self.H0 = self.epoch0.H0
        self._omega_r0 = self.epoch0._omega_r0
        self._sigma_8 = self.epoch0._sigma_8
        self._w0 = self.epoch0._w0
        self._wa = self.epoch0._wa

        if self._w0 != -1:
            self._de_pressure_array = numpy.zeros_like(self._z_array)

        self._flat = self.epoch0._flat
        self._open = self.epoch0._open
        self._closed = self.epoch0._closed

        self._k_min = self.epoch0._k_min
        self._k_max = self.epoch0._k_max
        self._n = self.epoch0._n
        self.delta_H = self.epoch0.delta_H

        self._initialize_splines()

    def _initialize_splines(self):
        for idx in xrange(self._z_array.size):
            dist = integrate.romberg(
                self.epoch0.E, 0.0, self._z_array[idx], vec_func=True,
                tol=defaults.default_precision["global_precision"],
                rtol=defaults.default_precision["cosmo_precision"],
                divmax=defaults.default_precision["divmax"])
            self._chi_array[idx] = dist
        self._chi_spline = InterpolatedUnivariateSpline(
            self._z_array, self._chi_array)
        self._z_spline = InterpolatedUnivariateSpline(
            self._chi_array, self._z_array)

        self.growth_norm = self.epoch0.growth_norm

        for idx in xrange(self._z_array.size):
            a = 1.0/(1.0 + self._z_array[idx])
            growth = integrate.romberg(
                self.epoch0._growth_integrand, 1e-16, a, vec_func=True,
                tol=defaults.default_precision["global_precision"],
                rtol=defaults.default_precision["cosmo_precision"],
                divmax=defaults.default_precision["divmax"])
            growth *= 2.5*self._omega_m0*numpy.sqrt(
                self.epoch0.E0(self._z_array[idx]))
            self._growth_array[idx] = growth/self.growth_norm

        self._growth_spline = InterpolatedUnivariateSpline(
            self._z_array, self._growth_array)

    def set_redshift(self, z_min, z_max):
        """
        Reset the minimum and maxinum redshift to evaluate the cosmology.
        
        Args:
            z_min: float value for minimum redshift
            z_max: float value for maximum redshift
        """
        self.z_max = z_max
        self.z_min = z_min
        if self.z_min < 0.0: self.z_min = 0.0

        self._z_array = numpy.linspace(
            self.z_min, self.z_max,
            defaults.default_precision["cosmo_npoints"])
        self._chi_array = numpy.zeros_like(self._z_array)
        self._growth_array = numpy.zeros_like(self._z_array)

        self._initialize_splines()

    def set_cosmology(self, cosmo_dict, z_min=None, z_max=None):
        """
        Reset the cosmology to the new values contained within cosmo_dict. Also
        reinitialize the maximum and minimum redshift if desired.

        Args:
            cosmo_dict: dictionary of float values containing cosmological 
                parameters (see defaults.py for details)
            z_min: float value for minimum redshift
            z_max: float value for maximum redshift
        """
        if z_min is None:
            z_min = self.z_min
        if z_max is None:
            z_max = self.z_max
        self.__init__(z_min, z_max, cosmo_dict)

    def E(self, redshift):
        """
        1/H(z). Used to compute cosmological distances. Units are Mpc/h

        Args:
            redshift: float array redshift at which to compute E.
        Returns:
            Float array of 1/H.
        """
        return self.epoch0.E(redshift)

    def comoving_distance(self, redshift):
        """
        Returns the comoving distance at redshift self._redshift in units Mpc/h.

        Args:
            redshift: float array redshift at which to caculate the comoving 
                distance
        Returns:
            float array comoving distance in Mpc/h
        """
        distance = 0.0

        distance = numpy.where(numpy.logical_and(redshift<=self.z_max,
                                                 redshift>=self.z_min),
                               self._chi_spline(redshift), 0.0)
        if numpy.any(numpy.logical_or(redshift>self.z_max,
                                      redshift<self.z_min)):
            print ("In cosmology.MultiEpoch.comoving_distance:"
                   "\n\tWarning: a requested redshift was outside of bounds!  "
                   "\n\tReturning 1e-16 for this redshift.")

        return distance

    def luminosity_distance(self, redshift):
        """
        Returns the luminosity distance at redshift self._redshift in units
        Mpc/h.

        Args:
            redshift: float array redshift at which to caculate the luminosity
                distance
        Returns:
            float array luminosity distance in Mpc/h
        """
        return (1.0 + redshift)*self.comoving_distance(redshift)

    def angular_diameter_distance(self, redshift):
        """
        Returns the luminosity distance at redshift self._redshift in units
        Mpc/h.

        Args:
            redshift: float array redshift at which to caculate the luminosity
                distance
        Returns:
            float array luminosity distance in Mpc/h
        """
        return self.comoving_distance(redshift)/(1.0 + redshift)

    def redshift(self, comoving_distance):
        """
        Given a comoving distance in Mpc/h return the redshift to this distance
        assuming the observer as at redshift=0.

        Args:
            comoving_distance: float array comoving distance in Mpc/h
        Returns:
            float array of redshift.
        """
        return self._z_spline(comoving_distance)

    def growth_factor(self, redshift):
        """
        Linear growth factor, normalized to unity at z = 0.

        Args:
            redshift: float array at which to compute the growth factor
        Returns:
            float array of the growth factor
        """
        growth = 1.0

        growth = numpy.where(numpy.logical_and(redshift<=self.z_max,
                                               redshift>=self.z_min),
                             self._growth_spline(redshift), 1.0)
        # if numpy.any(numpy.logical_or(redshift>self.z_max,
        #                               redshift<self.z_min)):
        #     print ("Warning: a requested redshift was outside of bounds!  "
        #            "Returning unity for this redshift.")

        return growth

    def omega_m(self, redshift=None):
        """
        Total mass density as a precent of the critical density as a function 
        of redshift

        Args:
            redshift: float array at which to compute omega_m
        Returns:
            omega_m: a float that is total mass density as a precent of the 
            critical density at z=redshift.
        """
        if redshift is None: redshift = 0.0
        return self._omega_m0*(1.0 + redshift)**3/self.epoch0.E0(redshift)

    def omega_l(self, redshift=None):
        """
        Dark Energy density as a function of the critical density as a function
        of redshift

        Args:
            redshift: float array at which to compute omega_l
        Returns:
            float array that is dark energy density as a precent of the 
            critical density
        """
        if redshift is None: redshift = 0.0
        return self._omega_l0/self.epoch0.E0(redshift)

    def delta_c(self, redshift=None):
        """
        Over-density threshold for linear, spherical collapse.

        Args:
            redshift: float array redshift to compute delta_c
        Returns:
            float array delta_c
        """
        # Fitting function taken from NFW97
        delta_c = 0.15*(12.0*numpy.pi)**(2.0/3.0)
        if self._open:
            delta_c *= self.omega_m(redshift)**0.0185
        if self._flat and self._omega_m0 < 1.0001:
            delta_c *= self.omega_m(redshift)**0.0055
        if redshift is None:
            return delta_c
        else:
            return delta_c/self.growth_factor(redshift)

    def delta_v(self, redshift=None):
        """
        Over-density threshold for a virialized halo

        Args:
            redshift: float array redshift to compute delta_v
        Returns:
            float array delta_v
        """
        # Fitting function taken from NFW97
        delta_v = 178.0
        if self._open:
            delta_v /= self.omega_m(redshift)**0.7
        if self._flat and self._omega_m0 < 1.0001:
            delta_v /= self.omega_m(redshift)**0.55
        if redshift is None:
            return delta_v
        else:
            return delta_v/self.growth_factor(redshift)

    def rho_crit(self, redshift=None):
        """
        Critical density in solar masses per cubic Mpc.

        Args:
            redshift: float array redshift to compute rho_crit
        Returns:
            float array rho_crit
        """
        if redshift is None: redshift = 0.0
        return self.epoch0.E0(redshift)*1.0e-29*1.0e-33*2.937999e+73

    def rho_bar(self, redshift=None):
        """
        Critical density in solar masses per cubic Mpc.

        Args:
            redshift: float array redshift to compute rho_bar
        Returns:
            float array rho_bar
        """
        return self.rho_crit(redshift)*self.omega_m(redshift)

    def delta_k(self, k, redshift=None):
        """
        k^3*P(k)/2*pi^2: dimensionless linear power spectrum normalized to by
        sigma_8

        Args:
            k: float array wave number at which to compute power spectrum.
        Returns:
            float array dimensionless linear power spectrum k^3*P(k)/2*pi^2
        """
        delta_k = self.epoch0.delta_k(k)
        if not redshift is None:
            delta_k *= self.growth_factor(redshift)**2
        return delta_k

    def linear_power(self, k, redshift=None):
        """
        Linear power spectrum P(k) in units Mpc^3/h^3 normalized to by sigma_8.

        Args:
            k: float array wave number at which to compute power spectrum.
        Returns:
            float array linear power spectrum P(k)
        """
        return 4.0*numpy.pi*numpy.pi*self.delta_k(k, redshift)/(k*k*k)

    def sigma_r(self, scale, redshift=None):
        """
        RMS power on scale in Mpc/h. sigma_8 is defined as sigma_r(8.0).
        
        Args:
            scale: float length scale on which to compute RMS power
            redshift: float array redshift value to compute sigma_r
        Returns:
            float array RMS power
        """
        sigma = self.epoch0.sigma_r(scale)
        if redshift is not None:
            sigma *= self.growth_factor(redshift)

        return sigma

    def sigma_m(self, mass, redshift=None):
        """
        RMS power on scale in Mpc/h. sigma_8 is defined as sigma_r(8.0).
        
        Args:
            mass: M_solar mass scale on which to compute RMS power
            redshift: float array redshift value to compute sigma_r
        Returns:
            float array RMS power
        """
        """RMS power on scale subtended by total mean mass in solar masses."""
        scale = (3.0*mass/(4.0*numpy.pi*self.rho_bar(redshift)))**(1.0/3.0)

        return self.sigma_r(scale, redshift)

    def nu_r(self, scale, redshift=None):
        """
        Ratio of (delta_c/sigma(R))^2.

        Args:
            mass: M_solar scale on which to compute nu
            redshift: float array redshift value to compute nu_m
        Returns:
            float array normalized fluctuation of scale
        """
        sqrt_nu = self.delta_c(redshift)/self.sigma_r(scale, redshift)
        return sqrt_nu*sqrt_nu

    def nu_m(self, mass, redshift=None):
        """
        Ratio of (delta_c/sigma(R))^2.

        Args:
            scale: length scale on which to compute nu
            redshift: float array redshift value to compute nu_r
        Returns:
            float array normalized fluctuation of mass 
        """
        sqrt_nu = self.delta_c(redshift)/self.sigma_m(mass, redshift)
        return sqrt_nu*sqrt_nu

    def write(self, output_file_name, output_power_file_name=None):
        """
        Write current values of all splines from z=z_min - z_max
        
        Args:
            output_file_name: string name of file to write output cosmology to
            output_power_file_name: string name of file to write output 
                powerspectra to
        Returns:
            None
        """
        f = open(output_file_name, "w")
        f.write("#ttype1 = z\n#ttype2 = chi [Mpc/h]\n#ttype3 = growth\n"
                "#ttype4 = omega_m\n#ttype5 = omega_l\n#ttype6 = delta_c\n"
                "#ttype7 = delta_v\n#ttype8 = sigma_8\n")
        for z, chi, growth in zip(
            self._z_array, self._chi_array, self._growth_array):
            if z <= self.z_max:
                f.write(
                    "%1.10f %1.10f %1.10f %1.10f "
                    "%1.10f %1.10f %1.10f %1.10f\n" % (
                    z, chi, growth, self.omega_m(z), self.omega_l(z),
                    self.delta_c(z), self.delta_v(z), self.sigma_r(8.0, z)))
        f.close()

        if not output_power_file_name is None:
            k_min = self._k_min
            k_max = self._k_max

            ln_k_array = numpy.linspace(numpy.log(k_min), numpy.log(k_max), 100)
            f = open(output_power_file_name, "w")
            for ln_k in ln_k_array:
                k = numpy.exp(ln_k)
                f.write("%1.10f %1.10f\n" % (k, self.linear_power(k)))
            f.close()
