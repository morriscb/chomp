import cosmology
import defaults
import halo
import hod
import mass_function
import numpy
from scipy import integrate
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import RectBivariateSpline


class HaloTrispectrum(halo.Halo):
    """
    Derived Halo class for computing the higher order power specrum of matter.
    
    Given an input cosmology, an input mass function, and an input pertubation
    theory definition (also cosmology dependent), compute the trispectrum power
    in the halo model. Note that the mass function used must have a second order
    bias method bias_2_nu. The default API for MassFunction does not contain
    this. Instead a derived class like MassFunctionSecondOrder must be used.
    
    The API for the trispectrum works differently than that of the Halo class in
    that instead of simplily inputting a length of a vector k we need a set of 
    4 k vectors to specify a trispectrum amplitude. For the cacluations we are 
    currently considering we only consider 4 vectors that form a paraleagram.
    That is T(k1, k2, k3, k4) -> T(k1, -k1, k2, -k2) == T(k1,k2). Or in terms of
    scalars we have T(k1, k2) == T(|k1|, |k2|, cos(theta)) where theta is the
    angle between vectors k1 and k2.
    
    For this class we follow the definitions as presented in Cooray & Hu 2001
    
    Attributes:
        redshift: float redshift at which to compute the halo model
        input_hod: Set to None by default. No implimation in higher order
            spectra currently.
        cosmo_single_epoch: SingleEpoch object from cosmology.py
        mass_func_second: MassFunctionSecondOrder object from mass_function.py
        perturbation: PertubationTheory object from perturbation_spectra.py
        halo_dict: dictionary of floats defining halo properties. (see 
            defaults.py for details)
        
    """
    
    def __init__(self, redshift=0.0, single_epoch_cosmo=None,
                 mass_func_second=None, perturbation=None, halo_dict=None):
        self.pert = perturbation
        halo.Halo.__init__(self, redshift, None, single_epoch_cosmo,
                      mass_func_second, halo_dict)
        self._initialized_i_0_4 = False
        self._initialized_i_1_2 = False
        self._initialized_i_1_3 = False
        self._initialized_i_2_1 = False
        self._initialized_i_2_2 = False
        
        self._initialized_tri_proj = False
        
    def set_cosmology(self, cosmo_dict, redshift=None):
        """
        Reset the internal cosmology to the values in cosmo_dict and 
        re-initialize the internal splines. Optionaly reset the internal
        redshift value.

        Args:
            cosmo_dict: dictionary of floats defining a cosmology. (see
                defaults.py for details)
            redshift: float redshift at which to compute halo model.
        """
        if redshift==None:
            redshift = self._redshift
        self.cosmo_dict = cosmo_dict
        self._redshift = redshift
        self.cosmo = cosmology.SingleEpoch(redshift, cosmo_dict)
        self.pert.set_cosmology_object(self.cosmo)
        self.delta_v = self.cosmo.delta_v()
        self.rho_bar = self.cosmo.rho_bar()
        self._h = self.cosmo._h

        self.c0 = self.halo_dict["c0"]/(1.0 + redshift)

        self.mass.set_cosmology_object(self.cosmo)

        self._calculate_n_bar()
        self._initialize_halo_splines()

        self._initialized_h_m = False
        self._initialized_h_g = False

        self._initialized_pp_mm = False
        self._initialized_pp_gm = False
        self._initialized_pp_gg = False
        
        self._initialized_i_0_4 = False
        self._initialized_i_1_2 = False
        self._initialized_i_1_3 = False
        self._initialized_i_2_1 = False
        self._initialized_i_2_2 = False
        
        self._initialized_tri_proj = False
        
    def set_redshift(self, redshift):
        """
        Reset internal variable redshift and recompute all internal splines:
        
        Args:
            redshift: float value at which to compute all cosmology dependent
                values/
        """
        self.set_cosmology(self.cosmo.cosmo_dict, redshift)
        
    def trispectrum_parallelogram(self, k1, k2, z):
        """
        Return the trispectrum of the matter power spectrum given the input
        lengths of the parallelogram sides and the cosine of the interior angle.
        
        Args:
            k1, k2: float length of wavevector
            z: float cosine of angle between k1, k2
        Returns:
            float trispectrum power
            
        TODO: (Chris)
            Force trispectrum to give well behaved results in the case where
            z=-1,1.
        """
        if not self._initialized_i_0_4:
            self._initialize_i_0_4()
        if not self._initialized_h_m:
            self._initialize_h_m()
        if not self._initialized_i_1_2:
            self._initialize_i_1_2()
        if not self._initialized_i_1_3:
            self._initialize_i_1_3()
        if not self._initialized_i_2_1:
            self._initialize_i_2_1()
        if not self._initialized_i_2_2:
            self._initialize_i_2_2()
        return (self.t_1_h(k1, k2) +
                self.t_2_h(k1, k2, z) +
                self.t_3_h(k1, k2, z) +
                self.t_4_h(k1, k2, z))
        
    def trispectrum_projected(self, k1, k2):
        if not self._initialized_tri_proj:
            self._initialize_tri_proj()
        return numpy.where(
            numpy.logical_and(
                numpy.logical_and(k1 >= self._k_min, k1 <= self._k_max),
                numpy.logical_and(k2 >= self._k_min, k2 <= self._k_max)),
            self._tri_proj_spline(numpy.log(k1), numpy.log(k2)), 0.0)[0][0]
        
    def _initialize_tri_proj(self):
        _tri_proj_array = numpy.empty((len(self._ln_k_array),
                                       len(self._ln_k_array)))
        
        for idx1, ln_k1 in enumerate(self._ln_k_array):
            for idx2, ln_k2 in enumerate(self._ln_k_array):
                k1 = numpy.exp(ln_k1)
                k2 = numpy.exp(ln_k2)
                theta_int = 2.0*integrate.romberg(
                    self._trispectrum_parallelogram_wrap, 
                    0.0 , numpy.pi,
                    args=(k1, k2), vec_func=True,
                    tol=defaults.default_precision["halo_precision"],
                    divmax=defaults.default_precision["divmax"])
                _tri_proj_array[idx1, idx2] = (
                    numpy.pi*self.t_1_h(k1, k2) + theta_int/(2.0*numpy.pi))
                if idx1%10 == 0 and idx2%10 == 0:
                    print "Running idx1, idx2:", idx1, k1, idx2, k2
                    print "\tvalue:", _tri_proj_array[idx1, idx2]
        
        self._tri_proj_spline = RectBivariateSpline(
            self._ln_k_array, self._ln_k_array, _tri_proj_array)
        
        print "Initialized::Tripectrum Projection"
        self._initialized_tri_proj = True
    
    def _trispectrum_parallelogram_wrap(self, theta, k1, k2):
        z = numpy.cos(theta)
        return (self.t_2_h(k1, k2, z) + self.t_3_h(k1, k2, z)  +
                self.t_4_h(k1, k2, z))
        
    def t_1_h(self, k1, k2):
        """
        First compoment of the trispectrum, the poisonian term of correlations
        within one halo.
        
        Args:
            k1, k2: float length of wavevector
        Retuns:
            float value of the piossonian term in the trispectrum.
        """
        return self.i_0_4_parallelogram(k1, k2)
    
    def t_2_h(self, k1, k2, z):
        """
        Trispectrum compoment describing correlations between 2
        different halos. This can either be 3 points in one halo and one in the
        other or 2 in both halos.
        
        Args:
            k1, k2: float length of wavevector
            z: cosine of the angle between the two wavevectors
        Return:
            float value of the 2 halo correlation component of the trispectrum.
        """
        
        ### convienience variables to caculate the mass integrals of the halos
        ### ahead of time
        i_1_2_k1k2 = self.i_1_2(k1, k2)
        i_1_3_k1k1k2 = self.i_1_3_parallelogram(k1, k2)
        i_1_3_k2k2k1 = self.i_1_3_parallelogram(k2, k1)
        
        ### Equation representing correlations between 3 points in one halo and 
        ### one in another halo. There are 4 terms here however there are 2
        ### pairs that are identical for a parallelogram. Hence we show only 2
        ### and double the output.
        T_31 = 2.0 * (self.cosmo.linear_power(k1) * 
                      i_1_3_k2k2k1 * self.i_1_1(k1) +
                      self.cosmo.linear_power(k2) * 
                      i_1_3_k1k1k2 * self.i_1_1(k2))
        
        ### Need to know the length of the subtraction of our two k vectors
        k1m2 = numpy.sqrt(k1*k1 + k2*k2 - 2.0*k1*k2*z)
        k1p2 = numpy.sqrt(k1*k1 + k2*k2 + 2.0*k1*k2*z)
        ### Equation for 2 sets of points in 2 distinct halos. Again there is 
        ### symetry for a parallelogram that we exploit here.
        P_k1m2 = numpy.where(k1m2 > 0.0 + 
                             defaults.default_precision["halo_precision"],
                             self.cosmo.linear_power(k1m2), 0.0)
        P_k1p2 = numpy.where(k1p2 > 0.0 + 
                             defaults.default_precision["halo_precision"],
                             self.cosmo.linear_power(k1p2), 0.0)
        T_22 =  2.0* i_1_2_k1k2 * i_1_2_k1k2 * (P_k1m2 + P_k1p2)
        
        return T_31 + T_22
    
    def t_3_h(self, k1, k2, z):
        """
        Trispectrum compoment representing correlations between 3 halos with 2
        points in 1 and 1 point in each of the others. This is the most
        complicated of the terms and would be exected to be the slowest in
        calculations
        
        Args:
            k1, k2: float length of wavevector
            z: cosine of the angle between the two wavevectors
        Return:
            float value of the 3 halo correltion compoment of the trispectrum
        """
        ### convinience variables to compute the needed mass integrals ahead of
        ### time.
        i_1_1_k1 = self._h_m(k1)
        i_1_1_k2 = self._h_m(k2)
        i_1_2_k1 = self.i_1_2(k1, k1)
        i_1_2_k2 = self.i_1_2(k2, k2)
        i_1_2_k1k2 = self.i_1_2(k1, k2)
        i_2_2_k1 = self.i_2_2(k1, k1)
        i_2_2_k2 = self.i_2_2(k2, k2)
        i_2_2_k1k2 = self.i_2_2(k1, k2)
        
        ### We are going to need the linear power spectrum several times over
        ### so we precompute it for both wavevectors
        P_k1 = self.linear_power(k1)
        P_k2 = self.linear_power(k2)
        
        ### We need to know the length of the addition and subtraction of the 
        ### two input vectors as well as the cosine of the angle between them
        lenplus = numpy.sqrt(k1 * k1 + 2.0 * k1 * k2 * z + k2 * k2)
        lenminus = numpy.sqrt(k1 * k1 - 2.0 * k1 * k2 * z + k2 * k2)
        z1plus = numpy.where(lenplus > 0.0, 
                             (k1 * k1 + k1 * k2 * z)/(k1 * lenplus), 0.0)
        z2plus = numpy.where(lenplus > 0.0,
                             (k2 * k2 + k1 * k2 * z)/(k2 * lenplus), 0.0)
        z1minus = numpy.where(lenminus > 0.0,
                              (k1 * k1 - k1 * k2 * z)/(k1 * lenminus), 0.0)
        z2minus = numpy.where(lenminus > 0.0,
                              (k2 * k2 - k1 * k2 * z)/(k2 * lenminus), 0.0)
        
        
        ### Since we are dealing only with parallelograms terms in the
        ### bispectrum with Bi(K, -K, 0) are identically zero eliminating the
        ### bispectrum components 
        perm_1 = P_k1 * P_k1 * i_2_2_k2 * i_1_1_k1 * i_1_1_k1
        perm_2 = P_k2 * P_k2 * i_2_2_k1 * i_1_1_k2 * i_1_1_k2
        ### For the remaining components there are 2 pairs of identical
        ### components for plus and minus pairings of k1, k2
        ### Originally this term is +-- in z
        bispec_plus = numpy.where(
            lenplus > defaults.default_precision["halo_precision"],
            self.pert.bispectrum_len(k1, k2, lenplus, z, -z1plus, -z2plus),
            2.0 * (self.pert.Fs2_len(k1, k2, z) * P_k1 * P_k2))
        perm_3 = (bispec_plus * i_1_2_k1k2 * i_1_1_k1 * i_1_1_k2 + 
                  P_k1 * P_k2 * i_2_2_k1k2 * i_1_1_k1 * i_1_1_k2)
        ### Originally this term is ---
        bispec_minus = numpy.where(
            lenminus > defaults.default_precision["halo_precision"],
            self.pert.bispectrum_len(k1, k2, lenminus, -z, -z1minus, -z2minus),
            2.0 * (self.pert.Fs2_len(k1, k2, -z) * P_k1 * P_k2))
        perm_4 = (self.pert.bispectrum_len(k1, k2, lenminus,
                                           -z, -z1minus, -z2minus) *
                  i_1_2_k1k2 * i_1_1_k1 * i_1_1_k2 + 
                  P_k1 * P_k2 * i_2_2_k1k2 * i_1_1_k1 * i_1_1_k2)
        return perm_1 + perm_2 + 2.0 * (perm_3 + perm_4)
        
        ### OLD VERSION:: KEEPING IT AROUND FOR NOW
       
        # bispect_k1k1k2k2 = 2.0 * (
        #     self.pert.Fs2_len(k1, k1, 1.0)) * P_k1 * P_k1
        # bispect_k2k2k1k1 = 2.0 * (
        #     self.pert.Fs2_len(k2, k2, 1.0)) * P_k2 * P_k2
        
        # perm_1 = (bispect_k1k1k2k2 * i_1_2_k2 * i_1_1_k1 * i_1_1_k1 +
        #           P_k1 * P_k1 * i_2_2_k2 * i_1_1_k1 * i_1_1_k1)
        # perm_2 = (bispect_k2k2k1k1 * i_1_2_k1 * i_1_1_k2 * i_1_1_k2 +
        #           P_k2 * P_k2 * i_2_2_k1 * i_1_1_k2 * i_1_1_k2)
        
        ### The next two permutations (taking care of the remaining 4 in
        ### Cooray & Hu.) These compoments are either dependent on the vector
        ### k1 - k2 (perm_3) or k1 + k2 (perm_4). Since we are using symetric
        ### version of all functions, we just need to compute these and double 
        ### them in the output.
        # perm_3 = (self.pert.bispectrum_len(k1, k2, lenminus,
        #                                    z, z1minus, z2minus) * 
        #           i_1_2_k1k2 * i_1_1_k1 * i_1_1_k2 + 
        #           P_k1 * P_k2 * i_2_2_k1k2 * i_1_1_k1 * i_1_1_k2)
        # perm_4 = (self.pert.bispectrum_len(k1, k2, lenplus,
        #                                    z, z1plus, z2plus) * 
        #           i_1_2_k1k2 * i_1_1_k1 * i_1_1_k2 + 
        #           P_k1 * P_k2 * i_2_2_k1k2 * i_1_1_k1 * i_1_1_k2)
        # return perm_1 + perm_2 + 2.0 * perm_3 + 2.0 * perm_4
    
    def t_4_h(self, k1, k2, z):
        """
        Trispectrum term representing correlations between 4 distict halos. 
        
        Args:
            k1, k2: length of the wavevector
            z: cosine of the angle between wavevectors k1 and k2
        Returns:
            float trispectrum correlation between 4 distict halos.
        """
        ### Precompute our mass integrals for later use
        i_1_1_k1 = self._h_m(k1)
        i_1_1_k2 = self._h_m(k2)
        i_2_1_k1 = self.i_2_1(k1)
        i_2_1_k2 = self.i_2_1(k2)
        
        ### also with the linear power spectra
        P_k1 = self.linear_power(k1)
        P_k2 = self.linear_power(k2)
        
        ### Multilpy and output (eq 23 in Cooray & Hu)
        return i_1_1_k1 * i_1_1_k1 * i_1_1_k2 * i_1_1_k2 * (
           self.pert.trispectrum_parallelogram(k1, k2, z) + 
           2.0 * (i_2_1_k1 * P_k1 * P_k2 * P_k2 +
                  i_2_1_k2 * P_k2 * P_k1 * P_k1))      
         
    def i_0_4(self, k1, k2, k3, k4):
        """
        Integral over mass for 4 points contained within a single halo. Since
        all halos considered are spherically symetric the input values are all
        scalars
        Args:
            k1,...k4: float length of wavevector
        Returns:
            float Integral over all halo masses for 4 points in a halo
        """
        return integrate.romberg(
            self._i_0_4_integrand, numpy.log(self.mass.nu_min),
            numpy.log(self.mass.nu_max), vec_func=True,
            tol=defaults.default_precision["halo_precision"],
            divmax=defaults.default_precision["divmax"],
            args=(k1, k2, k3, k4))/(self.rho_bar*self.rho_bar*self.rho_bar)
            
    def i_0_4_parallelogram(self, k1, k2):
        k1 = numpy.where(k1 < self._k_min, self._k_min, k1)
        k2 = numpy.where(k2 < self._k_min, self._k_min, k2)
        return numpy.where(
            numpy.logical_and(k1 <= self._k_max, k2 <= self._k_max),
            self._i_0_4_spline(numpy.log(k1), numpy.log(k2))[0,0], 0.0)
        
    def _initialize_i_0_4(self):
        _i_0_4_array = numpy.empty((len(self._ln_k_array),
                                    len(self._ln_k_array)))
        
        for idx1, ln_k1 in enumerate(self._ln_k_array):
            for idx2, ln_k2 in enumerate(self._ln_k_array):
                k1 = numpy.exp(ln_k1)
                k2 = numpy.exp(ln_k2)
                _i_0_4_array[idx1, idx2] = self.i_0_4(k1, k1, k2, k2)
        
        self._i_0_4_spline = RectBivariateSpline(
            self._ln_k_array, self._ln_k_array, _i_0_4_array)
        
        print "Initialized::I_0_4_parallelogram"
        
        self._initialized_i_0_4 = True
        
    def _i_0_4_integrand(self, ln_nu, k1, k2, k3, k4):
        nu = numpy.exp(ln_nu)
        mass = self.mass.mass(nu)
        y1 = self.y(numpy.log(k1), mass)
        y2 = self.y(numpy.log(k2), mass)
        y3 = self.y(numpy.log(k3), mass)
        y4 = self.y(numpy.log(k4), mass)

        return nu*self.mass.f_nu(nu)*y1*y2*y3*y4*mass*mass*mass
    
    def i_1_1(self, k):
        """
        Integral over mass for 1 point contained within a single halo and the
        halo average halo bias as a function of nu. This integral uses the 
        spline from the Halo class as it is identical to the term _h_m. Since 
        all halos considered are spherically symetric the input values are all
        scalars
        
        Args:
            k: float length of wavevector
        Returns:
            float Integral over all halo masses for 4 points in a halo
        """
        return self._h_m(k)
        
    def i_1_2(self, k1, k2):
        """
        Integral over the mass and bias for 2 points within a halo.
        
        Args:
            k1, k2: float length of wavevector
        Returns:
            float Integral over all halo masses for 4 points in a halo
        """
        k1 = numpy.where(k1 < self._k_min, self._k_min, k1)
        k2 = numpy.where(k2 < self._k_min, self._k_min, k2)
        return numpy.where(
            numpy.logical_and(k1 <= self._k_max, k2 <= self._k_max),
            self._i_1_2_spline(numpy.log(k1), numpy.log(k2))[0, 0], 0.0)
        
    def _initialize_i_1_2(self):
        _i_1_2_array = numpy.empty((len(self._ln_k_array),
                                    len(self._ln_k_array)))
        
        for idx1, ln_k1 in enumerate(self._ln_k_array):
            for idx2, ln_k2 in enumerate(self._ln_k_array):
                _i_1_2_array[idx1, idx2] = integrate.romberg(
                    self._i_1_2_integrand, numpy.log(self.mass.nu_min),
                    numpy.log(self.mass.nu_max), vec_func=True,
                    args=(numpy.exp(ln_k1), numpy.exp(ln_k2)),
                    tol=defaults.default_precision["halo_precision"],
                    divmax=defaults.default_precision["divmax"])/(self.rho_bar)
                    
        self._i_1_2_spline = RectBivariateSpline(self._ln_k_array,
                                                 self._ln_k_array,
                                                 _i_1_2_array)
        print "Initialized::I_1_2"
        self._initialized_i_1_2 = True
    
    def _i_1_2_integrand(self, ln_nu, k1, k2):
        nu = numpy.exp(ln_nu)
        mass = self.mass.mass(nu)
        y1 = self.y(numpy.log(k1), mass)
        y2 = self.y(numpy.log(k2), mass)
        
        return nu*self.mass.f_nu(nu)*self.mass.bias_nu(nu)*y1*y2*mass
        
    def i_1_3(self, k1, k2, k3):
        """
        Integral over the mass and bias for 3 points within a halo.
        
        Args:
            k1,...k3: float length of wavevector
        Returns:
            float Integral over all halo masses for 4 points in a halo
        """
        return integrate.romberg(
            self._i_1_3_integrand, numpy.log(self.mass.nu_min),
            numpy.log(self.mass.nu_max), vec_func=True,
            tol=defaults.default_precision["halo_precision"],
            divmax=defaults.default_precision["divmax"],
            args=(k1, k2, k3))/(self.rho_bar*self.rho_bar)
            
    def i_1_3_parallelogram(self, k1, k2):
        k1 = numpy.where(k1 < self._k_min, self._k_min, k1)
        k2 = numpy.where(k2 < self._k_min, self._k_min, k2)
        return numpy.where(
            numpy.logical_and(k1 <= self._k_max, k2 <= self._k_max),
            self._i_1_3_spline(numpy.log(k1), numpy.log(k2))[0,0], 0.0)
        
    def _initialize_i_1_3(self):
        _i_1_3_array = numpy.empty((len(self._ln_k_array),
                                    len(self._ln_k_array)))
        
        for idx1, ln_k1 in enumerate(self._ln_k_array):
            for idx2, ln_k2 in enumerate(self._ln_k_array):
                k1 = numpy.exp(ln_k1)
                k2 = numpy.exp(ln_k2)
                _i_1_3_array[idx1, idx2] = self.i_1_3(k1, k1, k2)
        
        self._i_1_3_spline = RectBivariateSpline(
            self._ln_k_array, self._ln_k_array, _i_1_3_array)
        
        print "Initialized::I_1_3_parallelogram"
        
        self._initialized_i_1_3 = True
    
    def _i_1_3_integrand(self, ln_nu, k1, k2, k3):
        nu = numpy.exp(ln_nu)
        mass = self.mass.mass(nu)
        y1 = self.y(numpy.log(k1), mass)
        y2 = self.y(numpy.log(k2), mass)
        y3 = self.y(numpy.log(k3), mass)
        
        return nu*self.mass.f_nu(nu)*self.mass.bias_nu(nu)*y1*y2*y3*mass*mass
    
    def i_2_1(self, k):
        """
        Integral over the mass and the second order bias for 2 points 
        within a halo.
        
        Args:
            k1, k2: float length of wavevector
        Returns:
            float Integral over all halo masses for 4 points in a halo
        """
        k = numpy.where(k < self._k_min, self._k_min, k)
        return numpy.where(k <= self._k_max,
                           self._i_2_1_spline(numpy.log(k)), 0.0)
    
    def _initialize_i_2_1(self):
        _i_2_1_array = numpy.empty(self._ln_k_array.shape)
        
        for idx, ln_k in enumerate(self._ln_k_array):
            _i_2_1_array[idx] = integrate.romberg(
                self._i_2_1_integrand, numpy.log(self.mass.nu_min),
                numpy.log(self.mass.nu_max), vec_func=True,
                tol=defaults.default_precision["halo_precision"],
                divmax=defaults.default_precision["divmax"],
                args=(numpy.exp(ln_k),))
        
        self._i_2_1_spline = InterpolatedUnivariateSpline(
            self._ln_k_array, _i_2_1_array)
        print "Initialized::I_2_1"
        self._initialized_i_2_1 = True
    
    def _i_2_1_integrand(self, ln_nu, k):
        nu = numpy.exp(ln_nu)
        mass = self.mass.mass(nu)
        y = self.y(numpy.log(k), mass)

        return nu*self.mass.f_nu(nu)*self.mass.bias_2_nu(nu)*y
                
    def i_2_2(self, k1, k2):
        """
        Integral over the mass and the second order bias for 2 points 
        within a halo.
        
        Args:
            k1, k2: float length of wavevector
        Returns:
            float Integral over all halo masses for 4 points in a halo
        """
        k1 = numpy.where(k1 < self._k_min, self._k_min, k1)
        k2 = numpy.where(k2 < self._k_min, self._k_min, k2)
        return numpy.where(
            numpy.logical_and(k1 <= self._k_max, k2 <= self._k_max),
            self._i_2_2_spline(numpy.log(k1), numpy.log(k2))[0, 0], 0.0)
        
    def _initialize_i_2_2(self):
        _i_2_2_array = numpy.empty((len(self._ln_k_array),
                              len(self._ln_k_array)))
        
        for idx1, ln_k1 in enumerate(self._ln_k_array):
            for idx2, ln_k2 in enumerate(self._ln_k_array):
                _i_2_2_array[idx1, idx2] = integrate.romberg(
                    self._i_2_2_integrand, numpy.log(self.mass.nu_min),
                    numpy.log(self.mass.nu_max), vec_func=True,
                    args=(numpy.exp(ln_k1), numpy.exp(ln_k2)),
                    tol=defaults.default_precision["halo_precision"],
                    divmax=defaults.default_precision["divmax"])/(self.rho_bar)
                    
        self._i_2_2_spline = RectBivariateSpline(self._ln_k_array,
                                                 self._ln_k_array,
                                                 _i_2_2_array)
        print "Initialized::I_2_2"
        self._initialized_i_2_2 = True
    
    def _i_2_2_integrand(self, ln_nu, k1, k2):
        nu = numpy.exp(ln_nu)
        mass = self.mass.mass(nu)
        y1 = self.y(numpy.log(k1), mass)
        y2 = self.y(numpy.log(k2), mass)
        
        return nu*self.mass.f_nu(nu)*self.mass.bias_2_nu(nu)*y1*y2*mass
    
    