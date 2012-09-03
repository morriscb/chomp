#!/usr/bin/env python
# encoding: utf-8
import cosmology
import numpy

"""
Class for perturbation theory expressions for the matter 3 and 4-point 
functions.

References

[GGRW]  M. H. Goroff, B. Grinstein, S.-J. Rey, and M. B. Wise,
“Coupling of modes of cosmological mass density fluctuations,”
ApJ, vol. 311, pp. 6–14, Dec. 1986.

[SZH]  R. Scoccimarro, M. Zaldarriaga, and L. Hui,
“Power Spectrum Correlations Induced by Nonlinear Clustering,”
ApJ, vol. 527, no. 1, pp. 1–15, Dec. 1999.

[MW]  A. Meiksin and M. White,
“The growth of correlations in the matter power spectrum,”
MNRAS, vol. 308, no. 4, pp. 1179–1184, Oct. 1999.

[CH]  A. Cooray and W. Hu,
“Power Spectrum Covariance of Weak Gravitational Lensing,”
ApJ, vol. 554, no. 1, pp. 56–66, Jun. 2001.

[BCGS] F. Bernardeau, S. Colombi, E. Gaztanaga, and R. Scoccimarro,
“Large-scale structure of the Universe and cosmological perturbation theory,”
Physics Reports, vol. 367, no. 1, pp. 1–248, Sep. 2002.
"""

__author__ = ("Michael Schneider <mischnei@gmail.com>")


def alpha_BCGS(k1, k2):
    """
    Eq. 39 from Ref BCGS
    """
    k1sq = numpy.vdot(k1, k1)
    if k1sq == 0.0:
        return 0.0
    else:
        return numpy.vdot(k1 + k2, k1) / k1sq


def gamma_BCGS(k1, k2):
    """
    Eq. 68 from Ref BCGS
    """
    k1a = numpy.vdot(k1, k1)
    k2a = numpy.vdot(k2, k2)
    if k1a * k2a == 0.:
        return 0.0
    else:
        return 1 - (numpy.vdot(k1, k2)) ** 2 / (k1a * k2a)


class PerturbationTheory(object):
    """
    Bispectrum and trispectrum from perturbation theory
    """
    def __init__(self, redshift=0.0, cosmo_single_epoch=None, **kws):
        """
        Initialization method
        """
        self._redshift = redshift
        if cosmo_single_epoch is None:
            cosmo_single_epoch = cosmology.SingleEpoch(redshift)
        elif cosmo_single_epoch._redshift != redshift:
            cosmo_single_epoch.set_redshift(redshift)
        self.cosmo = cosmo_single_epoch
        return None
    
    def set_cosmology(self, cosmo_dict, redshift=None):
        if redshift is None:
            redshift = self._redshift
        self.cosmo.set_cosmology(cosmo_dict, redshift)
        self._redshift = redshift
    
    def set_cosmology_object(self, cosmo_single_epoch):
        self.cosmo = cosmo_single_epoch
        self._redshift = self.cosmo._redshift
    
    def set_redshift(self, redshift):
        self._redshift = redshift
        self.cosmo.set_redshif(self._redshift)

    def Fs2(self, k1, k2):
        """
        Eq. A.2 in GGRW
        Eq. 45 in BCGS

        - k arguments should be length-3 numpy arrays
        """
        d = numpy.dot(k1, k2)
        k1a = numpy.sqrt(numpy.dot(k1, k1))
        k2a = numpy.sqrt(numpy.dot(k2, k2))
        if k1a < 1.e-8 or k2a < 1.e-8:
            res = 5. / 7.
        else:
            rat = d / (k1a * k2a)
            res = ((5. / 7. + (rat / 2.) * (k1a / k2a + k2a / k1a) + 
                   (2. / 7.) * rat * rat))
        return res
    
    def Fs2_len(self, k1, k2, z):
        res = numpy.where(numpy.logical_or(k1 < 1.e-8, k2 < 1.e-8), 5.0/7.0,
                          ((5. / 7. + (z / 2.) * (k1 / k2 + k2 / k1) + 
                            (2. / 7.) * z * z)))
        return res

    def Fs2_parallelogram(self, k1, mu):
        """
        Eq. A.2 in GGRW for the special case Fs2(k1, -k1).

        Args:
            k1: wavenumber in h/Mpc
            mu: cosine of the angle between the wavevectors k1 and k2

        Returns:
            Fs2 evaluated for the special arguments k1, -k1
        """
        return (5. / 7.) + mu + (2. / 7.) * mu * mu

    def Fs3(self, k1, k2, k3):
        """
        Eq. A.3 in Goroff, Grinstein, Rey, Wise (1986)

        - k arguments should be length-3 numpy arrays
        """
        k1a = (numpy.vdot(k1, k1))
        k2a = (numpy.vdot(k2, k2))
        k3a = (numpy.vdot(k3, k3))
        k12a = (numpy.vdot(k1 + k2, k1 + k2))
        k23a = (numpy.vdot(k2 + k3, k2 + k3))
        k123a = numpy.vdot(k1 + k2 + k3, k1 + k2 + k3)

        # k123 = k1a ** 2 * k2a ** 2 * k3a ** 2

        b1 = ((1. / 21.) * numpy.vdot(k1, k2) * k12a +
              (1. / 14.) * k2a * numpy.vdot(k1, k1 + k2))
        b2 = (7. * k3a * numpy.vdot(k1 + k2, k1 + k2 + k3) +
              numpy.vdot(k3, k1 + k2) * k123a)
        b3 = ((1. / 21.) * numpy.vdot(k2, k3) * k23a +
              (1. / 14.) * k3a * numpy.vdot(k2, k2 + k3))
        b4 = (numpy.vdot(k2, k3) * k23a + 5. * k3a * numpy.vdot(k2, k2 + k3))

        if k12a < 1.0e-8:
            c1 = 0.0
        else:
            c1 = 1. / (3. * k1a * k2a * k3a * k12a)
        if k23a < 1.0e-8:
            c3 = 0.0
        else:
            c3 = ((numpy.vdot(k1, k2 + k3) * k123a) / 
                  (3. * k1a * k2a * k3a * k23a))
        c4 = numpy.vdot(k1, k1 + k2 + k3) / (18. * k1a * k2a * k3a)
        return c1 * b1 * b2 + c3 * b3 + c4 * b4

    def Fs3_parallelogram(self, k1, k2, mu):
        """
        Eq. A.3 in GGRW for the special case F3(k1, -k1, k2)

        Args:
            k1: wavenumber in h/Mpc
            k2: wavenumber in h/Mpc
            mu: cosine of the angle between the wavevectors k1 and k2

        Returns:
            Fs3 evaluated for the special arguments k1, -k1, k2
        """
        x = k2 / k1
        y = x * mu - 1.0
        z = 1.0 + x * x - 2.0 * x * mu
        term1 = (1. / 21.) * x * y * ((mu / 3.) + 0.5 * x * y / z)
        term2 = -(mu / 18.) * (mu * z + 5 * x * y)
        return term1 + term2

    def F3(self, k1, k2, k3):
        """
        Eq. 73 in BCGS

        - k arguments should be length-3 numpy arrays
        """
        # nu2, nu3, and lambda3 are time-dependent, see eq. 67 of BCGS.
        nu2 = 34. / 21.  # Eq. 52 in BCGS
        nu3 = 682. / 189.  # Eq. 52 in BCGS
        lambda3 = 9. / 10.  # Eq. 67 in BCGS
        k12 = k1 + k2
        g312 = gamma_BCGS(k3, k12)
        g12 = gamma_BCGS(k1, k2)
        R11 = ((0.5 * alpha_BCGS(k3, k12) + 0.5 * alpha_BCGS(k12, k3) - 
                (1. / 3.) * g312) * alpha_BCGS(k1, k2))
        R12 = ((-1.5 * alpha_BCGS(k12, k3) - (4. / 3.) * alpha_BCGS(k3, k12) + 
                2.5 * g312) * g12)
        R2 = 0.75 * (alpha_BCGS(k3, k12) + alpha_BCGS(k12, k3) - 
                      3. * g312) * g12
        R3 = (3. / 8.) * g312 * g12
        R4 = ((2. / 3.) * g312 * alpha_BCGS(k1, k2) - ((1. / 3.) * 
            alpha_BCGS(k3, k12) + 0.5 * g312) * g12)
        return (R11 + R12) + nu2 * R2 + nu3 * R3 + lambda3 * R4

    def Fs3_BCGS(self, k1, k2, k3, F3=None):
        if F3 is None:
            F3 = self.F3
        return (F3(k1, k2, k3) + F3(k3, k1, k2) + F3(k2, k3, k1) +
            F3(k2, k1, k3) + F3(k3, k2, k1) + F3(k1, k3, k2)) / 6.

    def bispectrum(self, k1, k2, k3):
        """
        perturbation theory bispectrum
        (eq. 22 in Cooray and Hu (2000))

        - k arguments should be length-3 numpy arrays
        """
        k1a = numpy.sqrt(numpy.vdot(k1, k1))
        k2a = numpy.sqrt(numpy.vdot(k2, k2))
        k3a = numpy.sqrt(numpy.vdot(k3, k3))

        p1 = self.cosmo.linear_power(k1a)
        p2 = self.cosmo.linear_power(k2a)
        p3 = self.cosmo.linear_power(k3a)

        res = 2. * (self.Fs2(k1, k2) * p1 * p2 +
                    self.Fs2(k1, k3) * p1 * p3 +
                    self.Fs2(k2, k3) * p2 * p3)
        return res
    
    def bispectrum_len(self, k1, k2, k3, z12, z13, z23):
        p1 = self.cosmo.linear_power(k1)
        p2 = self.cosmo.linear_power(k2)
        p3 = self.cosmo.linear_power(k3)

        res = 2. * (self.Fs2_len(k1, k2, z12) * p1 * p2 +
                    self.Fs2_len(k1, k3, z13) * p1 * p3 +
                    self.Fs2_len(k2, k3, z23) * p2 * p3)
        return res
        

    def trispectrum(self, k1, k2, k3, k4):
        """
        Perturbation theory trispectrum
        (eq. 24 in CH or eq. 6 in SZH)

        Failed tests - needs debugging. [July 17, 2012]

        Args:
            k1: wavevector, length-3 numpy array
            k2: wavevector, length-3 numpy array
            k3: wavevector, length-3 numpy array
            k4: wavevector, length-3 numpy array
        Returns:
            Perturbation theory trispectrum of the 3D mass density
        """
        p1 = self.cosmo.linear_power(numpy.sqrt(numpy.dot(k1, k1)))
        p2 = self.cosmo.linear_power(numpy.sqrt(numpy.dot(k2, k2)))
        p3 = self.cosmo.linear_power(numpy.sqrt(numpy.dot(k3, k3)))
        p4 = self.cosmo.linear_power(numpy.sqrt(numpy.dot(k4, k4)))
        p12 = self.cosmo.linear_power(numpy.sqrt(numpy.dot(k1 + k2, k1 + k2)))
        p13 = self.cosmo.linear_power(numpy.sqrt(numpy.dot(k1 + k3, k1 + k3)))
        p14 = self.cosmo.linear_power(numpy.sqrt(numpy.dot(k1 + k4, k1 + k4)))
        p23 = self.cosmo.linear_power(numpy.sqrt(numpy.dot(k2 + k3, k2 + k3)))
        p24 = self.cosmo.linear_power(numpy.sqrt(numpy.dot(k2 + k4, k2 + k4)))
        p34 = self.cosmo.linear_power(numpy.sqrt(numpy.dot(k3 + k4, k3 + k4)))

        if numpy.isnan(p12):
            p12 = 0.0
        if numpy.isnan(p34):
            p34 = 0.0

        b1 = (self.Fs2(k1 + k2, -k1) * self.Fs2(k1 + k2, k3) * p1 * p12 * p3 +
              self.Fs2(k2 + k3, -k2) * self.Fs2(k2 + k3, k1) * p2 * p23 * p1 +
              self.Fs2(k3 + k1, -k3) * self.Fs2(k3 + k1, k2) * p3 * p13 * p2 +
              self.Fs2(k1 + k2, -k1) * self.Fs2(k1 + k2, k4) * p1 * p12 * p4 +
              self.Fs2(k2 + k4, -k2) * self.Fs2(k2 + k4, k1) * p2 * p24 * p1 +
              self.Fs2(k4 + k1, -k4) * self.Fs2(k4 + k1, k2) * p4 * p14 * p2 +
              self.Fs2(k1 + k3, -k1) * self.Fs2(k1 + k3, k4) * p1 * p13 * p4 +
              self.Fs2(k3 + k4, -k3) * self.Fs2(k3 + k4, k1) * p3 * p34 * p1 +
              self.Fs2(k4 + k1, -k4) * self.Fs2(k4 + k1, k3) * p4 * p14 * p3 +
              self.Fs2(k2 + k3, -k2) * self.Fs2(k2 + k3, k4) * p2 * p23 * p4 +
              self.Fs2(k3 + k4, -k3) * self.Fs2(k3 + k4, k2) * p3 * p34 * p2 +
              self.Fs2(k4 + k2, -k4) * self.Fs2(k4 + k2, k3) * p4 * p24 * p3)
        b2 = (self.Fs3(k1, k2, k3) * p1 * p2 * p3 +
              self.Fs3(k1, k2, k4) * p1 * p2 * p4 +
              self.Fs3(k1, k3, k4) * p1 * p3 * p4 +
              self.Fs3(k2, k3, k4) * p2 * p3 * p4)
        # print 6. * b2, 4. * b1
        res = 4. * b1 + 6. * b2
        return res

    def trispectrum_parallelogram(self, k1, k2, mu):
        """
        Parallelogram configuration of the trispectrum that
        contributes to the matter power spectrum covariance.

        Eq. 7 in SZH.

        Args:
            k1: wavenumber in h/Mpc
            k2: wavenumber in h/Mpc
            mu: cosine of the angle between wavevectors k1 and k2
        Returns:
            Perturbation theory trispectrum of the 3D mass density.
        See also:
            pertTheory.trispectrum(k1, k2, k3, k4)
        """
        x = k2 / k1
        z = 1 + x ** 2 - 2 * x * mu
        p1 = self.cosmo.linear_power(k1)
        p2 = self.cosmo.linear_power(k2)
        k1mk2 = k1 * numpy.sqrt(z)
        p12 = self.cosmo.linear_power(k1 * numpy.sqrt(z)) # k1 - k2
        F21 = self.Fs2_parallelogram(k1 - k2, k2)
        F22 = self.Fs2_parallelogram(k2 - k1, k1)
        a1 = 12. * self.Fs3_parallelogram(k1, -k1, k2) * (p1 ** 2) * p2
        a2 = 8. * F21 ** 2 * p12 * p2 ** 2
        a3 = 16. * F21 * F22 * p1 * p2 * p12

        b1 = 12. * self.Fs3_parallelogram(k2, -k2, k1) * (p2 ** 2) * p1
        b2 = 8. * F22 ** 2 * p12 * p1 ** 2

        res = a1 + b1 + a2 + b2 + 2. * a3
        # print a1 + b1, a2 + b2 + 2. * a3
        return res