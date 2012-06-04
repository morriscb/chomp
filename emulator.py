import correlation
import cosmology
import defaults
import halo
import lhs

class Emulator(object):
    """
    Template class for interpolating over any of the high level outputs from
    the halo model as a function of cosmology and HOD parameters.
    """

    def __init__(self,space_dict):
        pass

    def _interpolate(self):
        pass

    def _pca_decomp(self):
        pass

    def function_return(self, value_dict):
        pass

class PowerSpectrumEmulator(Emulator):
    pass

class CorrelationEmulator(Emulator):
    pass
