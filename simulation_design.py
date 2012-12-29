"""
simulation_design.py

Created by Michael Schneider on 2012-10-12 modified by Chris Morrison 2012-11-4
"""

import defaults
import numpy
import pandas

__author__ = ("Michael Schneider, Chris Morrison")

default_parameter_dict = {"cosmo_dict":defaults.default_cosmo_dict,
                          "halo_dict" : defaults.default_halo_dict,
                          "hod_dict"  :  defaults.default_hod_dict}

def random_lhs(n, k):
    """
    Random Latin Hypercube Sample (non-optimized).

    Algorithm copied from randomLHS function in the R package, 'lhs'
    Rob Carnell (2012). lhs: Latin Hypercube Samples. R package version 0.10.
      http://CRAN.R-project.org/package=lhs

    Args:
        n: The number of simulation design points
        k: The number of variables
    """
    P = numpy.zeros((n, k), dtype='float64')
    for i in xrange(k):
        P[:, i] = numpy.random.permutation(range(n))
    P = P + numpy.random.uniform(size=(n, k))
    return P / n


class SimulationDesign(object):
    """
    Wrapper object for computing design points in parameter space given an input
    object, method, and parameter range. Curently only varies only parameters
    in cosmo_dict.
    
    Attributes:
        input_chomp_object: an input python object from anywhere in the chomp
            code base
        method_name: string name of method to compute. Must be member of
            input_chomp_object
        params: dictionary of parameter names and ranges over which to compute
            design points. Ranges should be a list of ["mid", "low", high"].
            Parameter names can be found in defaults.py
        n_design: int number of design points at which to compute the cosmology
        independent var: array like or None type. Some methods in chomp require
           a range over values over which to compute the method (i.e. halo power
           spectra). Input this if the method require it, else don't touch it.
        default_param_dict: dictionary of dictionaries defining the parameters
           for cosmology, halo properties, and HOD. Defaults to the default.py
           dictionaries {"cosmo":   default_cosmo_dict,
                         "halo_dict":default_halo_dict,
                         "hod_dict":  default_hod_dict}
    """
    
    def __init__(self, input_chomp_object, method_name, params, n_design=100,
                 independent_var=None, default_param_dict=None):
        self._input_object = input_chomp_object
        self._method = method_name
        self.params = pandas.DataFrame(params, index=['center', 'min', 'max'])
        self.n_design = n_design
        self._ind_var = independent_var
        self._initialized_design = False
        if default_param_dict is None:
            default_param_dict = default_parameter_dict
        self._default_param_dict = default_param_dict
        
        self._vary_cosmology = False
        self._vary_halo = False
        self._vary_hod = False  
        self._param_types = []  
        for key in self.params.keys():
            try:
                default_param_dict["cosmo_dict"][key]
                self._param_types.append("cosmo_dict")
                self._vary_cosmology = True
                continue
            except KeyError:
                pass
            try:
                default_param_dict["halo_dict"][key]
                self._param_types.append("halo_dict")
                self._vary_halo = True
                continue
            except KeyError:
                pass
            try:
                default_param_dict["hod_dict"][key]
                self._param_types.append("hod_dict")
                self._vary_hod = True
                continue
            except KeyError:
                pass
    
    def _init_design_points(self):
        """
        Internal function for setting up the Latin Hypercube Sampling
        in the paramerter space specified.
        """
        self.params = self.params.append(pandas.DataFrame((self.params.xs('max') -
                                                           self.params.xs('min')),
                                                          columns=['diff']).transpose())
        print self.params
        points = pandas.DataFrame(random_lhs(self.n_design,
                                             self.params.shape[1]),
                                  columns=self.params.columns)
        self.lhs = points
        self.points = points * self.params.xs('diff') + self.params.xs('min')
        self._initialized_design = True

    def _run_des_point(self, point):
        """
        Compute the simulation prediction for a point in the input parameter space.

        Args:
            point: a pandas Series object declaring input parameters for a CHOMP model.

        Returns:
            The output of the CHOMP model, flattened to a 1-d array.
        """
        if self._vary_cosmology:
            self.set_cosmology(self._default_param_dict['cosmo_dict'], point)
        if self._vary_halo:
            self.set_halo(point)
        if self._vary_hod:
            self.set_hod(self._default_param_dict['hod_dict'], point)
        values = 0
        if self._ind_var is None:
            values = self._input_object.__getattribute__(self._method)()
        else:
            values = self._input_object.__getattribute__(self._method)(
                              self._ind_var)
        return values.flatten()
    
    def run_design(self):
        """
        Function for running the design and initializing the Latin Hypercube
        Sampling if nessasary. Selectivly sets the which sub parameters 
        (cosmology, halo, hod) will be set.
        
        Returns:
           values_frame: a pandas DataFrame with columns defining the parameters
               and object return values.
        """
        if not self._initialized_design:
            self._init_design_points()
        self.design_values = self.points.transpose().apply(self._run_des_point)
        return self.design_values
    
    def set_cosmology(self, cosmo_dict=None, values=None):
        """
        Template setter for the cosmology dependent object. Currently defaults
        to set every varaible independently. Users are encouraged to declare
        inherited classes of SimulationDesign to declare designs with
        interdependent variables (such as flat cosmologies)
        
        Args:
            values: a dictionary or pandas Series object declaring paramerters
                to vary as keys with the requested values.
        """
        if cosmo_dict is None:
            cosmo_dict = self._default_param_dict['cosmo_dict']
        for key in self.params.keys():
            try:
                cosmo_dict[key] = values[key]
            except KeyError:
                continue
        self._input_object.set_cosmology(cosmo_dict)
    
    def set_halo(self, halo_dict, values):
        """
        Template setter for the halo dependent object. Currently defaults
        to set every varaible independently. Users are encouraged to declare
        inherited classes of SimulationDesign to declare designs with
        interdependent variables (such as flat cosmologies)
        
        Args:
            values: a dictionary or pandas Series object declaring paramerters
                to vary as keys with the requested values.
        """
        for key in self.params.keys():
            try:
                self._default_param_dict['halo_dict'][key] = values[key]
            except KeyError:
                continue
        self._input_object.set_halo(halo_dict)
    
    def set_hod(self, hod_dict, values):
        """
        Template setter for the HOD dependent object. Currently defaults
        to set every varaible independently. Users are encouraged to declare
        inherited classes of SimulationDesign to declare designs with
        interdependent variables (such as flat cosmologies)
        
        Args:
            values: a dictionary or pandas Series object declaring paramerters
                to vary as keys with the requested values.
        """
        for key in self.params.keys():
            try:
                self._default_param_dict['hod_dict'][key] = values[key]
            except KeyError:
                continue
        self._input_object.set_hod(hod_dict)
        
    def write(self, output_name):
        """
        Write out value_frame pandas DataFrame to requested file.
        
        Args:
            output_name: string defining the name of the file to write to.
        """
        self.values_frame.to_csv(output_name, index=False, sep=',')


class SimulationDesignFlatUniverse(SimulationDesign):
    
    def __init__(self, input_chomp_object, method_name, params, n_design=100,
                 independent_var=None, default_param_dict=None):
        SimulationDesign.__init__(self, input_chomp_object, method_name, params,
                                  n_design, independent_var, default_param_dict)
    
    def set_cosmology(self, cosmo_dict, values):
        for idx, key in enumerate(self.params.keys()):
            try:
                cosmo_dict[key] = values[idx]
            except KeyError:
                continue
        cosmo_dict['omega_l0'] = (1.0 - cosmo_dict['omega_m0'] -
                                  cosmo_dict['omega_r0'])
        self._input_object.set_cosmology(cosmo_dict)


class SimulationDesignHODWakeAssumptions(SimulationDesign):
    
    def __init__(self, input_chomp_object, method_name, params, n_design=100,
                 independent_var=None, default_param_dict=None):
        SimulationDesign.__init__(self, input_chomp_object, method_name, params,
                                  n_design, independent_var, default_param_dict)
    
    def set_cosmology(self, cosmo_dict, values):
        for idx, key in enumerate(params.keys()):
            try:
                cosmo_dict[key] = values[idx]
            except KeyError:
                continue
        cosmo_dict['omega_l0'] = (1.0 - cosmo_dict['omega_m0'] -
                                  cosmo_dict['omega_r0'])
        self._input_object.set_cosmology(cosmo_dict)
        
    def set_hod(self, hod_dict, values):
        for idx, key in enumerate(params.keys()):
            try:
                hod_dict[key] = values[idx]
            except KeyError:
                continue
        hod_dict['log_M_0'] = hod_dict['log_M_min']
        self._input_object.set_hod(cosmo_dict)

