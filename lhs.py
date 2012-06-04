import numpy

class InvalidSampleSizeError(Exception):
    pass


class LHS(object):
    """
    Generates samples from a Latin Hypercube given the imput parameters.
 
    Atributes:
        n_dim: integer of the number of dimentions the Latin Hypercube spans
        limits: a list of lists containing the ranges for which each
            dimension the hypercube spans. Default None returns the raw 
            indieces of the leaving the user to determain the translation to
            'physical' quantities.
        n_samples: number of samples to draw from the hypercube. This is 
            also defines the segmenting of each axis. The axis segmention is
            inclusive so for a limit [0,1] the lowest index refers to 0, the
            highest 1.
    """

    def __init__(self, n_dim, limits=None, n_samples=10):
        self._n_dim = n_dim
        self._limits = limits
        self._n_samples = n_samples
        self._indexes = []

        for junk in xrange(self._n_dim):
            self._indexes.append(range(self._n_samples))

    def generate_samples(self):
        self._samples = []
        while len(self._samples) < self._n_samples:
            draw = numpy.random.randint(numpy.shape(self._indexes)[1],
                                        size=self._n_dim)
            out = []
            for n in xrange(self._n_dim):
                out.append(self._indexes[n].pop(draw[n]))
            self._samples.append(out)

    def get_samples(self):
        if (self._limits is not None and
            numpy.shape(self._limits)[0] == self._n_dim):
            for sample in self._samples:
                for n in xrange(self._n_dim):
                    diff = self._limits[n][1] - self._limits[n][0]
                    sample[n] = (sample[n]*diff/(1.0*self._n_samples)+
                                 self._limits[n][0])
        return self._samples

    def clear(self):
        self._samples = []
        self._indexes = []

        for junk in xrange(self._n_dim):
            self._indexes.append(range(self._n_samples))


class OLHS(LHS):
    """
    Generates samples from an Othoginal Latin Hypercube given the imput 
    parameters.
 
    Atributes:
        n_dim: integer of the number of dimentions the Latin Hypercube spans
        limits: a list of lists containing the ranges for which each
            dimension the hypercube spans. Default None returns the raw 
            indieces of the leaving the user to determain the translation to
            'physical' quantities.
        n_samples: number of samples to draw from the hypercube. This is 
            also defines the segmenting of each axis. The axis segmention is
            inclusive so for a limit [0,1] the lowest index refers to 0, the
            highest 1.
    """

    def __init__(self, n_dim, limits=None, n_samples=10):
        if subspace_factor == None:
            factors = []
            if n_samples%2 != 0:
                print "Invalid samples size request."
                print "\tPick a number of samples divisable by 2"
                raise InvalidSampleSizeError
            for f in numpy.arange(2, n_samples/2+1, dtype='int'):
                if n_samples%f == 0:
                    factors.append(f)
            for f in reversed(factors):
                if (f < n_samples and 
                    numpy.power(n_samples/f,n_dim) >= n_samples):
                    self._factor = int(f)
                    break
        self._used_subspace = []
        LHS.__init__(self, n_dim, limits, n_samples)

    def generate_samples(self):
        self._samples = []
        while len(self._samples) < self._n_samples:
            draw = numpy.random.randint(numpy.shape(self._indexes)[1],
                                        size=self._n_dim)
            out = []
            for n in xrange(self._n_dim):
                out.append(self._indexes[n][draw[n]])   
            for subspace in self._used_subspace:
                if numpy.all(subspace==(numpy.array(out)/self._factor)):
                    continue
            self._used_subspace.append(numpy.array(out)/self._factor)
            for n in xrange(self._n_dim):
                self._indexes[n].remove(out[n])
            self._samples.append(out)

    def clear(self):
        self._used_subpace = []
        LHS.clear(self)
