import numpy

degToRad = numpy.pi/180.0

def cosmology_unit_test():
    import cosmology
    print "\n****************************"
    print "*                          *"
    print "* Testing Cosmology Module *"
    print "*                          *"
    print "****************************\n"
    print "Testing Single Epoch"
    print "****************************"

    ### Create single epoch cosmologies at the redshift specified
    ### outputs to stdout
    cosmo = cosmology.SingleEpoch(redshift=0.0)
    cosmo.write()

    cosmo.set_redshift(redshift=0.5)
    cosmo.write()

    cosmo.set_redshift(redshift=1.0)
    cosmo.write()
    
    cosmo.set_redshift(redshift=2.0)
    cosmo.write()

    cosmo.set_redshift(redshift=3.0)
    cosmo.write()

    ### Compute example multi epoch cosmologies from redshift z=0.0 to z=5.0
    ### output are the computed comoving distnace as a funciton of redshift and
    ### several other cosmological variables (Omega_m(z), Omega_L(z), etc.)
    print "\nTesting Multi Epoch"
    print "****************************"
    cosmo = cosmology.MultiEpoch(z_min=0.0, z_max=5.0)
    z = 0.0
    print ("Multi Epoch: (z, chi [Mpc/h], growth, omega_m(z), omega_l(z), "
           "detla_c, delta_v, sigma_8)")
    for z in [0.0, 0.5, 1.0, 2.0, 3.0]:
        print (z, cosmo.comoving_distance(z), cosmo.growth_factor(z),
               cosmo.omega_m(z), cosmo.omega_l(z), cosmo.delta_c(z),
               cosmo.delta_v(z), cosmo.sigma_r(8.0, z))
    print ""
    
    ### Backup write command if more information is needed
    # cosmo.write('test_cosmology.ascii')

def mass_function_unit_test():
    import mass_function
    print "\n********************************"
    print "*                              *"
    print "* Testing Mass Function Module *"
    print "*                              *"
    print "********************************\n"
    
    ### Compute the mass function at redshift z=0.0
    mass_func = mass_function.MassFunction(redshift=0.0)
    mass_array = numpy.logspace(9, 16, 5)
    print "Mass Funct: (mass [M_solar/h], nu, dN/dM)"
    for mass in mass_array:
        print "\t", (mass, mass_func.nu(mass), mass_func.f_m(mass))
    print ""

    ### Backup write command if more information is needed
    # mass_func.write('test_mass_function.ascii')

def hod_unit_test():
    import hod
    print "\n**********************"
    print "*                    *"
    print "* Testing HOD Module *"
    print "*                    *"
    print "**********************\n"
    ### Create a Zheng et al. 2007 HOD object with central mass 10**13, 
    ### satellite mass difference. The other parameters are fixed M_0 = M_min,
    ### alpha = 1.0, log_sigma_m = 0.15
    zheng = hod.HODZheng(10**12, 0.15, 10**12, 10**13, 1.0)
    mand = hod.HODZheng(10**12, 1.0)
    mass_array = numpy.logspace(9, 16, 5)
    ### compute the first three moments of the HOD and print to screen
    print "HOD: (mass [M_solar/h], <N>, <N(N-1)>, <N(N-1)(N-2)>)"
    for mass in mass_array:
        print "\t", (mass, mand.first_moment(mass), mand.second_moment(mass),
                     mand.nth_moment(mass, 3))
    print ""

def camb_unit_test():
    pass

def halo_unit_test():
    import halo
    import hod
    print "\n***********************"
    print "*                     *"
    print "* Testing Halo Module *"
    print "*                     *"
    print "***********************\n"
    

    ### We test each of the 4 avalible power spectra as a function of redshift
    ### first create a halo occupation distribution object using a Zheng07 HOD
    zheng = hod.HODZheng(10**13.0, 0.15, 10**13.0, 10**14.0, 1.0)
    ### initialize the halo model object at z=0.0 with the Zheng HOD
    h = halo.Halo(redshift=0.0, input_hod=zheng)
    print ("Halo: (k [Mpc/h], linear_power, power_mm, "
           "power_gm, power_gg [(Mpc/h)^3])")
    k_array = numpy.logspace(-3, 2, 5)
    for k in k_array:
        print"\t", (k , h.linear_power(k), h.power_mm(k),
                    h.power_gm(k), h.power_gg(k))
    print ""

    ### Backup write commands if more information is needed
    # h.write('test_halo_power_spectra.ascii')
    # h.write_halo('test_halo_properties.ascii')
    # h.write_power_components('test_halo_power_components.ascii')

def kernel_unit_test():
    import cosmology
    import kernel
    print "\n*************************"
    print "*                       *"
    print "* Testing kernel Module *"
    print "*                       *"
    print "*************************\n"

    print "Testing dNdz"
    print "*************************"

    ### To define a Kernel object we need several things first. We need redshift
    ### distributions as well as the corresponding window functions.
    
    ### initilized to galaxy redshift distributions one as a magnitude limited
    ### sample, the other a Guassian with mean z=1.0
    lens_dist = kernel.dNdzMagLim(z_min=0.0, z_max=2.0, a=2, z0=0.3, b=2)
    source_dist = kernel.dNdzGaussian(z_min=0.0, z_max=2.0, z0=1.0, sigma_z=0.2)
    ### normalize the distributions and create PDFs
    lens_dist.normalize()
    source_dist.normalize()

    z_array = numpy.linspace(0.0, 2.0, 5)
    print "Lens dNdz: (z, p(z)dz)"
    for z in z_array:
        print "\t",(z, lens_dist.dndz(z))

    print "Source dNdz: (z, p(z)dz)"
    for z in z_array:
        print "\t",(z, source_dist.dndz(z))
    print ""

    print "Testing WindowFunction"
    print "*************************"
    cosmo = cosmology.MultiEpoch(0.0, 2.0)

    ### using the distributions defined above compute the distance weighted
    ### window functions for use in projecting a powerspectrum
    ### Define a galaxy window function
    chi_array = cosmo.comoving_distance(z_array)
    lens_window = kernel.WindowFunctionGalaxy(redshift_dist=lens_dist)
    print "Lens Window: (chi [Mpc/h], window value [h/Mpc])"
    for chi in chi_array:
        print "\t",(chi, lens_window.window_function(chi))
    ### Backup write command if more information is needed
    # lens_window.write('test_galaxy_window_function.ascii')

    ### Define a lensed population of galaxies
    source_window = kernel.WindowFunctionConvergence(redshift_dist=source_dist)
    print "Source Window: (chi [Mpc/h], window value [h/Mpc])"
    for chi in chi_array:
        print "\t",(chi, source_window.window_function(chi))

    ### Backup write command if more information is needed
    # source_window.write('test_convergence_window_function.ascii')

    print "Testing Kernel"
    print "*************************"
    
    ### Initialize the kernel objects for projecting a power spectrum in z space
    ### Initilize the kernel for galaxy clustering
    ln_ktheta_array = numpy.linspace(-15, -1, 5)
    k_Auto = kernel.Kernel(ktheta_min=0.001*degToRad*0.001, 
                           ktheta_max=1.0*degToRad*100.0,
                           window_function_a=lens_window, 
                           window_function_b=lens_window)
    print "Auto Kernel: (k*theta [h/Mpc*Radians], kernel value [(h/Mpc)^2])"
    for ln_ktheta in ln_ktheta_array:
        print "\t",(numpy.exp(ln_ktheta), k_Auto.kernel(ln_ktheta))
    ### Backup write command if more information is needed
    # k_Auto.write('test_clustering_kernel.ascii')

    ### Kernel computing lensing convergence
    k_Con = kernel.Kernel(0.001*degToRad*0.001, 1.0*degToRad*100.0,
                          lens_window, source_window)
    print ("Convergence Kernel: (k*theta [h/Mpc*Radians], "
           "kernel value [(h/Mpc)^2])")
    for ln_ktheta in ln_ktheta_array:
        print "\t",(numpy.exp(ln_ktheta), k_Con.kernel(ln_ktheta))
    print ""
    ### Backup write command if more information is needed
    # k_Con.write("test_convergence_kernel.ascii")    
    
    ### Print out the redshifts for which the kernel is maximaly sensitive
    print "Peak Sensitivity at Redshifts:", k_Auto.z_bar, k_Con.z_bar

def correlation_unit_test():
    import correlation
    import halo
    import hod
    import kernel
    print "\n******************************"
    print "*                            *"
    print "* Testing Correlation Module *"
    print "*                            *"
    print "******************************\n"

    ### Definining a correlation object requires first two window functions
    ### (these could in principle be the same window function), theta bounds to
    ### compute the correlation over, and optionaly and HOD object and a 
    ### specification as to which power spectrum to use. Note: the different
    ### correlation classes have approprate default values.

    ### As in kernel_unit_test create galaxy distributions
    lens_dist = kernel.dNdzMagLim(0.0, 2.0, 2, 0.3, 2)
    source_dist = kernel.dNdzGaussian(0.0, 2.0, 1.0, 0.2)

    ### create appropreate window objects
    lens_window = kernel.WindowFunctionGalaxy(lens_dist)
    source_window = kernel.WindowFunctionConvergence(source_dist)

    ### Turn those window functions into kernels for projecting 3-D functions
    ### onto the sky. auto_kernel represents a auto-correlation clustering
    ### measurment. mag_kernel is a convergence correlation.
    auto_kernel = kernel.Kernel(0.001*0.001*degToRad, 100.0*1.0*degToRad,
                                lens_window, lens_window)
    mag_kernel = kernel.Kernel(0.001*0.001*degToRad, 100.0*1.0*degToRad,
                               lens_window, source_window)


    ### define an hod (optional but needed in order to use power_gm or power_gg)
    zheng = hod.HODZheng(10**13.0, 0.15, 10**13.0, 10**14.0, 1.0)
    halo_model = halo.Halo(redshift=0.0, local_hod=zheng)

    ### Define the correlation objects. Note that each of these correlations
    ### is computed using the nonlinear dark matter power spectrum, power_mm.
    ### other options are linear_power which computes the correlation for the
    ### linear spectrum only, power_gm which is the galaxy-matter cross spectrum
    ### and power_gg which is the galaxy-galaxy power spectrum.
    ### Here we define the correlation function of galaxy clustering, note that
    ### it takes only one window function as the second is assumed identical
    theta_array = numpy.logspace(-3, 0, 5)
    auto = correlation.Correlation(theta_min=0.001*degToRad,
                                   theta_max=1.0*degToRad, 
                                   input_kernel=auto_kernel,
                                   input_halo=halo_model,
                                   powSpec='power_gg')
    print "Auto Correlation: (theta [deg], wtheta)"
    for theta in theta_array:
        print "\t",(theta, auto.correlation(theta*degToRad))
    ### Define the correlation for galaxy-galaxy magnification. Note it takes
    ### and WindowFunctionGalaxy object and a WindowFunctionConvergence Object
    mag = correlation.Correlation(0.001*degToRad, 1.0*degToRad, 
                                  input_kernel=mag_kernel,
                                  input_halo=halo_model,
                                  powSpec='power_gm')
    print "Convergence Correlation: (theta [deg], wtheta)"
    for theta in theta_array:
        print "\t",(theta, mag.correlation(theta*degToRad))
    print ""
   
    ### Backup write command if more information is needed
    ### Compute the correlation functions between the angular bounds
    # auto.compute_correlation()
    # mag.compute_correlation()
    # auto.write('test_clustering_correlation.ascii')
    # mag.write('test_convergence_correlation.ascii')
    
def camb_unit_test():
    pass

if __name__ == "__main__":
    cosmology_unit_test()
    mass_function_unit_test()
    hod_unit_test()
    halo_unit_test()
    kernel_unit_test()
    correlation_unit_test()
    camb_unit_test()
