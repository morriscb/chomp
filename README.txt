================================================================================
CHOMP:
Cosmology and HalO Model Python code
version: Beta 2 (6/4/12)

Authors:
Christopher Morrison
Ryan Scranton
================================================================================

================================================================================
1. Introduction
================================================================================

CHOMP is an object oriented python code for predicting power spectra and 
correlation functions designed to be flexible accommodate new theoretical models
as the field of cosmology and large-scale structure progresses.

It currently implements a halo model from Seljak et al. 2000 however users are
encouraged to expand on code base using the current class template APIs,
implementing their preferred model.

================================================================================
2. Installation
================================================================================

The mininum requirements for the code are:

python2.7
numpy
scipy
matplotlib is recommended for plotting but not required

Installing the code is as simple as adding the source directory to your
PYTHONPATH. Testing the code is as simple as running the command

"python unit_test.py"

================================================================================
3. Modules
================================================================================

For current details on each model see their respective .py files. For an example
on implementing the code see example_script.py.

================================================================================
3.1 cosmology
================================================================================

================================================================================
3.2 mass_function
================================================================================

================================================================================
3.3 hod
================================================================================

================================================================================
3.4 halo
================================================================================

================================================================================
3.5 kernel
================================================================================

================================================================================
3.6 correlation
================================================================================

Refs: 
Seljak et al. 2000
Sheth & Tormen 1999
(Others I still need to write down)
