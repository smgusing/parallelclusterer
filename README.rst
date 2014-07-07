===============================================
Parallelclusterer
===============================================

:Author: Gurpreet Singh and Alex Chen
:Contact: togurpreet@gmail.com
:License: Read LICENSE.txt 

-----------------------------------------------
Introduction
----------------------------------------------- 
Parallelclusterer is a python module that can be used for clustering 3D objects (proteins) in parallel.
Currently only Daura clustering algorithm is implemented. 

-----------------------------------------------
Installation
-----------------------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Prerequists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Numpy (developed using version 1.8.0)
- PyYaml
- Gromacs (version 4.5.*) developed using 4.5.5
- gp_grompy (uses Gromacs-4.5.5)
- gcc compiler
    + gcc-4.7.3 and above. 
      **NOTE: The gcc-4.4.* and gcc-4.1.* give troubles**
      
- tables (optional)
    + some utilities will not work if is absent  
- mpi4py (developed using version 1.3)

- Msmbuilder (optional) version 2.7

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Quick install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Gromacs include and lib directories are needed for compilation of C code.::

 python setup.py build_ext --include-dirs < gromacs include directory >   --library-dirs < gromacs library directory >
 python setup.py install


The installation script ``inst.sh`` is provided. Please modify it according to your environment.
The script will **not** work as is.

.. DANGER:: The script naively tries to remove the previous installation using ``rm`` command, **DO NOT USE IT WITHOUT MODIFICATIONS**.  

In some cases ``-fexcess-precsion=fast`` is not supported. Deleting the flag from setup.py should do the trick.

-------------------------------------------------
Usage
-------------------------------------------------
The directory ``test_dataset0`` contains ``test.sh``. The script shows the basic usage. Use ``cluster.py -h`` for further help.
   







