import os, sys
from glob import glob
#from distutils.core import setup,Extension
# setuptools needs to come before numpy.distutils to get install_requires
#import setuptools 
import numpy
from distutils.core import setup
from distutils.extension import Extension

#from distutils import sysconfig
#from numpy.distutils.core import setup, Extension
#from numpy.distutils.misc_util import Configuration

VERSION = "0.1"
__author__ = "Alex and Gurpreet"
__version__ = VERSION

# metadata for setup()
metadata = {
    'name': 'parallelclusterer',
    'version': VERSION,
    'author': __author__,
    'license': 'GPL version 2 or later',
    'license': 'GPL version 2 or later',
    'author_email': 'togurpreet@gmail.com',
    'url': 'https://sites.google.com/site/togurpreet/Home',
    #'install_requires': ['numpy','pyyaml','gp_grompy'],
    'platforms': ["Linux"],
    'description': "parallel clustering in python",
    'long_description': """
    """}






# add the scipts, so they can be called from the command line
scripts = [e for e in glob('scripts/*.py') if not e.endswith('__.py')]
## add cmetric
gmx_rmsd = Extension('_gmxrmsd',sources=glob('src/gmx_rmsd/*.c'),
                 extra_compile_args = ["-O3","-fopenmp","-fomit-frame-pointer",
                 "-finline-functions","-Wall","-Wno-unused","-msse2",
                 "-funroll-all-loops","-std=gnu99","-fexcess-precision=fast", "-pthread"],
                 extra_link_args = ["-lmd","-lgmx","-lfftw3f","-lnsl","-lm","-lgomp"],
                 include_dirs = ["src/gmx_rmsd"],
                 #library_dirs=['/home/gurpreet/sft/gmx455/lib'],
                 )
gmx_rmsd_custom1 = Extension('_gmxrmsd_custom1',sources=glob('src/gmx_rmsd_custom1/*.c'),
                 extra_compile_args = ["-O3","-fopenmp","-fomit-frame-pointer",
                 "-finline-functions","-Wall","-Wno-unused","-msse2",
                 "-funroll-all-loops","-std=gnu99","-fexcess-precision=fast", "-pthread"],
                 extra_link_args = ["-lmd","-lgmx","-lfftw3f","-lnsl","-lm","-lgomp"],
                 include_dirs = ["src/gmx_rmsd_custom1"],
                 #library_dirs=['/home/gurpreet/sft/gmx455/lib'],
                 )


setup(packages = ["parallelclusterer"],
      package_dir = {'parallelclusterer':'src'},
      ext_package = "parallelclusterer",
      ext_modules = [gmx_rmsd,gmx_rmsd_custom1],
      scripts=scripts,
      **metadata)



