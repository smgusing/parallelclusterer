import os,sys
from os import environ
from ctypes import cdll
import pkg_resources as pkgres
import logging
### Set up version information####
__version__ = pkgres.require("parallelclusterer")[0].version
########################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(fmt='[%(name)s] %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.propagate = False
#########################################
pth=os.path.dirname(os.path.abspath(__file__))
libcmetricpath=("%s/_cmetric.so"%pth)
cmetric = cdll.LoadLibrary(libcmetricpath)
