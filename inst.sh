#! /bin/bash
source $HOME/.bashrc
# Source your python environment
lpy
# Source gromacs variables
gmx455
##install_dir=/home/gurpreet/sft/python/pyenv2
##incdir=/home/gurpreet/sft/gpgmx455/include/gromacs
##libdir="/home/gurpreet/sft/gpgmx455/lib:/storage/gurpreet/install/fftw-3.2.1/lib"
##################################################################################
install_dir=/home/gurpreet/sft/python/env1
#install_dir=/home/gurpreet/sft/python/env1/lib/python2.7/site-packages/parallelclusterer

gmxdir=`which mdrun`
gmxdir=${gmxdir%/bin/mdrun}
echo Gromacs directory $gmxdir
incdir=$gmxdir/include/gromacs
libdir="$gmxdir/lib"

function clean {
echo "Will clean the installation"
sleep 1
## Unless deleted manually, pip is unable to clear all the crap
## so here I am deleting almost all the files manually
## the files in the bin are not cleared
yes | pip uninstall parallelclusterer
}

function inst {
clean
#python setup.py clean
#python setup.py sdist
python setup.py build_ext --include-dirs $incdir   --library-dirs $libdir
python setup.py install --prefix $install_dir
#python setup.py install 
}

function doc {
# In order to build the documentation the path to the installed package $pkg_path should be set
pkg_path=${install_dir}/lib/python2.7/site-packages/parallelclusterer
sphinx-apidoc $pkg_path -o doc --full
cd doc; make html

}
inst
doc
