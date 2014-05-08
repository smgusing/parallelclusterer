#! /bin/bash
source $HOME/.bashrc
lpy
gmx455
##install_dir=/home/gurpreet/sft/python/pyenv2
##incdir=/home/gurpreet/sft/gpgmx455/include/gromacs
##libdir="/home/gurpreet/sft/gpgmx455/lib:/storage/gurpreet/install/fftw-3.2.1/lib"
##################################################################################
#install_dir=/home/gurpreet/sft/python/env1
install_dir=/home/gurpreet/sft/python/env1/lib/python2.7/site-packages/parallelclusterer

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
rm -rf build dist parallelclusterer.egg-info $install_dir 
}


clean
#python setup.py clean
#python setup.py sdist
python setup.py build_ext --include-dirs $incdir   --library-dirs $libdir
#python setup.py install --prefix $install_dir
python setup.py install 
