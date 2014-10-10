#! /usr/bin/env python
import argparse
import numpy as np
import logging
import parallelclusterer as pcl
from parallelclusterer.gmx_metric_rmsd import Gmx_metric_rmsd
#from parallelclusterer.gmx_metric_custom1 import Gmx_metric_custom1
from gp_grompy import Gmstx,Gmxtc
from parallelclusterer.framecollection import Framecollection
logger = logging.getLogger("parallelclusterer")

    
parser = argparse.ArgumentParser(description='''
 Create project file    
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--gro", dest='gro_filepath', default='nr.gro',
                    help='gro or pdb file')

parser.add_argument("--ndx", dest='ndx_filepath', default='nr.ndx',
                    help='index file')

parser.add_argument("--tpr", dest='tpr_filepath', default='null.tpr',
                    help='tpr file')

parser.add_argument("--ndim", dest='number_dimensions', help='number of dimensions',
    default=3, type=int)

parser.add_argument("--stride", dest='stride', help='stride for reading data',
    default=3, type=int)

parser.add_argument( "-ndim",dest='ndim', help='number of dimensions for fitting (2 or 3) Z is not included in 2',
    default=3, type=int)

parser.add_argument("--xtc", dest='xtc_filepath', default='nr.xtc',
                    help='xtc file')


def readRefStr(InFn):
    stxReader = Gmstx()
    stxReader.read_stx(InFn)
    fr = stxReader.x_to_array()
    return fr.reshape(1,fr.shape[0],3)
    
def loadTraj(InFn):
    xtcReader = Gmxtc()
    traj = xtcReader.load_traj(InFn)
    return traj
    



def main(args):
    
    metric = Gmx_metric_rmsd(tpr_filepath=args.tpr_filepath, 
                                ndx_filepath=args.ndx_filepath, 
                                stx_filepath = args.gro_filepath,
                                number_dimensions=args.ndim)
    
    refstr = readRefStr(args.gro_filepath)
    traj = loadTraj(args.xtc_filepath)
    trajComb = np.vstack([refstr,traj])
    globalIDs = np.arange(len(trajComb))
    localIDs = np.copy(globalIDs)
    traj_container = Framecollection(globalIDs = globalIDs,
                                   localIDs = localIDs,
                                   frames = trajComb)
    

    metric.preprocess(
            frame_array_pointer = traj_container.get_first_frame_pointer(),
            number_frames = traj_container.number_frames,
            number_atoms = traj_container.number_atoms)
    
    rmsd = np.zeros(len(trajComb), dtype=np.float32)
    metric.fit_trajectory(traj_container, 0, rmsd)
    rmsdoutf="rmsd00.txt"
    np.savetxt(rmsdoutf, rmsd)
    
    

    

if __name__ == '__main__':
    args = parser.parse_args()
    print "#################################"
    print "## Program version",pcl.__version__
    print "## Invoked with Following Arguments " 
    for key, value in vars(args).items():
        print "# %s = %s"%(key,value)
    print "#################################"
   
    main(args)




