#! /usr/bin/env python
import argparse
import sys,os
import logging
import parallelclusterer as pcl
from parallelclusterer.project import Project

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='''
 Create project file    
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--projectfile", dest='projectfile', default='my_project.yaml',
                    help='Project file')

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


parser.add_argument("--trajtype", dest='traj_type', help='type of trajectory',
    default='xtc')

parser.add_argument("--trajlist", dest='trajlist', help='file containing trajectory paths',)


def get_trajpaths(filename):
    with open(filename) as f:
        alist = [line.rstrip() for line in f]
    
    return alist
        
    



def main(args):
    
    trajectory_filepaths = get_trajpaths(args.trajlist)

    my_project = Project(   trajectory_filepaths=trajectory_filepaths,
                            trajectory_type=args.traj_type,
                            gro_filepath=args.gro_filepath,
                            ndx_filepath=args.ndx_filepath,
                            tpr_filepath=args.tpr_filepath,
                            number_dimensions=args.number_dimensions,
                            stride=args.stride,  )
                            
    my_project.write_project("./my_project.yaml")

if __name__ == '__main__':
    args = parser.parse_args()
    print "#################################"
    print "## Program version",pcl.__version__
    print "## Invoked with Following Arguments " 
    for key, value in vars(args).items():
        print "# %s = %s"%(key,value)
    print "#################################"
   
    main(args)




