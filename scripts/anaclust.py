#! /usr/bin/env python
import argparse
import numpy as np
import logging
from parallelclusterer.utils import Utilities
import parallelclusterer as pcl
from parallelclusterer.gmx_metric_rmsd import Gmx_metric_rmsd
from parallelclusterer.gmx_metric_custom1 import Gmx_metric_custom1

logger = logging.getLogger("parallelclusterer")

    
parser = argparse.ArgumentParser(description='''
 perform routine analysis on cluster results     
''',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-projfn",dest='projfn',default='Proj.yaml',help='Project file')

parser.add_argument( "-timestep",dest='timestep', help='timestep for traj dumping (in ps)',
    type=float,required=True)

parser.add_argument( "-ndim",dest='ndim', help='number of dimensions for fitting (2 or 3) Z is not included in 2',
    default=3, type=int)

parser.add_argument( "-stepsize",dest='stepsize', help='number of integration steps per recorded observeration',
     required=True,type=int)

parser.add_argument( "-clusters",dest='cltagsfn', help='''assignments file''',
default='clusters.txt')

parser.add_argument( "-centers",dest='clcentfn', help='''cluster center file''',
default='centers.txt')

parser.add_argument( "-o",dest='outf', help='''Output file''',
    default='aligned.xtc')

parser.add_argument( "-cxtc",dest='centfile', help='''Output file''',
    default='clcenters.xtc')

parser.add_argument( "-clid",dest='clid',type=int,default=0,help='''cluster number''')

parser.add_argument( "-nconf",dest='nconf',type=int,help='''cluster elements''',
    default=0)

parser.add_argument( "-splitat",dest='splitat',type=int,help=''' split number of
trajectories into two sets based on this value
''',default=0)

parser.add_argument("-metric", dest='metric', default='gmx_rmsd',
                help='metric to apply. options are "gmx_rmsd" or "custom_1"')

parser.add_argument("--nopreprocessing", action="store_true",dest='no_preprocess', default=False,
                help='if true, the trajectories will not be preprocessed for COM removal')


parser.add_argument("-l",dest='loglevel', help='level of logging (info,warn,debug)',
    default="debug")

parser.add_argument("--get_centers",action="store_true", dest="bCenters", default=False,
              help="get cluster centers as xtc")

parser.add_argument("--get_clusterno",action="store_true", dest="bClustno", default=False,
              help="get random conformation from cluster")

parser.add_argument("--get_graph",action="store_true", dest="bGraph", default=False,
              help="get graph representation")


parser.add_argument("--get_size",action="store_true", dest="bSize", default=False,
              help="get graph representation")


parser.add_argument("--msm",action="store_true", dest="bMsm", default=False,
              help="write msmbuilder compatible output")

parser.add_argument("-convert_assignment", dest="assignoutfn", default=None,
              help="write msmbuilder compatible output")


parser.add_argument( "-dumpmatrix",dest='dumplagtime', help='lagtime',
                     default=0, type=int)

parser.add_argument( "-matrixoutfn",dest='matrixoutfn', help='matrixoutfn',
                     default="count_matrix.txt")

parser.add_argument( "-assignfn",dest='assignfn', help='assignment file name',
                     default="Assignments.h5")

parser.add_argument("--get_countgraph",action="store_true", dest="bcountGraph", default=False,
              help="get graph representation")

    
    
    
def main(args):    

    if args.metric == 'gmx_rmsd':
        Metric = Gmx_metric_rmsd
        logger.debug("Will use standard RMSD fitting with LSF")
    elif args.metric == 'custom_1':
        Metric = Gmx_metric_custom1
        logger.debug("Will use custom LSF and RMSD")
    else:
        logger.error("%s not implimented",args.metric)
        raise SystemExit("Quitting on this")
    
    
    utils=Utilities(Metric,
                    projfn=args.projfn,
                    clcenterfn=args.clcentfn,
                    clusterfn=args.cltagsfn,
                    stepsize=args.stepsize,
                    timestep=args.timestep,
                    flag_nopreprocess=args.no_preprocess )
    
    if args.bSize == True:
        utils.print_clustersizes()
        
    if args.bCenters == True:
        utils.get_clustercenters(outf=args.centfile)
        
    if args.bClustno == True:
        utils.get_cluster(args.clid,nconfs=args.nconf)
        
    if args.bGraph == True:
        utils.build_network_rmsd(args.centfile,args.nconf)

    if args.bMsm == True:
        print "MSMbuilder dependent utilites"
        if args.assignoutfn is not None:
            utils.write_msm_output(filename=args.assignoutfn)
            
        if args.dumplagtime != 0:
            utils.dump_count_matrix(args.assignfn,args.dumplagtime,
                                    args.matrixoutfn)
                
        if args.bcountGraph == True:
            utils.build_network_count(args.matrixoutfn)
        
        
if __name__ == '__main__':
    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logger.setLevel(numeric_level)
    print "#################################"
    print "## Program version",pcl.__version__
    print "## Invoked with Following Arguments " 
    for key, value in vars(args).items():
        print "# %s = %s"%(key,value)
    print "#################################"
   
    main(args)
    
