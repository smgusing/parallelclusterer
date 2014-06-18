#! /usr/bin/env python
import argparse
import sys,os
import logging

import parallelclusterer as pcl
import parallelclusterer.daura_clustering as daura
from parallelclusterer.gmx_metric_rmsd import Gmx_metric_rmsd
from parallelclusterer.gmx_metric_custom1 import Gmx_metric_custom1

logger = logging.getLogger("parallelclusterer")

parser = argparse.ArgumentParser(description='''
 program to assign and refine clusters. (only assign for now)    
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-projectfile", dest='projectfile', default='my_project.yaml',
                    help='Project file')

parser.add_argument("-cutoff", dest='cutoff', help='metric cutoff for clustering',
    default=0.05, type=float)

parser.add_argument("-newclusterfile", dest='clusterfile', default='new_clusters.txt',
                    help='output cluster assignment file')

parser.add_argument("-newcenterfile", dest='centerfile', default='new_centers.txt',
                    help='new cluster centers with unclustered included')

parser.add_argument("-oldcenterfile", dest='oldcenterfile', default='centers.txt',
                    help='old cluster center file')

parser.add_argument("-metric", dest='metric', default='gmx_rmsd',
                    help='metric to apply. options are "gmx_rmsd" or "custom_1"')

parser.add_argument("--nopreprocessing", action="store_true",dest='no_preprocess', default=False,
                    help='if true, the trajectories will not be preprocessed for COM removal')

parser.add_argument("-l", dest='loglevel', help='level of logging (info,warn,debug)',
        default="debug")


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

    clusters = daura.assign(Metric, args.projectfile, args.cutoff,
                            args.oldcenterfile,
                            args.no_preprocess)
    daura.cluster_dict_to_text(clusters, args.centerfile, args.clusterfile)


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
    

