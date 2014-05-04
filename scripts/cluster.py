#! /usr/bin/env python
import argparse
import sys,os
import parallelclusterer
import parallelclusterer.daura_clustering as daura
import logging

logger = logging.getLogger("parallelclusterer")

parser = argparse.ArgumentParser(description='''
 perform clustering using Daura's clustering algorithm    
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-projectfile", dest='projectfile', default='my_project.yaml',
                    help='Project file')

parser.add_argument("-cutoff", dest='cutoff', help='metric cutoff for clustering',
    default=0.05, type=float)

parser.add_argument("-clusterfile", dest='clusterfile', default='clusters.txt',
                    help='output cluster assignment file')

parser.add_argument("-centerfile", dest='centerfile', default='centers.txt',
                    help='Project file')

parser.add_argument("--usecheckpoint", action="store_true",dest='checkpoint', default=False,
                    help='Start from the checkpointfile ".daura_last.checkpoint"')

parser.add_argument("-l", dest='loglevel', help='level of logging (info,warn,debug)',
        default="debug")


def main(args):
    if args.checkpoint == True:
        checkpointfile = ".daura_last.checkpoint"

        if not os.path.isfile(checkpointfile):
            fchkp =".daura_first.checkpoint"
            logger.info("File %s does not exist. Will look for %s ",checkpointfile,fchkp)
            checkpointfile = fchkp
        else:
            logger.info(" Will use checkpoint file %s",checkpointfile)
            
            if not os.path.isfile(checkpointfile):
                logger.error(" File %s does not exist",checkpointfile)
                raise SystemExit("Quitting over this. Either provide the files or remove the --usecheckpoint flag")
            else:
                logger.info("Will use checkpoint file %s",checkpointfile)
    else:
        checkpointfile = None
        logger.info("Will not use any checkpoint")
        
    clusters = daura.cluster(args.projectfile, args.cutoff,checkpointfile)
    daura.cluster_dict_to_text(clusters, args.centerfile, args.clusterfile)


if __name__ == '__main__':
    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logger.setLevel(numeric_level)
    print "#################################"
    print "## Program version",parallelclusterer.__version__
    print "## Invoked with Following Arguments " 
    for key, value in vars(args).items():
        print "# %s = %s"%(key,value)
    print "#################################"
   
    main(args)
    

