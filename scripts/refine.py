#!/share/bigstor2/opt/bin/python
import argparse
import sys
import parallelclusterer.daura_clustering as daura
import logging

def main():
    parser = argparse.ArgumentParser(description='''
     perform clustering using Daura's clustering algorithm    
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-projectfile", dest='projectfile', default='my_project.yaml',
                        help='Project file')
    
    parser.add_argument("-cutoff", dest='cutoff', help='metric cutoff for clustering',
        default=0.05, type=float)

    parser.add_argument("-clusterfile", dest='clusterfile', default='clusters1.txt',
                        help='output cluster assignment file')

    parser.add_argument("-centerfile", dest='centerfile', default='centers1.txt',
                        help='Project file')

    parser.add_argument("-checkpoint", dest='checkpoint', default=None,
                        help='Checkpoint file')

    args = parser.parse_args()
    #print args

    clusters  = daura.refine(args.projectfile, args.cutoff,"centers.txt")
#     clusters = daura.cluster(args.projectfile, args.cutoff,args.checkpoint)
    daura.cluster_dict_to_text(clusters, args.centerfile, args.clusterfile)
    

main()
