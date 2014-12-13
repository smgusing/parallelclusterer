#! /usr/bin/env python
# -*- coding: utf-8 -*-
##############################################################################
# Functions for analysis of clustered data 
#
#
#############################################################################
import sys
import numpy as np

from gp_grompy import libgmx, rvec, matrix, \
                 c_real, Gmstx, Gmxtc, Gmndx
from ctypes import byref, c_int, c_uint, c_float, cdll, POINTER, Structure
import txtreader
from project import Project
from framecollection import Framecollection
import logging


logger = logging.getLogger("parallelclusterer")
# #POINTERTYPES
c_int_p = POINTER(c_int)
c_real_p = POINTER(c_real)
c_uint_p = POINTER(c_uint)

def split(longlist, lengths):
    """Split a long list into segments

    Parameters
    ----------
    longlist : array_like
        Long trajectory to be split
    lengths : array_like
        list of lengths to split the long list into

    Returns
    -------
    A list of lists
    
    copied from msmbuilder
    """

    if not sum(lengths) == len(longlist):
        raise Exception('sum(lengths)=%s, len(longlist)=%s' % (sum(lengths), len(longlist)))
    func = lambda (length, cumlength): longlist[cumlength - length: cumlength]
    iterable = zip(lengths, np.cumsum(lengths))
    output = map(func, iterable)
    return output


## bad bad class######

class Utilities():
    ''' contains utilities to perform analysis on cluster output
    '''

    def __init__(self, Metric, projfn="my_project.yaml", 
                 clcenterfn="centers.txt",
                 clusterfn="clusters.txt",
                 stepsize=None, timestep=None, 
                 flag_nopreprocess=False):
        '''
        load the project information and cluster output
        '''
        prj = Project(existing_project_file=projfn)
        trajs_lengths = np.array(prj.get_trajectory_lengths())
        
        # frame numbers always correspond to full trajectory irrespective of stride
        #trajs_lengths = trajs_lengths 
        self.trajs_lengths = trajs_lengths.astype(np.int)
        self.trajnames = prj.get_trajectory_filepaths()
        self.ntrajs = len(self.trajnames)
        self.stride = prj.get_stride()
        self.frames_per_traj=np.ceil(self.trajs_lengths/(self.stride*1.))
        
        self.ndim = prj.get_number_dimensions()
        self.trajectory_type = prj.get_trajectory_type()

        self.grof = prj.gro_filepath
        self.tprf = prj.tpr_filepath
        self.ndxf = prj.ndx_filepath
        
        self.Metric = Metric
        # # Get input data


        centids = txtreader.readcols(clcenterfn)
        self.centids = centids[:, 1]
        
        self.cltags = txtreader.readcols(clusterfn)       
        self.assignments = self._get_assignments()
        self.nodesizes = np.bincount(self.assignments[self.assignments > -1])
        
        self.stepsize = stepsize
        self.timestep = timestep
        logger.debug("Dimensionality %d",self.ndim)

    def  get_clustermembers(self, clid):
        """ returns cluster members of a given clusterid
        """
        
        return self.cltags[:,0][np.where(self.cltags[:,1] == clid)]

    def  get_traj_and_frnos(self, frids):
        '''
        Return the trajectory number and timestamp of given frames
        
        '''
        
        
        # get the trajectory ids, to which these frame belongs
        cumlengths = np.cumsum(self.trajs_lengths)
        trajnos = np.digitize(frids, cumlengths)
        
        timefrs = []
        for i in range(trajnos.size):
            #frno = (frids[i] - self.trajs_lengths[:trajnos[i]].sum()) * self.stride
            frno = (frids[i] - self.trajs_lengths[:trajnos[i]].sum()) 
            frno = int(frno)
            logger.debug("Frid %d : Traj %s -> frame number %d" % (frids[i],
                                            self.trajnames[trajnos[i]], frno))
            
            # to convert frno to step number multiply by step_per_obs
            stepno = self.stepsize * frno
            time = self.timestep * frno
            timefrs.append(time)
            logger.debug("xtc:%s stepno: %d time: %f",
                          self.trajnames[trajnos[i]], stepno, time)
    
        return trajnos, np.array(timefrs, dtype=np.int)


#     def invert_assignments(self):
#         '''
#         return dictonary with key as cluster number and 
#         list with trajnos, and corroponding trajectory time frame
#         as two lists
#         '''
#             
#         clust = {}
#         for i in range(-1, self.assignments.max() + 1):
#             clust[i] = [[], []]
#         for i in range(self.ntrajs):
#             for j in range(self.trajs_lengths[i]):
#                 key = self.assignments[i, j]
#                 trajno=i
#                 clust[key][0].append(trajno)
#                 timefr = j * self.timestep * self.stride
#                 #timefr = j * self.timestep 
#                 clust[key][1].append(timefr)
#         return clust
        
    def get_clustercenters(self, outf="clcenters.xtc"):
        
        '''
        Function to retrive frames belonging to cluster centers
        '''
                
        trajnos, timefrs = self.get_traj_and_frnos(self.centids)
        gx = Gmxtc()
        gxout = Gmxtc()
        
        gxout.open_xtc(outf, "w")
        for i in range(trajnos.size):
            gx.read_timeframe(self.trajnames[trajnos[i]], time=timefrs[i])
            gxout.copy(gx)
            gxout.time=c_float(i)
            gxout.step=c_int(i)
            gxout.write_xtc()
        gxout.close_xtc()
             
                   
    
    def get_cluster(self, clid, nconfs=0,flag_nopreprocess = False):
        '''Get aligned clustermembers back'''
     
        #nodesizes = np.bincount(self.assignments[self.assignments > -1])

        #clusts = self.invert_assignments()
        
        gx = Gmxtc()
        gxout = Gmxtc()
        
        # # first load the centroid
        trajno, timefr = self.get_traj_and_frnos([self.centids[clid]])
        logger.debug("Loading centerid %s",clid)
        gx.read_timeframe(self.trajnames[trajno[0]], time=timefr[0])
        data1 = gx.x_to_array()
        traj1 = data1.reshape(1, data1.shape[0], 3)
        
        # Get all members of cluster
        clmembers = self.get_clustermembers(clid)
        trajnos, timefrs = self.get_traj_and_frnos(clmembers)

        # I need box information for writing xtc
        traj2 = []
        boxs, times = [], []
        boxs.append(gx.box)
        times.append(c_float(0))
        prec = gx.prec
        random = np.random
        
        if nconfs != 0:
            randomindices = random.permutation(self.nodesizes[clid])[:nconfs]
        else:
            randomindices = np.arange(self.nodesizes[clid])
        outf = "clid_%s.xtc" % clid
    
        logger.debug("Loading Rest of  clid %s members",clid)
        # ##Now load the rest
        for i,rindex in enumerate(randomindices):
            trajno, timefr = trajnos[rindex], timefrs[rindex]
            gx.read_timeframe(self.trajnames[trajno], time=timefr)
            data2 = gx.x_to_array()
            traj2.append(data2.reshape(1, data2.shape[0], 3))
            boxs.append(gx.box)
            gx.time=c_float(i)
            times.append(c_float(i+1))
            # prec=(data2[3])
            
        #join traj1 and 2
        traj2 = np.vstack(traj2)
        traj = np.vstack([traj1, traj2])
        
        # # Hack to use container class so that I can pass data to metric class
        # # TODO: need a better solution
        globalIDs=np.arange(len(traj))
        localIDs = np.copy(globalIDs)
        
        traj_container = Framecollection(globalIDs = globalIDs,
                                   localIDs = localIDs,
                                   frames = traj)
        
        metric = self.Metric(tpr_filepath=self.tprf, 
                                ndx_filepath=self.ndxf, 
                                stx_filepath = self.grof,
                                number_dimensions=self.ndim)
        if flag_nopreprocess == False:
            logger.info("will do preprocessing")
            metric.preprocess(
                    frame_array_pointer = traj_container.get_first_frame_pointer(),
                    number_frames = traj_container.number_frames,
                    number_atoms = traj_container.number_atoms)
        
        rmsd = np.zeros(len(traj), dtype=np.float32)
        metric.fit_trajectory(traj_container, 0, rmsd)
        rmsdoutf=outf.replace(".xtc","_rmsd.txt")
        np.savetxt(rmsdoutf, rmsd)
        gxout.write_array_as_traj(outf, traj_container.frames, boxs, times, prec)


    def print_clustersizes(self):
        np.savetxt("clsize.txt",self.nodesizes)


    def _get_assignments(self):
        """ 
        split clusters per trajectory and add them as numpy array
        of dimension(ntraj,maxtrajlength)
        
        copied from msmbuilder    
    
        """
        assgn_list = split(self.cltags[:,1], self.frames_per_traj)
        output = -1 * np.ones((len(self.trajs_lengths), max(self.trajs_lengths)), dtype='int')
        for i, traj_assign in enumerate(assgn_list):
            output[i][0:len(traj_assign)] = traj_assign
        return output
        
############TO BE IMPLIMENTED###############################################
    def build_network(self, nodes, nodesizes, weights,
                      outfile = 'network.dot', nodeprop=None):
        '''
        Not working yet
        Nodes represents the states
        '''
        nodes=np.array(nodes)
        import networkx as nx
        G = nx.Graph()
        
        if nodeprop is not None:
            for i in range(nodes.size):
                G.add_node(int(nodes[i]), size=int(nodesizes[i]),
                            label=int(nodes[i]), prop=nodeprop[i])
        else:
            for i in range(nodes.size):
                G.add_node(int(nodes[i]), size=int(nodesizes[i]),
                            label=int(nodes[i]))
             
        for i in range(weights.shape[0]):
            w = weights[i, 2]
            G.add_edge(int(weights[i, 0]), int(weights[i, 1]),
                        weight=float(w))
            
        nx.write_dot(G, outfile)
        outfile1 = outfile.replace('.dot', '.gml')
        nx.write_gml(G, outfile1)
     
    def get_dmatrix(self,xtcfile, nframes, flag_nopreprocess = False):
        """
        Not working yet
 
            takes xtcfile as input and returns half matrix of comparisons
        """
        # # Get input data
        gx = Gmxtc()
        no_preprocess = False

        stride = 1
        bPBC = None
        traj = gx.load_traj(xtcfile, stride, bPBC, nframes)
        natoms = traj.shape[1]
        nframes = traj.shape[0]
        # # TODO: need a better solution
        globalIDs=np.arange(len(traj))
        localIDs = np.copy(globalIDs)
        
        traj_container = Framecollection(globalIDs = globalIDs,
                                   localIDs = localIDs,
                                   frames = traj)
        metric = self.Metric(tpr_filepath=self.tprf, 
                                ndx_filepath=self.ndxf, 
                                stx_filepath = self.grof,
                                number_dimensions=self.ndim)
        
        if flag_nopreprocess == False:
            logger.info("will do preprocessing")
            metric.preprocess(
                    frame_array_pointer = traj_container.get_first_frame_pointer(),
                    number_frames = traj_container.number_frames,
                    number_atoms = traj_container.number_atoms)
        
        
        rmsd = np.zeros(len(traj), dtype=np.float32)
        weights = []
        for k in range(nframes - 1):
            self.metric.fit_trajectory(traj_container, k, rmsd)
            for j in range(k+1, rmsd.size):
                weights.append((k,j,rmsd[j]))
                
        weights = np.array(weights)
        return weights
     
    def build_network_rmsd(self,centxtcfile,nframes):
        ''' '''
        
        weights = self.get_dmatrix(centxtcfile, nframes)
        # ##Get Populations
        matrixfn = "rmsdmatrix.npy"
        np.save(matrixfn, weights)
        
        for i in range(weights.shape[0]):
            w = weights[i, 2]
            
            if w < 0.0001:
                logger.warning("Extremly low RMSD found!!! %d %s ", i,
                                weights[i, :])
                raise SystemExit("Exiting")
            else:
                w = 1.0 / w
                weights[i,2] = w
                
        nodenames = [str(i) for i in range(self.centids.size)]
        self.build_network(nodenames, self.nodesizes, weights)

    def build_network_count(self,countmatrixfn,outfile="countnetwork.dot"):
        ''' '''
        
        nodeA,nodeB,counts = np.loadtxt(countmatrixfn)
        weights = np.array((nodeA,nodeB,counts),dtype=np.int).transpose()
        # ##Get Populations
        nodenames = [str(i) for i in range(self.centids.size)]
        self.build_network(nodenames, self.nodesizes, weights,outfile=outfile)

#############################################################################################
    def write_msm_output(self,outarray=None,filename="Assignments.h5"):
        from msmbuilder import io
        if outarray is None:
            outarray = self.assignments
        io.saveh(filename,outarray)
        
    def dump_count_matrix(self,assignfn,lagtime=1,outfn="count_matrix.txt"):
        from msmbuilder import io
        from msmbuilder import MSMLib
        
        assignments = io.loadh(assignfn, 'arr_0')
        # returns sparse lil_matrix
        counts = MSMLib.get_count_matrix_from_assignments(assignments, lag_time=lagtime,
                                                          sliding_window=True)
        counts = counts.tocoo()
        np.savetxt(outfn,(counts.row, counts.col, counts.data))
