"""

"""

# for Python2

# External modules.
from mpi4py import MPI
import gp_grompy
import txtreader
# Built-in modules.

import ctypes
import numpy as np
import itertools
import cPickle

from time import time       
import logging
# Project modules.

from project import Project
from framecollection import Framecollection
from loadmanager import Loadmanager

from error_reporting import analyze_errorno1
 
logger = logging.getLogger(__name__)

# ================================================================
# Helper functions which I found that I needed
# which, later, may need to be put into some classes.

lfunc = {"INFO":logger.info,"DEBUG":logger.debug,
              "WARN":logger.warning,"ERROR":logger.error}
    
def print0(msg='',rank=0,msgtype="INFO"):
    
    if rank == 0:
        try:
            lfunc[msgtype](msg)
        except KeyError:
            print "%s not a valid msgtype"%msg
    else: 
        None


# ================================================================
# Clustering Main Function.

def cluster(Metric, project_filepath, cutoff, checkpoint_filepath=None,
            flag_nopreprocess = False):
    """ Cluster data
    
    Parameters
    ---------- 
    project_filepath : String
       The path to the YAML project file.
       
    cutoff           : Floating 
        The cutoff distance passed to the Metric class.
        
    chekpoint_filepath : string,optional
    flag_nopreprocess : bool,optional
        switches off preprocessing
    
    Returns
    -------
    clusters: dict
        a map (dictionary) from cluster center vertices C to lists of vertices,
        where the lists represent the set of vertices belonging to the cluster with center C.
        i.e. a map: FrameID -> [FrameID]
        
    """

    # ================================================================
    # Instantiation of helper classes.
    
    # Initialize MPI.
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    my_rank = comm.Get_rank()
    comm.Barrier()
    # Print only at node 0.
#     def my_print(x):
#         print x
#     print0 = lambda x: my_print(x) if my_rank == 0 else None

    # Say hello.
    print0(rank=my_rank,msg="Initialized MPI.")
    logger.debug("Hello, from node %s",my_rank)

    logger.info("Reading project file at node %s",my_rank)
    project = Project(existing_project_file = project_filepath)
    
    logger.debug("Initializing manager at node %s",my_rank)
    manager = Loadmanager(project.get_trajectory_lengths(),
                          project.get_trajectory_filepaths(),
                          mpi_size,my_rank)
    # metric has to be instantiated by only one mpi process as it needs user input
    # ie index groups
    if my_rank == 0:
         
        metric = Metric(tpr_filepath = project.get_tpr_filepath(),
                       stx_filepath = project.get_gro_filepath(),
                       ndx_filepath = project.get_ndx_filepath(),
                       number_dimensions = project.get_number_dimensions() )
        # since we have to broadcast it we need to destroy all pointers to arrays
        metric.destroy_pointers()
        logger.debug("Metric initialized at node 0")
    else:
        metric = None
    metric =  comm.bcast(metric, root = 0)
    print0(rank=my_rank,msg="metric object broadcasted.")
    # recreate all pointers in the object's instance
    metric.create_pointers()
    
    manager.do_partition()
    
    
    # Take work share.
    my_partition = manager.myworkshare
    
    (my_trajectory_filepaths, my_trajectory_lengths, \
     my_trajectory_ID_offsets, my_trajectory_ID_ranges) = \
     map(list, my_partition)


    #print0(my_rank,"\tDistribution: {0}".format(frame_globalID_distribution))
    logger.info("Reading trajectories at %s",my_rank)
    my_frames = Framecollection.from_files(
            stride = project.get_stride(),
            trajectory_type = project.get_trajectory_type(),
            trajectory_globalID_offsets = my_trajectory_ID_offsets,
            trajectory_filepath_list = my_trajectory_filepaths,
            trajectory_length_list = my_trajectory_lengths, )

        
    if flag_nopreprocess == False:
        # ----------------------------------------------------------------
        # Preprocess trajectories (modifying them in-place).
        # Metric preprocessing.
        logger.debug(" Preprocessing trajectories at rank %s",my_rank)
        metric.preprocess( frame_array_pointer = my_frames.get_first_frame_pointer(),
                           number_frames = my_frames.number_frames,
                           number_atoms = my_frames.number_atoms)
    else:
        
        print0(rank=my_rank,msg=" Will not preprocess trajectories")


    # ================================================================
    # Initial round of all-to-all neighbour counting.


    # Count the number of neighbours for all frames.
    # If frames are vertices and edges join frames having rmsd within the cutoff,
    # then we compute and record the degree of each vertex.

    if checkpoint_filepath is None:
        print0(rank=my_rank,msg="Counting 'neighbours' for all frames.")

        my_neighbour_count = allToAll_neighbourCount(cutoff, comm, mpi_size, my_rank,
                                metric, my_frames,manager) # :: Map Integer Integer

        print0(rank=my_rank,msg="Synchronizing neighbour counts.")
        neighbour_count_recvList = comm.allgather(my_neighbour_count)

        neighbour_counts = {}
        for node_neighbour_counts in neighbour_count_recvList:
            for frameID in node_neighbour_counts:
                try:
                    neighbour_counts[frameID] += node_neighbour_counts[frameID]
                except KeyError:
                    neighbour_counts[frameID]  = node_neighbour_counts[frameID]
    else :
        print0(rank=my_rank,msg="Using checkpoint file.")
        neighbour_counts = None



    print0(rank=my_rank,msg="Start clustering.")
    
    T=time()

    clusters = daura_clustering(neighbour_counts,
                    cutoff, comm, mpi_size, my_rank, manager, 
                    metric, my_frames, checkpoint_filepath)
    
    print0(rank=my_rank,msg=" Finished ... Total time: {0}".format(time()-T))

                    
    return clusters



# --------------------------------------------------------------------------------------------------------

def allToAll_neighbourCount(cutoff, comm, mpi_size, my_rank, metric, my_frames, manager):
    """ Does initial all to all comparisions
    
    The load is divided differently based on whether mpi_size is even or odd
    
    - In case of odd:
        --This is simple, first each node compare to itself.
          then to its kth neighbour where k=1 to mpi_size/2
    - In case of even:
        -- one node of smaller rank does both self comparisons, while the other node
           does the outer neigbour count.
    
    """

    # Init for my frames.
    my_neighbour_count = dict(( (x, 0) for x in my_frames.globalIDs_iter ))
    

    print0(rank=my_rank,msg=" All-to-all: Round %s of %s"%(1, (mpi_size+1)/2))

    T = time()
    # if odd
    if mpi_size % 2 == 1:
        logger.debug("Single Inner neighbour counting at rank %s",my_rank)
        innerNeighbourCount(metric, cutoff, my_neighbour_count, my_frames)
        logger.debug("Finished Single Inner neighbour counting at rank %s",my_rank)

    else:
        my_received_frames = Framecollection()
        mpi_send_lambda = my_frames.for_sending(comm)
        mpi_recv_lambda = my_received_frames.for_receiving(comm)

        ring_schedule_k = manager.make_ring_schedule(mpi_size/2)
        
        ## This should be abstracted ie collection class must have send and receive
        ## implmented to work with manager
        manager.run_schedule(ring_schedule_k, my_rank,
                                                mpi_send_lambda, mpi_recv_lambda)

        other_rank = (my_rank + mpi_size/2) % mpi_size
        
        # One with the lower rank does self comparisons for both
        if my_rank < other_rank:
            logger.debug("Two Inner neighbour counting at rank %s",my_rank)
            innerNeighbourCount(metric, cutoff, my_neighbour_count, my_frames)
            innerNeighbourCount(metric, cutoff, my_neighbour_count, my_received_frames)
            logger.debug(" Finished Two Inner neighbour counting at rank %s",my_rank)
        else:
            logger.debug(" Outer neighbour counting at rank %s",my_rank)
            outerNeighbourCount(metric, cutoff, my_neighbour_count, my_frames, my_received_frames)
            logger.debug(" Finished Outer neighbour counting at rank %s",my_rank)

    for k in xrange(1, (mpi_size+1)/2):
        T = time()
        my_received_frames = Framecollection()
        mpi_send_lambda = my_frames.for_sending(comm)
        mpi_recv_lambda = my_received_frames.for_receiving(comm)


        ring_schedule_k = manager.make_ring_schedule(k)
        manager.run_schedule(ring_schedule_k, my_rank,
                                                mpi_send_lambda, mpi_recv_lambda)

        logger.debug("Outer neighbour counting at rank %s round %d",my_rank,k)
        outerNeighbourCount(metric, cutoff, my_neighbour_count, my_frames, my_received_frames)
        
    logger.info("all-to-all:rank:%s, time: %s",my_rank,time()-T)

    return my_neighbour_count

def innerNeighbourCount(my_metric, cutoff, neighbour_count_dict, the_frames):
    """ Count neighbours on a given node.
    
    .. NOTE:: Inplace update of neighbour_count_dict
    
    """
    
    frame_array0_idx = np.arange(the_frames.number_frames,dtype=np.int32)
    the_count_buffer = np.zeros(the_frames.number_frames,dtype=np.int32)

    my_metric.count_neighbours_within( 
        cutoff                  = cutoff,
        frame_array0_pointer    = the_frames.get_first_frame_pointer(),
        frame_array0_number     = the_frames.number_frames,
        frame_array0_idx        = frame_array0_idx,
        frame_array0_count      = the_count_buffer,
        frame_array0_idxsize    = the_frames.number_frames,
        number_atoms            = the_frames.number_atoms )

    for the_frameID, the_count in zip(the_frames.globalIDs_iter, the_count_buffer):
        # Because no frame is compared with itself, so add it here.
        try:
            neighbour_count_dict[the_frameID] += the_count + 1
             
        except KeyError:
            neighbour_count_dict[the_frameID]  = the_count + 1
     

def outerNeighbourCount(my_metric, cutoff, neighbour_count_dict, my_frames, my_received_frames):
    """ count neigbours between two frame collections
    
    .. NOTE:: Inplace update of neighbour_count_dict
    
    """

    frame_array0_idx = np.arange(my_frames.number_frames,dtype=np.int32)
    frame_array1_idx = np.arange(my_received_frames.number_frames,dtype=np.int32)
    
    frame_array0_count = np.zeros(my_frames.number_frames,dtype=np.int32)
    frame_array1_count = np.zeros(my_received_frames.number_frames,dtype=np.int32)

    my_metric.count_neighbours_between( 
        cutoff                  = cutoff,
        frame_array0_pointer = my_frames.get_first_frame_pointer(),
        frame_array1_pointer    = my_received_frames.get_first_frame_pointer(),
        frame_array0_number     = my_frames.number_frames,
        frame_array1_number     = my_received_frames.number_frames,
        frame_array0_idx        = frame_array0_idx,
        frame_array1_idx        = frame_array1_idx,
        frame_array0_count      = frame_array0_count,
        frame_array1_count      = frame_array1_count,
        frame_array0_idxsize     = my_frames.number_frames,
        frame_array1_idxsize     = my_received_frames.number_frames,
        number_atoms            = my_frames.number_atoms )

    for array0_frameID, array0_count in zip(my_frames.globalIDs_iter, frame_array0_count):
        try:
            neighbour_count_dict[array0_frameID] += array0_count
        except KeyError:
            neighbour_count_dict[array0_frameID]  = array0_count
    

    for array1_frameID, array1_count in zip(my_received_frames.globalIDs_iter, frame_array1_count):
        try:
            neighbour_count_dict[array1_frameID] += array1_count
        except KeyError:
            neighbour_count_dict[array1_frameID]  = array1_count
    

# --------------------------------------------------------------------------------------------------------

def daura_clustering(neighbour_counts, cutoff, comm, mpi_size, my_rank, manager,
                     metric, my_frames, checkpoint_filepath):
    
    ''' Impliments Daura clustering.
    
    - Define the largest cluster to be the set of vertices adjacent to the vertex with maximum degree.
    - Clustering is performed by identifying the largest cluster,
      assigning its vertices to one cluster, removing its vertices from the graph,
      and repeating the process.
      
     Returns
     -------
     Clusters: dict
        a map (dictionary) from cluster center vertices C to lists of vertices,
        where the lists represent the set of vertices belonging to the cluster with center C.
        i.e. a map: FrameID -> [FrameID]
           
      
    '''




    # Extract container metadata.
    my_localIDs  = my_frames.global_to_local




    # Initialization.
    """
    - removed_vertices:
        a set of globalIDs of vertices that have been assigned to a cluster
    - mask_removed_vertices:
        a list (numpy_array) recording, for each vertex in my_frames, whether it has been
        assigned to a cluster, where
            - a zero in the i'th index represents that the vertex with localID i has not been assigned
                and non-zero value represents that the vertex has been assigned.
            - localIDs are with respect to the container,
                i.e. they are determined by the container's localIDs mapping.
        (perhaps this should be merged into the container class?)
    """
    mask_removed_vertices = np.zeros(my_frames.number_frames, dtype=np.int32)
    mask_removed_vertices_ptr = mask_removed_vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    if checkpoint_filepath is None:
        neighbour_counts = dict(neighbour_counts) # copy dict

        clusters = {}
        removed_vertices = set()
        if my_rank == 0:
            logger.info( "writing checkpoint file at rank %s",my_rank)
            write_checkpoint(".daura_first.checkpoint", neighbour_counts, clusters, removed_vertices)

    else:
        neighbour_counts, clusters, removed_vertices = read_checkpoint(checkpoint_filepath)

        # Recreate mask.
        for v in removed_vertices:
            if v in my_localIDs:
                mask_removed_vertices[my_localIDs[v]] = 1

    # Do Clustering!
    while len(neighbour_counts) != 0:

        # Find and broadcast the vertex with highest degree -- the cluster center.
        # If two vertices both have maximum degree, choose the one with lowest frameID.

        center_degree = -1
        center_id = -1
        for frameID, degree in neighbour_counts.items():
            if (degree > center_degree) or ((degree == center_degree) and (frameID < center_id)):
                center_degree = degree
                center_id = frameID
                    
        print0(rank=my_rank,msg="Center Frame %s: Members %s"%(center_id,center_degree))
        center_host_node = manager.find_node_of_frame(center_id)
        if center_host_node is None:
            raise KeyError("Next cluster center ID not found within any node.")
            
            
        # Broadcasting of center.
        if my_rank == center_host_node:
            center_frame = my_frames.get_frame(center_id)
        else:
            shape = (my_frames.number_atoms, 3)
            center_frame = np.empty(shape, dtype=my_frames.frames.dtype)

        comm.Bcast([center_frame, my_frames.mpi_frametype ], root=center_host_node)
        

        # Find, broadcast the frames of, and record the cluster members,
        # that is, the vertices adjacent to the cluster center,

        # Find cluster members.
        center_frame_pointer = center_frame.ctypes.data_as(ctypes.POINTER(gp_grompy.rvec))
        rmsd_buffer = np.empty(my_frames.number_frames, dtype=my_frames.frames.dtype)

        metric.compute_distances( 
            reference_frame_pointer = center_frame_pointer,
            frame_array_pointer     = my_frames.get_first_frame_pointer(),
            number_frames           = my_frames.number_frames,
            number_atoms            = my_frames.number_atoms,
            real_output_buffer      = rmsd_buffer, # compute_distances writes results to this buffer.
            mask_ptr                = mask_removed_vertices_ptr,
            mask_dummy_value        = -1.0,
            )
        
        lower_bool = rmsd_buffer <= cutoff
        upper_bool = rmsd_buffer >= 0
        my_members =  my_frames.globalIDs[lower_bool * upper_bool]
        my_members =np.setdiff1d(my_members,removed_vertices,assume_unique = True)

        # (note: using globalIDs here is necessary because we allow striding in the input frames)

        # Broadcasting of members.
        members_gathered = comm.allgather(my_members)
        members = list(itertools.chain(*members_gathered))
        

        # Check consistency of data.
        if len(members) != center_degree:
            error_string = "Number of cluster members ({0}) & degree of the center vertex ({1}) are different".format(len(members), center_degree)
            print0(rank=my_rank,msg='members %s'%(members))
            logger.error(error_string)
            analyze_errorno1(comm, metric, my_frames, center_frame, 
                     center_degree, center_id, clusters, removed_vertices, cutoff)

            raise SystemExit("Exiting ")


        # Record new cluster, update 'removed_vertices' and 'mask_removed_vertices'.
        removed_vertices.update(members)

        for v in members:
            if v in my_localIDs:
                mask_removed_vertices[my_localIDs[v]] = 1
        
        clusters[center_id] = list(members)


        # Broadcast of the member frames of the new cluster.
        shape = (my_frames.number_atoms, 3) # 3 because rvec has type c_real[3]
        
        #Note: I Need to confirm that Allgather sequence is by rank order. It is assumed here
        member_frames_byNode = [ [None] * len(xs) for xs in members_gathered ]
        for i, frames_sublist in enumerate(member_frames_byNode):
            if my_rank == i:
                for j in xrange(len(frames_sublist)):
                    frames_sublist[j] = my_frames.get_frame(members_gathered[i][j])
            else:
                for j in xrange(len(frames_sublist)):
                    frames_sublist[j] = np.empty(shape, dtype=my_frames.frames.dtype)

        for i, frames_sublist in enumerate(member_frames_byNode):
            for j in xrange(len(frames_sublist)):
                comm.Bcast([frames_sublist[j], my_frames.mpi_frametype], root=i)

        frame_array = np.array(list(itertools.chain(*member_frames_byNode)))
        
        if len(frame_array) != len(members):
            logger.error("members retrieved %s . Expected %s", len(frame_array),len(members))
            raise SystemExit("Member retrieved do not match expected members ... Exiting on this")
        

        member_frames = Framecollection(frames = frame_array,
                                        globalIDs = np.array(members,dtype=np.int32),
                                        localIDs = np.arange(len(frame_array),dtype=np.int32),
                                        )
        # Remove the newly clustered vertices from the graph,
        # and update the degrees of the remaining vertices.

        # For the remaining vertices:
        # count the number of adjacent vertices that were newly clustered 

        
        
        # to get index of elements that are zero, and set astype to int32
        frame_array0_idx = np.nonzero(mask_removed_vertices == 0)[0].astype(np.int32)
        frame_array1_idx = np.arange(member_frames.number_frames,dtype=np.int32)
        
        count_buffer = np.zeros(my_frames.number_frames, dtype=np.int32)
        frame_array1_count = np.zeros(member_frames.number_frames,dtype=np.int32)
    
        metric.count_neighbours_between( 
            cutoff                  = cutoff,
            frame_array0_pointer = my_frames.get_first_frame_pointer(),
            frame_array1_pointer    = member_frames.get_first_frame_pointer(),
            frame_array0_number     = my_frames.number_frames,
            frame_array1_number     = member_frames.number_frames,
            frame_array0_idx        = frame_array0_idx,
            frame_array1_idx        = frame_array1_idx,
            frame_array0_count      = count_buffer,
            frame_array1_count      = frame_array1_count,
            frame_array0_idxsize     = frame_array0_idx.size,
            frame_array1_idxsize     = member_frames.number_frames,
            number_atoms            = my_frames.number_atoms )


        my_neighbours_decr = {}
        for i, frameID in enumerate(my_frames.globalIDs_iter):
            my_neighbours_decr[frameID] = count_buffer[i]


        # Broadcast neighbour decrements.
        neighbours_decr_gathered = comm.allgather(my_neighbours_decr)
        neighbours_decr = {}
        for x in neighbours_decr_gathered:
            neighbours_decr.update(x)


        # Remove, from the neighbour_counts map, the newly clustered nodes.
        for frameID in members:
            del neighbour_counts[frameID]


        # Update the degrees of the remaining vertices.
        for frameID in neighbours_decr.keys():
            try:
                neighbour_counts[frameID] -= neighbours_decr[frameID]
            except KeyError:
                pass


        # Write checkpoint file.
        if my_rank == 0:
            write_checkpoint(".daura_last.checkpoint",
                        neighbour_counts, clusters, removed_vertices)

        # if largest clustersize is one, then no need to cluster further
        if center_degree == 1:
            for frameID, degree in neighbour_counts.items():
                clusters[frameID] = [frameID]
                print0(rank=my_rank,msg="frame %s degree %s"%(frameID, degree))
                
                if degree > 1:
                    logger.error("node %s: frame %s degree %s is greater than expected",my_rank,frameID, degree)
                    raise SystemExit("Exiting on this")
                else:
                    del neighbour_counts[frameID]
        
        
        
    return clusters


# ================================================================
# Checkpoint file handling.

def write_checkpoint(filepath, neighbour_counts, clusters, removed_vertices):
    logger.debug("Writing %s",filepath)
    
    checkpoint_objects = (neighbour_counts, clusters, removed_vertices)

    with open(filepath, 'wb') as f:
        cPickle.dump(checkpoint_objects, f, protocol=2)

def read_checkpoint(filepath):
    logger.debug("Reading %s",filepath)
    with open(filepath, 'rb') as f:
        checkpoint_objects = cPickle.load(f)

    return checkpoint_objects


# ================================================================
# Something to change the formatting of the cluster data.

def cluster_dict_to_text(clusters, centers_filepath, clusters_filepath):
    """ write clusters and centers to given files
    
    """

    frames_to_clusters = {} # :: frameID -> clusterID
    clusterID = 0

    with open(centers_filepath, 'w') as f:
        for center in reversed( sorted(clusters.keys(), key=lambda x: len(clusters[x])) ):

            for frameID in clusters[center]:
                frames_to_clusters[frameID] = clusterID

            f.write("{0} {1}\n".format(clusterID, center))

            clusterID += 1

    
    with open(clusters_filepath, 'w') as f:
        for frameID in sorted(frames_to_clusters.keys()):
            f.write("{0} {1}\n".format(frameID, frames_to_clusters[frameID]))


    return None


def assign(Metric, project_filepath, cutoff, centerfile, flag_nopreprocess):
    ''' Method to add rest of the data to the cluster centers 
    '''
    # Initialize MPI.
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    my_rank = comm.Get_rank()
    comm.Barrier()
 
    # Print only at node 0.
#     def my_print(x):
#         print x
#     print0 = lambda x: my_print(x) if my_rank == 0 else None
    logger.info("refine start")
    # Say hello.
    print0(rank=my_rank,msg=" Initialized MPI.")
    logger.debug("Hello, from node %s",my_rank)
 
    # Load project file.
    print0(rank=my_rank,msg=" Reading project yaml file.")
    project = Project(existing_project_file = project_filepath)
 
    logger.debug("Initializing manager at node %s",my_rank)
    manager = Loadmanager(project.get_trajectory_lengths(),
                          project.get_trajectory_filepaths(),
                          mpi_size,my_rank)

    # Instantiate Metric class.
    if my_rank == 0:
        metric = Metric(tpr_filepath = project.get_tpr_filepath(),
                       stx_filepath = project.get_gro_filepath(),
                       ndx_filepath = project.get_ndx_filepath(),
                       number_dimensions = project.get_number_dimensions() )
        
        metric.destroy_pointers()
    else:
        metric = None
 
    metric = comm.bcast(metric, root=0)
    print0(rank=my_rank,msg="metric object broadcasted.")
 
    metric.create_pointers()
    
    manager.do_partition()
    # Take work share.
    my_partition = manager.myworkshare

 
    (my_trajectory_filepaths, my_trajectory_lengths, \
        my_trajectory_ID_offsets, my_trajectory_ID_ranges) = \
        map(list, my_partition)
 
    logger.info("Reading trajectories at %s",my_rank)
    my_frames = Framecollection.from_files(
            stride = 1,
            trajectory_type = project.get_trajectory_type(),
            trajectory_globalID_offsets = my_trajectory_ID_offsets,
            trajectory_filepath_list = my_trajectory_filepaths,
            trajectory_length_list = my_trajectory_lengths, )

        
    if flag_nopreprocess == False:
        # ----------------------------------------------------------------
        # Preprocess trajectories (modifying them in-place).
        # Metric preprocessing.
        logger.debug(" Preprocessing trajectories at rank %s",my_rank)
        metric.preprocess( frame_array_pointer = my_frames.get_first_frame_pointer(),
                           number_frames = my_frames.number_frames,
                           number_atoms = my_frames.number_atoms)
    else:
        
        print0(rank=my_rank,msg=" Will not preprocess trajectories")

         
    # ----------------------------------------------------------------
    # Preprocess trajectories (modifying them in-place).
    # Metric preprocessing.
#    print0(my_rank,"[Cluster] Preprocessing trajectories (for Metric).")
#    my_metric.preprocess(   frame_array_pointer = my_frames.get_first_frame_pointer(),
#                            number_frames = my_frames.number_frames(),
#                            number_atoms = my_frames.number_atoms(), )
 
    clustercenters = txtreader.readcols(centerfile)[:,1]
 
    clusters = {} # :: FrameID (cluster center) -> [FrameID] (the cluster -- its list of frames)
    my_unclustered = set([i for i in my_frames.globalIDs_iter])
    removed_vertices = set()
 
    for center_id in clustercenters:
        center_host_node = manager.find_node_of_frame(center_id)
        if center_host_node is None:
            raise KeyError("Next cluster center ID not found within any node.")
             
        
        # Broadcasting of center.
        if my_rank == center_host_node:
            center_frame = my_frames.get_frame(center_id)
        else:
            shape = (my_frames.number_atoms, 3)
            center_frame = np.empty(shape, dtype=my_frames.frames.dtype)
        
        comm.Bcast([center_frame, my_frames.mpi_frametype], root=center_host_node)
        if my_rank == 0 : logger.debug("Searching for members for center id %s", center_id)
        
        center_frame_pointer = center_frame.ctypes.data_as(ctypes.POINTER(gp_grompy.rvec))
        rmsd_buffer = np.empty(my_frames.number_frames, dtype=my_frames.frames.dtype)
        
        metric.compute_distances( 
            reference_frame_pointer = center_frame_pointer,
            frame_array_pointer = my_frames.get_first_frame_pointer(),
            number_frames = my_frames.number_frames,
            number_atoms = my_frames.number_atoms,
            real_output_buffer = rmsd_buffer, # writes results to this buffer.
            mask_ptr = None,
            mask_dummy_value = -1.0,
            )
         
        fst = lambda x: x[0]
        existsAndWithinCutoff = lambda x: (x[0] not in removed_vertices) and (0.0 <= x[1] <= cutoff)
        my_members = map(fst, filter(existsAndWithinCutoff,
                        zip(my_frames.globalIDs_iter, rmsd_buffer))) # for striding.
        
        # Broadcasting of members.
        members_gathered = comm.allgather(my_members)
        members = list(itertools.chain(*members_gathered))
        if my_rank == 0 : logger.debug("Found %s members",len(members))
        removed_vertices.update(members)
        clusters[center_id] = list(members)
        
        my_unclustered = my_unclustered.difference(set(members))
        
    unclustered_gathered = comm.allgather(my_unclustered)
    unclustered = list(itertools.chain(*unclustered_gathered))
    logger.debug("Unclustered %s", unclustered)
    
    for i in unclustered:
        clusters[i]=[i]

    logger.info("refine end")
         
    return clusters 
      
     
