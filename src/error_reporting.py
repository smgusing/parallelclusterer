import ctypes
import numpy as np
import itertools
import logging

import gp_grompy
import daura_clustering
logger = logging.getLogger(__name__)


def analyze_errorno1(comm, metric, my_frames, center_frame, 
                     center_degree, center_id, clusters, old_removed_vertices, cutoff):
    ''' Do detailed analysis on error where number of members returned do not 
    match degree of the center. Note that this will be executed by all mpi processes
    
    1. Get all members belonging to the center
    2. Remove the members that are in other clusters
    3. Compare degrees and print
    
    Input parameters:
        
        center_frame: ndarray
        
        center_degree: int
        
        center_id: int
        
        my_frames: object of framecollection
        
        metric: object of gmx_metric
        
        cutoff: float
    '''
    my_rank = comm.Get_rank()
    
    checkpoint_filepath = ".daura_first.checkpoint"
    ckp_counts, ckp_clusters,ckp_removed_vert = daura_clustering.read_checkpoint(checkpoint_filepath)
    
    # Center frame is already broadcasted at this point
    center_frame_pointer = center_frame.ctypes.data_as(ctypes.POINTER(gp_grompy.rvec))
    rmsd_buffer = np.empty(my_frames.number_frames, dtype=my_frames.frames.dtype)
   
    # Reset masks   
    mask_removed_vertices = np.zeros(my_frames.number_frames, dtype=np.int32)
    mask_removed_vertices_ptr = mask_removed_vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    metric.compute_distances( 
       reference_frame_pointer = center_frame_pointer,
       frame_array_pointer     = my_frames.get_first_frame_pointer(),
       number_frames           = my_frames.number_frames,
       number_atoms            = my_frames.number_atoms,
       real_output_buffer      = rmsd_buffer, # compute_distances writes results to this buffer.
       mask_ptr                = mask_removed_vertices_ptr,
       mask_dummy_value        = -1.0,
       )
   
 
    fst = lambda x: x[0]
    existsAndWithinCutoff = lambda x: (0.0 <= x[1] <= cutoff)
    
    my_members = map(fst, filter(existsAndWithinCutoff,
                    zip(my_frames.globalIDs_iter, rmsd_buffer)))
    # (note: using globalIDs here is necessary because we allow striding in the input frames)
    
    # Broadcasting of members.
    members_gathered = comm.allgather(my_members)
    members = set(list(itertools.chain(*members_gathered)))
    
    if len(members) == ckp_counts[center_id]:
        logger.debug("Number of counts before clustering matches")
    else:
        logger.debug("Number of counts in chkpoint file %s do not match member retreived %s",
                     ckp_counts[center_id],len(members))
        raise SystemExit('Error in counts at begining. Will exit now ')
    
    
    
    # load number of members from checkpoint and compare
    # if correct. Make decr list and compare counts with updated counter.
    
    

    if my_rank == 0: 
        logger.debug("Frame %s, Degree before removal %s",center_id, len(members))
        # sort clusters based on len(members)
        sorted_keys = sorted(clusters, key=lambda k: len(clusters[k]), reverse=True )
        
        for key in sorted_keys:
            logger.debug("members %s", members)
            common_memb = set(clusters[key]) & members
            members = members - common_memb
            
            logger.debug("Center %s: Total members %s,Common_members %s",len(clusters[key]), key, len(common_memb))
            logger.debug("Degree remaining %s", len(members))
        
        logger.debug("old degrees %s, new degrees %s", center_degree, len(members) )
        logger.debug("Final Members %s",members)
            
        
        
        
        
          
   
    
    