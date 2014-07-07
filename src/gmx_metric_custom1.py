# for python 2

# built-in modules
import ctypes

# external modules.
import gp_grompy
from gmx_metric import Gmx_metric
from parallelclusterer import _gmxrmsd_custom1 as cmetric 
import logging
# Instantiate helper classes.
gmstx = gp_grompy.Gmstx()
gmndx = gp_grompy.Gmndx()

logger = logging.getLogger(__name__)



# ================================================================

class Gmx_metric_custom1(Gmx_metric):
    """ A Metric class for clustering.

    """
    
    def __init__(self, tpr_filepath=None, stx_filepath=None, ndx_filepath=None, number_dimensions=3):
        """
        Obtains all information about the frames that is required to perform the metric computation.
        No other information is gathered at a later time.
        
        The metric between frames: least-rmsd allowing for rotation and translation.
    
        """

        super(Gmx_metric_custom1,self).__init__(tpr_filepath,stx_filepath, ndx_filepath,number_dimensions)        


    @staticmethod
    def _exampleOnly_mpi_Bcast(metric_obj, comm, root):
        rank = comm.Get_rank()
    
        if rank == root:
            metric_obj.destroy_pointers()
        else:
            metric_obj = None
    
        metric_obj = comm.bcast(metric_obj, root=root)
        metric_obj.create_pointers()
    
        return metric_obj
        

    # ================================================================
    # Methods for computation of the metric.
    def compute_distances(self, reference_frame_pointer, frame_array_pointer,
                            number_frames, number_atoms, real_output_buffer,
                            mask_ptr=None, mask_dummy_value=0.0):
        ''' compute distances
        
        Parameters
        ---------------
        reference_frame_pointer : pointer
            pointer to reference frame
            
        frame_array_pointer : pointer
            pointer to traj array
            
        number_frames : int
            
        number_atoms : int
            
        real_output_buffer : array
            array collecting number of neighbours  
        
        mask_ptr : pointer
            pointer to array containing masked indicies
            
        mask_dummy_value :
        
        Return
        --------
        None
        
        Modifies real_output_buffer
        
        '''
        
        real_output_buffer_ptr = real_output_buffer.ctypes.data_as(ctypes.POINTER(gp_grompy.c_real))
        c_mask_dummy_value = gp_grompy.c_real(mask_dummy_value)

        cmetric.oneToMany_computeRmsd( # C function.
            reference_frame_pointer, frame_array_pointer, number_frames, number_atoms,
            self.number_dimensions, self.fitting_weights_ptr, self.rms_indices_ptr, self.rms_weights_ptr,
            self.rms_size, real_output_buffer_ptr,
            mask_ptr, c_mask_dummy_value)


    def count_number_neighbours(self, cutoff, reference_frame_pointer, frame_array_pointer, 
                                number_frames, number_atoms, int_output_buffer,
                                mask_ptr=None):
        ''' Count number of neighbours 
        
        '''
        

        cutoff_c = gp_grompy.c_real(cutoff)
        int_output_buffer_ptr = int_output_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        count = cmetric.oneToMany_countWithinRmsd( # C function.
            cutoff_c, int_output_buffer_ptr, reference_frame_pointer, frame_array_pointer,
            number_frames, number_atoms, self.number_dimensions, self.fitting_weights_ptr,
            self.rms_indices_ptr, self.rms_weights_ptr, self.rms_size,
            mask_ptr)

        return count
    
    def count_neighbours_between(self, cutoff, frame_array0_pointer, frame_array1_pointer, 
            frame_array0_number, frame_array1_number, frame_array0_idx,
            frame_array1_idx,frame_array0_count,frame_array1_count,
            frame_array0_idxsize,frame_array1_idxsize, number_atoms):
        
        ''' Count neighbours between two trajectories
        
        Return
        ----------
        None
        
        Modifies frame_array_0_count and frame_array_1_count
        
        '''

        cutoff_c = gp_grompy.c_real(cutoff)
        frame_array0_number_c = ctypes.c_int(frame_array0_number)
        frame_array1_number_c =ctypes.c_int(frame_array1_number)
        frame_array0_idx_ptr = frame_array0_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        frame_array1_idx_ptr = frame_array1_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        frame_array0_count_ptr = frame_array0_count.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        frame_array1_count_ptr = frame_array1_count.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        frame_array0_idxsize_c = ctypes.c_int(frame_array0_idxsize)
        frame_array1_idxsize_c = ctypes.c_int(frame_array1_idxsize)
        number_atoms_c         = ctypes.c_int(number_atoms)
        
        cmetric.manytomany_between(
            cutoff_c, frame_array0_pointer, frame_array1_pointer, 
            frame_array0_number_c, frame_array1_number_c,
            frame_array0_idx_ptr, frame_array1_idx_ptr,
            frame_array0_count_ptr, frame_array1_count_ptr,
            frame_array0_idxsize_c, frame_array1_idxsize_c,
            number_atoms_c, self.number_dimensions_c, self.fitting_weights_ptr,
            self.rms_indices_ptr, self.rms_weights_ptr, self.rms_size )

    def count_neighbours_within(self, cutoff, frame_array0_pointer, 
            frame_array0_number, frame_array0_idx,frame_array0_count,
            frame_array0_idxsize,number_atoms):
        ''' Count neighbours within a trajectory
        
        Return
        ---------
        None:
        
        Inplace modification of frame_array0_count
        
        
        '''

        cutoff_c = gp_grompy.c_real(cutoff)
        frame_array0_number_c = ctypes.c_int(frame_array0_number)
        frame_array0_idx_ptr = frame_array0_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        frame_array0_count_ptr = frame_array0_count.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        frame_array0_idxsize_c = ctypes.c_int(frame_array0_idxsize)
        number_atoms_c         = ctypes.c_int(number_atoms)
        
        cmetric.manytomany_within(
            cutoff_c, frame_array0_pointer,  
            frame_array0_number_c, 
            frame_array0_idx_ptr, 
            frame_array0_count_ptr, 
            frame_array0_idxsize_c, 
            number_atoms_c, self.number_dimensions_c, self.fitting_weights_ptr,
            self.rms_indices_ptr, self.rms_weights_ptr, self.rms_size )
    
    
    def fit_trajectory(self,traj_container,ref_frno,real_output_buffer):
        ''' Function to fit a given frame number to all the frames in a trajectory.
            The fitted trajectory is returned.
        
        '''
         
        real_output_buffer_ptr = real_output_buffer.ctypes.data_as(ctypes.POINTER(gp_grompy.c_real))
        frame_array_pointer=traj_container.get_first_frame_pointer()
        reference_frame_pointer=traj_container.get_frame_pointer(ref_frno)
        
        cmetric.distance_onetomany( # C function.
            reference_frame_pointer, frame_array_pointer, traj_container.number_frames, 
            traj_container.number_atoms, self.number_dimensions, self.fitting_weights_ptr,
            self.rms_indices_ptr, self.rms_weights_ptr,
            self.rms_size, real_output_buffer_ptr)
        



