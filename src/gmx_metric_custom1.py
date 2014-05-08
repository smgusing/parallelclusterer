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
    """
    A Metric class for clustering.
    The metric between frames: least-rmsd allowing for rotation and translation.

    Metric classes must implement:
    - preprocess()
    - oneToMany_distance()
    - oneToMany_countWithinCutoff() # because it's faster (and also parallel) in C.

    Notes:
    - Ctypes knows how to do basic conversions, like converting Python floats/ints to C floats/ints.
    - I'm not yet thinking about making an interface for Metric classes,
      so then I'm creating whatever distance analysis methods I want!
      Especially since it's difficult to do things externally of C functions,
      since Python is both slow and single-threaded.
    """
    
    def __init__(self, tpr_filepath=None, stx_filepath=None, ndx_filepath=None, number_dimensions=3):
        """
        Obtains all information about the frames that is required to perform the metric computation.
        No other information is gathered at a later time.
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
                            # making new arguments have default values to prevent breaking
                            # but we should eventually properly update all calls to the function
        
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
                                # making new arguments have default values to prevent breaking
                                # but we should eventually properly update all calls to the function

        cutoff_c = gp_grompy.c_real(cutoff)
        int_output_buffer_ptr = int_output_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        count = cmetric.oneToMany_countWithinRmsd( # C function.
            cutoff_c, int_output_buffer_ptr, reference_frame_pointer, frame_array_pointer,
            number_frames, number_atoms, self.number_dimensions, self.fitting_weights_ptr,
            self.rms_indices_ptr, self.rms_weights_ptr, self.rms_size,
            mask_ptr)

        return count
    
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
        



