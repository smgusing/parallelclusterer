# for python 2

# built-in modules
import ctypes

# external modules.
import gp_grompy
from parallelclusterer.gmx_metric import Gmx_metric
from parallelclusterer import _gmxrmsd as cmetric 
import logging
# Instantiate helper classes.
gmstx = gp_grompy.Gmstx()
gmndx = gp_grompy.Gmndx()

logger = logging.getLogger(__name__)



# ================================================================

class Gmx_metric_rmsd(Gmx_metric):
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
        super(Gmx_metric_rmsd,self).__init__(tpr_filepath,stx_filepath,ndx_filepath,number_dimensions)
        


    # ================================================================
    # Methods for computation of the metric.

    def preprocess(self, frame_array_pointer, number_frames, number_atoms):
        """
        Translates a frame's atoms so that the center of mass is the origin.
        (In-place modification.)
        """

        cmetric.parallelFor_removeCenterOfMass( # C function.
            frame_array_pointer, number_frames, number_atoms, self.number_dimensions,
            self.fitting_size, self.fitting_indices_ptr, self.fitting_weights_ptr)


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

    def count_neighbours_between(self, cutoff, frame_array0_pointer, frame_array1_pointer, 
            frame_array0_number, frame_array1_number, frame_array0_idx,
            frame_array1_idx,frame_array0_count,frame_array1_count,
            frame_array0_idxsize,frame_array1_idxsize, number_atoms):
            # making new arguments have default values to prevent breaking
            # but we should eventually properly update all calls to the function

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
        



