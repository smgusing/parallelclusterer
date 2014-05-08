# for python 2

# built-in modules
import os.path
import ctypes

# external modules.
import numpy as np
import gp_grompy
import parallelclusterer
from parallelclusterer import cmetric 
import logging
# Instantiate helper classes.
gmstx = gp_grompy.Gmstx()
gmndx = gp_grompy.Gmndx()

logger = logging.getLogger(__name__)



# ================================================================

class Metric():
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

        # --------------------------------
        # Declare all fields.
        # No method or instantiator should create any other fields.
        self.topology = None
        self.index = None

        self.rms_weights = None
        self.rms_indices = None
        self.rms_weights_ptr = None
        self.rms_indices_ptr = None
        self.rms_size = None

        self.fitting_weights = None
        self.fitting_indices = None
        self.fitting_weights_ptr = None
        self.fitting_indices_ptr = None
        self.fitting_size = None

        self.number_dimensions = None

        # --------------------------------
        # Read Gromacs index file, topology file.
        # Obtain atom masses (atom masses are used as weights in rmsd-fitting).
        # Select subsets of to consider for fitting (interactive mode).

        # Set number of dimensions.
        self.number_dimensions = number_dimensions
        
        # Read index file.
        number_of_index_groups = 2 # Ask for two index groups: ...
        gmndx.read_index(ndx_filepath, number_of_index_groups) # Runs interactive mode for input. (GROMCAS)

        # ...The first group specifies the atoms between which the rmsd is computed.
        self.rms_size = gmndx.isize[0]

        # ...The second group specifies the atoms of frames which will be aligned in rotation fitting.
        self.fitting_size = gmndx.isize[1]

        logger.debug("Fitting group size: %s Rms Group size: %s",self.fitting_size,self.rms_size)
        # Copy indices, atom masses from the GROMACS files into our arrays...
        self.rms_indices = np.empty(self.rms_size, dtype=np.int32)
        self.fitting_indices = np.empty(self.fitting_size, dtype=np.int32)

        # fill up indices to our arrays
        for i in xrange(self.rms_size):
            j = gmndx.index[0][i]
            self.rms_indices[i] = j
            
        for i in xrange(self.fitting_size):
            j = gmndx.index[1][i]
            self.fitting_indices[i] = j
            
        if os.path.isfile(tpr_filepath):
            # Read topology file.
            gmstx.read_tpr(tpr_filepath)
            
            self.rms_weights = np.zeros(gmstx.natoms, dtype=np.float32)
            self.fitting_weights = np.zeros(gmstx.natoms, dtype=np.float32)
            
            # ... for group 1 (rmsd)
            for i in xrange(self.rms_size):
                j = self.rms_indices[i]
                self.rms_weights[j] = gmstx.top.atoms.atom[j].m # (m)ass
    
            # ... for group 2 (rotation fitting)
            for i in xrange(self.fitting_size):
                j = self.fitting_indices[i]
                self.fitting_weights[j] = gmstx.atoms.atom[j].m # (m)ass

        else:
            logger.info("No tprfile found will use mass of unity and grofile")
            gmstx.read_stx(stx_filepath)
            self.rms_weights = np.ones(gmstx.natoms, dtype=np.float32)
            self.fitting_weights = np.ones(gmstx.natoms, dtype=np.float32)

        # Create pointers to the newly filled arrays.
        self.create_pointers()


    # ----------------------------------------------------------------
    # Since we cannot transfer over mpi4py objects which contain C-pointers,
    # these are methods to destroy and create pointers to internal arrays
    # for before and after transfer over mpi4py, respectively.
    # Also provided is an example function which does this, using broadcasting as an example.

    def create_pointers(self):

        self.fitting_indices_ptr = self.fitting_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.fitting_weights_ptr = self.fitting_weights.ctypes.data_as(ctypes.POINTER(gp_grompy.c_real))

        self.rms_indices_ptr = self.rms_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.rms_weights_ptr = self.rms_weights.ctypes.data_as(ctypes.POINTER(gp_grompy.c_real))


    def destroy_pointers(self):

        self.fitting_indices_ptr = None
        self.fitting_weights_ptr = None

        self.rms_indices_ptr = None
        self.rms_weights_ptr = None


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
        



