import abc
import logging
import os
import numpy as np
import ctypes

import gp_grompy

logger = logging.getLogger(__name__)

gmstx = gp_grompy.Gmstx()
gmndx = gp_grompy.Gmndx()


class Gmx_metric(object):
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, tpr_filepath=None, stx_filepath=None, ndx_filepath=None, number_dimensions=3):
        ''' Base class for gromacs based rmsd calculations 
        
        Parameters
        ----------------
            tpr_filepath: str
                tpr file
            stx_filepath: str
                pdb or gro
            ndx_filepath: str
                ndx file
            
        
        '''
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
        self.number_dimensions_c = ctypes.c_int(number_dimensions)
        number_of_index_groups = 2 # Ask for two index groups: ...
        #
        if ndx_filepath is not None:
            if os.path.isfile(ndx_filepath):
                # Read index file.
                print ('''Select Two groups:first group specifies the atoms between which the rmsd is computed.\n
                 The second group specifies the atoms of frames which will be aligned in rotation fitting''')
                gmndx.read_index(ndx_filepath, number_of_index_groups) # Runs interactive mode for input. (GROMCAS)
            else:
                logger.error("cannot find %s",ndx_filepath)
                raise SystemExit("Quitting on this")
        else:
            logger.error("%s not set",ndx_filepath)
            raise SystemExit("Quitting on this")
            

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
        ''' Create pointers to numpy arrays for passing to c function
        
        Since we cannot transfer over mpi4py objects which contain C-pointers,
        these are methods to destroy and create pointers to internal arrays
        for before and after transfer over mpi4py, respectively.

        
        '''

        self.fitting_indices_ptr = self.fitting_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.fitting_weights_ptr = self.fitting_weights.ctypes.data_as(ctypes.POINTER(gp_grompy.c_real))

        self.rms_indices_ptr = self.rms_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.rms_weights_ptr = self.rms_weights.ctypes.data_as(ctypes.POINTER(gp_grompy.c_real))


    def destroy_pointers(self):
        ''' destroy pointers to numpy arrays.
            
            This is required. As objects with pointers cannot be passed around.
        '''

        self.fitting_indices_ptr = None
        self.fitting_weights_ptr = None

        self.rms_indices_ptr = None
        self.rms_weights_ptr = None
        
#     @abc.abstractmethod    
#     def preprocess(self, **args):
#         """
#         Translates a frame's atoms so that the center of mass is the origin.
#         (In-place modification.)
#         """
#         return

    @abc.abstractmethod    
    def compute_distances(self, **args):
        """ To compute distance between two frames 
        
        """
        return

    @abc.abstractmethod    
    def count_number_neighbours(self,**args):
        ''' count number of neighbours for frames
        '''
        return
    
    @abc.abstractmethod    
    def fit_trajectory(self,**args):
        ''' Function to fit a given frame number to all the frames in a trajectory.
            The fitted trajectory is returned.
        '''
        return
    
#    @staticmethod
#     def _exampleOnly_mpi_Bcast(metric_obj, comm, root):
#         rank = comm.Get_rank()
#     
#         if rank == root:
#             metric_obj.destroy_pointers()
#         else:
#             metric_obj = None
#     
#         metric_obj = comm.bcast(metric_obj, root=root)
#         metric_obj.create_pointers()
#     
#         return metric_obj
    
    
        