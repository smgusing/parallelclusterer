# for python2
#####
# This is a rewrite of original container class written by Alex.
# The idea behind this class is to serve as a data container for multiple trajectories   
# and implement methods that aid in processing of multiple trajectories in parallel using mpi4py
#  
#
####
# Built-in modules
import ctypes
from itertools import chain, izip
import logging
# External modules
import numpy as np
from mpi4py import MPI 
import gp_grompy
# Package classes
import trajectory_reader


logger = logging.getLogger(__name__)
# Type declarations.
# Need to match size of Gromacs' c_real. Which for now is 32-bit.
NumpyFloat = np.float32
CtypesFloat = ctypes.c_float
CtypesFloatPtr = ctypes.POINTER(ctypes.c_float)
MPIFloat32 = MPI.FLOAT
MPIInt32 = MPI.INT
MPIByte = MPI.CHAR
MAXTAGS = 10

class Framecollection():
    """
    A class to facilitate handling of strided trajectory data in parallel.

    - Allows indexing frames by a globalID.
    - Methods for sending data over mpi4py.

    .. Note:: No error checking in arguments implemented yet
    
    Parameters
    ----------
    globalIDs: array_like (int32)
         global number of the frame
    localID: array_like (int32)
        local number of the frame
    frames: array_like (float32)
        concatenated trajectory frames
    traj_filepath: list  
        filepath of trajs in this collections,in sequence
    traj_lengths: array_like (int32)
        number of frames in each trajectory that are part of this collection, in sequence
    mask: boolean_array ( 8 bit?)
        false by default
    
    Attributes
    ----------
    globalIDs: array_like
    
    localIDs: array_like
    
    frames:  array_like
    
    mask:  array_like
    
    traj_filepaths: list
    
    traj_lenghts: array_like
    
    tags: list
        tags for messages
    
    mpi_frametype:
     MPIFLOAT32
    
    
    """


    def __init__(self,globalIDs = None, localIDs=None, frames = None, mask = None,
                  traj_filepaths = None, traj_lenghts = None):
        """
        
        """
        self.globalIDs = globalIDs #
        self.localIDs  = localIDs
        self.frames = frames 
        self.mask = mask
        self.traj_filepaths = traj_filepaths
        self.traj_lenghts = traj_lenghts
        self.tags=[i for i in range(MAXTAGS)]
        self.mpi_frametype = MPIFloat32
        try:
            if globalIDs.size == localIDs.size:
                self.global_to_local = self._gen_dict()
                  
            else:
                raise SystemExit("number of globalIDs and localIDs are different ..exiting")
            
        except AttributeError:
                self.global_to_local = None
        
            
                         
                  
    def _gen_dict(self):
        ''' return dict with globalIDs as keys and localIDs as values
        '''
        
        return dict(zip(self.globalIDs,self.localIDs))
        

    @classmethod
    def from_files(cls, stride, trajectory_type,
                    trajectory_globalID_offsets, trajectory_filepath_list, trajectory_length_list):
        """
        Instantiate by providing a list of paths to trajectory files,
        a list of the lengths of those trajectories,
        and a list of their globalID offsets (the desired globalID of the first frame of the trajectory)
        and other metadata (stride, trajectory type).

        A trajectory reader will be used to load the trajectories,
        loading only every n'th frame, where n = stride, starting with the first.

        Each loaded frame will be labelled with a globalID.
        
        Parameters
        ----------
        stride                      : Integer
        trajectory_type             : String
        trajectory_globalID_offsets : [Integer]
        trajectory_filepath_list    : [String]
        trajectory_length_list      : [Integer]
        
        Returns
        -------
            framecollection object

        
        """
        new = cls()
        new.globalIDs = np.array(list(chain( *(  xrange(offset, length+offset, stride) 
                                        for offset, length
                                        in izip(trajectory_globalID_offsets, trajectory_length_list) 
                                    ))), dtype=np.int32)

        new.localIDs  = np.arange(new.globalIDs.size,dtype=np.int32)
        new.mask      = np.zeros(new.globalIDs.size,dtype=np.bool)
        
        trajectory_coordinate_arrays = [
             trajectory_reader.read_coordinates(trajectory_type, traj_filepath, traj_length, stride) for 
             (traj_filepath, traj_length) in izip(trajectory_filepath_list, trajectory_length_list)  ]
         
        new.frames = np.concatenate(trajectory_coordinate_arrays) # np.concatenate vs np.vstack?
        if (new.globalIDs.shape[0] != new.number_frames):
            logger.error(" Number of Frame Expected [%d] and number of frame read [%d] do not match",
                                                                        new.globalIDs.shape[0],new.number_frames)
            raise SystemExit("Exiting..")
        
        new.traj_filepaths = trajectory_filepath_list
        new.traj_lengths   = np.array(trajectory_length_list,dtype=np.int32)
        new.global_to_local = new._gen_dict()
        
        logger.debug("traj_lengths %s: globaIDs shape %s frames shape %s",
                     new.traj_lengths.shape,new.globalIDs.shape,new.frames.shape)
        return new



    # ================================================================
    # Copy/Slicing Stuff

    def copy(self):
        '''
        performs a deepcopy like operation. 
            
        '''

        new = Framecollection()
        new.globalIDs = np.copy(self.globalIDs)
        new.localIDs = np.copy(self.localIDs)
        new.frames = np.copy(self.frames)
        new.mask   = np.copy(self.mask)
        new.traj_filepaths = list(self.traj_filepaths)
        new.traj_lengths =np.copy(self.traj_lengths)
        new.global_to_local = new._gen_dict()
        
        return new



    # ================================================================
    # MPI Methods: To facilitate sending containers over mpi4py.
    #"""
    #Split the container into its array and the rest of its data (the metadata),
    #in order to send the array as a buffer (bypassing pickling).
    #"""
    def send_to_remote(self, comm, dest_rank):
        """ To transmit all the object's information
        
            Parameters
            -------------
            comm: object of Comm class from mpi4py
            dest_rank: rank of the node to send to
        """
        tag=0
        comm.send([self.frames.shape,self.globalIDs.shape,
                   self.traj_lengths.shape],dest=dest_rank,tag=tag)
        tag += 1
        comm.Send([self.frames, MPIFloat32], dest=dest_rank,tag=tag)
        tag += 1
        comm.Send([self.globalIDs, MPIInt32], dest=dest_rank,tag=tag)
        tag += 1
        comm.Send([self.localIDs, MPIInt32], dest=dest_rank,tag=tag)
        tag += 1
        comm.Send([self.mask, MPIByte], dest=dest_rank,tag=tag)
        tag += 1
        comm.Send([self.traj_lengths,MPIInt32],dest=dest_rank,tag=tag)
        tag += 1
        comm.send(self.traj_filepaths,dest=dest_rank,tag=tag)
        
        logger.debug("sent to destination %s tag %s",dest_rank,tag)

    def for_sending(self, comm):
        """ returns lambda which needs only rank of the process to send data to
        
            Parameters
            ------------
            comm: object of Comm class from mpi4py
            
            
        """
        return lambda dest_rank: self.send_to_remote(comm, dest_rank)


    def receive_from_remote(self, comm, send_rank):
        """ To recieve all the object's information
        
            Parameters
            --------------
            comm: object of Comm class from mpi4py
            send_rank: rank of the node to recieve from
        """
        tag = 0
        # need to get frame shape before we can receive
        frame_shape,ID_shape,trajlen_shape = comm.recv(source=send_rank,tag=tag)
        
        self.frames = np.empty(frame_shape,dtype=NumpyFloat)
        self.globalIDs = np.empty(ID_shape,dtype=np.int32)
        self.localIDs = np.empty(ID_shape,dtype=np.int32)
        self.mask = np.empty(ID_shape,dtype=np.bool)
        self.traj_lengths = np.empty(trajlen_shape,dtype=np.int32)
        
        tag += 1
        comm.Recv([self.frames, MPIFloat32], source=send_rank,tag=tag)
        tag += 1
        comm.Recv([self.globalIDs, MPIInt32], source=send_rank,tag=tag)
        tag += 1
        comm.Recv([self.localIDs, MPIInt32], source=send_rank,tag=tag)
        tag += 1
        comm.Recv([self.mask, MPIByte], source=send_rank,tag=tag)
        tag += 1
        comm.Recv([self.traj_lengths, MPIInt32], source=send_rank,tag=tag)
        tag += 1
        self.traj_filepaths = comm.recv(source=send_rank,tag=tag)
        
        logger.debug("Received from source %s tag %s",send_rank,tag)
        
        if self.frames.flags["C_CONTIGUOUS"] == False:
            logger.warning("non contiguous array.. remaking it")
            self.frames = np.ascontiguousarray(self.frames, dtype=NumpyFloat)
        
        self.global_to_local = self._gen_dict()
    
    def for_receiving(self,comm):
        """ returns lambda which needs only rank of the process to receive data from
            
            Parameters
            --------------
            comm: object of Comm class from mpi4py
        """
        
        return lambda send_rank: self.receive_from_remote(comm, send_rank)
        
    # ================================================================
    # Array indexing methods.

    def get_frame(self, globalID):
        ''' return frame for a given globalID
        
        '''
        
        try:
            localID = self.global_to_local[globalID]
        except KeyError:
            raise KeyError("Frame with globalID '{0}' not found in container.".format(globalID))
            
        return self.frames[localID]


    def get_frame_pointer(self, globalID):
        ''' return pointer that points to frame at given globalID
        '''
        try:
            localID = self.global_to_local[globalID]
        except KeyError:
            raise KeyError("Frame with globalID '{0}' not found in container.".format(globalID))
            
        return self.frames[localID:].ctypes.data_as(ctypes.POINTER(gp_grompy.c_real))


    def get_first_frame_pointer(self):
        ''' return pointer that points to the first frame
        '''
        
        try:
            ptr = self.frames[0:].ctypes.data_as(ctypes.POINTER(gp_grompy.c_real))
        except KeyError:
            raise KeyError("The container holds no frames.")

        return ptr


    # ================================================================
    # Metadata accessor methods.
    @property
    def globalIDs_iter(self):
        ''' return global id iterator
        '''
        return iter(self.globalIDs)
    
    @property
    def number_frames(self):
        ''' return total number of frames
        '''
        
        return self.frames.shape[0]

    @property
    def number_atoms(self):
        ''' return number of atoms in a frame
        '''
        
        return self.frames.shape[1]
    
    




