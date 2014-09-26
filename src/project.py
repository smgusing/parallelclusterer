# for python2

# modules
import yaml
import numpy as np
import os, os.path

# other classes
import trajectory_reader

class Project():
    """
    TODO: Need to get index groups, somehow.

This class handles reading and writing of project metadata, including:
- The list of trajectories analyzed
- The topologies used to generate the trajectories
  (this data is sometimes necessary for analysis)
- The recording of analysis options which affect the results of analyses
  (and so must be recorded for consistency).

Class Preconditions:

Class Guarantees:
- Objects are always instantiated with all data. (No missing data.)

Notes:
- The class can be instantiated with either an existing project file
or by providing the files for a new project.
- Project files are recorded on disk as YAML records.
- Call only methods having names without leading underscores.
    """

    # ================================
    # Methods for instantiation.

    def __init__(self,
        # 'by existing project' parameters:
        existing_project_file=None,
        # 'new' parameters:
        trajectory_filepaths=None, trajectory_type=None,
        gro_filepath=None, ndx_filepath=None, tpr_filepath=None,
        number_dimensions=None, stride=None,
        ):
        """
        existing_project_file :: String
            A path to an existing yaml project record.
        trajectory_filepaths :: [String]
            A list of paths to trajectories.
        """

        # --------------------------------
        # Declare all fields.

        # :: String
        # A path to the yaml record which stores project metadata.
        self.project_filepath = None 

        # :: String
        # Paths to the gromacs topologies corresponding to the trajectories.
        self.gro_filepath = None
        self.ndx_filepath = None
        self.tpr_filepath = None

        # :: String
        # A lower-case string denoting the type of the trajectories in trajectory_listing.
        self.trajectory_type = None

        # :: [(Integer traj_id, String traj_filepath, Integer traj_length)]
        # A record of the trajectories analyzed or to be analyzed.
        self.trajectory_listing = None

        # :: Integer
        # For analyzing the trajectories sparsely:
        # use only frames such that (frame_number mod frame_stride = 0), for indexing starting at 0.
        self.frame_stride = None

        # :: Integer
        # The spatial dimensionality of the data.
        # (Usually 3, but one may wish to consider the projection of the data onto a plane, for example.)
        self.number_dimensions = None
        self.stride = None

        # The number of trajectories can be obtained with "len(trajectory_listing)".

        # How to specify maximum number of processes (for multiprocessing)?
        # It's more of a runtime setting rather than a project setting.

        # --------------------------------
        # Choose instantiator.

        # new:
        args = (trajectory_filepaths, gro_filepath, ndx_filepath,
                tpr_filepath, trajectory_type, number_dimensions, stride)
        if all_not_none(*args):
            self._init_new(*args)

        # from exiting project:
        elif existing_project_file != None:
            self._init_by_existing_project(existing_project_file)

        # no match:
        else:
            raise ValueError("Parameters are not sufficient for loading an existing project nor creating a new project.")



    def _init_by_existing_project(self, existing_project_file):
        """
        existing_project_file :: String
            A path to an existing yaml project record.
        """

        # Check for existence of file.
        if not os.path.exists(existing_project_file):
            raise IOError("Project file '{0}' does not exist.".format(existing_project_file))
            
        # Record project file.
        self.project_filepath = existing_project_file

        # Load records.
        with open(self.project_filepath, 'r') as file_handle:
            record = yaml.load(file_handle)

            # Read topologies.
            self.gro_filepath = record['gro_filepath']
            self.ndx_filepath = record['ndx_filepath']
            self.tpr_filepath = record['tpr_filepath']

            # Read trajectory type.
            self.trajectory_type = record['trajectory_type']

            # Read trajectory listing.
            trajectory_listing = []
            trajectory_list = record['trajectory_listing']

            # trajectory_list :: [Dictionary]
            for traj in trajectory_list:
                traj_id = traj['id']
                traj_filepath = traj['filepath']
                traj_length = traj['length']
                trajectory_listing.append((traj_id, traj_filepath, traj_length))

            self.trajectory_listing = trajectory_listing

            # Read number of dimensions.
            self.number_dimensions = record['number_dimensions'] 
            self.stride = record['stride']



    def _init_new(self, trajectory_filepaths, gro_filepath, ndx_filepath,
                    tpr_filepath, trajectory_type, number_dimensions, stride):
        """
        trajectory_filepaths :: [String]
            A list of paths to trajectory files to be analyzed.
        gro_filepath, ndx_filepath, tpr_filepath :: String
            Paths to Gromacs topology files.
        trajectory_type :: String
            A lower-case string denoting the trajectory file type (e.g. 'xtc').
        """
        # Set topologies.
        self.gro_filepath = gro_filepath
        self.ndx_filepath = ndx_filepath
        self.tpr_filepath = tpr_filepath

        # Set trajectory type.
        self.trajectory_type = trajectory_type

        # Set trajectory listing.
        trajectory_listing = []

        for traj_id, traj_filepath in enumerate(trajectory_filepaths):
            print "Reading length of trajectory {0}".format(traj_filepath)
            traj_length = self._get_trajectory_length(trajectory_type, traj_filepath)

            trajectory_listing.append((traj_id, traj_filepath, traj_length))
        
        self.trajectory_listing = trajectory_listing

        # Set number dimensions.
        self.number_dimensions = number_dimensions
        self.stride = stride


    # doesn't need to be an object method
    def _get_trajectory_length(self, trajectory_type, trajectory_filepath):
        """
        trajectory_type :: String
            A lower-case string denoting the trajectory file type (e.g. 'xtc').
        trajectory_filepath :: String
            A path to a trajectory.
        """
        return trajectory_reader.read_frameLength(trajectory_type, trajectory_filepath)



    # ================================
    # Methods for writing to disk as a yaml record.

    def write_project(self, path=None):
        if path == None:
            path = self.project_filepath

        # Check if a file exists at the path.
        if os.path.exists(path):
            raise IOError, "File already exists; refusing to overwrite."

        with open(path, 'w') as file_handle:
            record = {}
            
            # Write topologies.
            record['gro_filepath'] = self.gro_filepath
            record['ndx_filepath'] = self.ndx_filepath
            record['tpr_filepath'] = self.tpr_filepath

            # Write trajectory type.
            record['trajectory_type'] = self.trajectory_type

            # Write trajectory listing.
            trajectory_list = record['trajectory_listing'] = []

            for traj_id, traj_filepath, traj_length in self.trajectory_listing:
                traj = {}

                traj['id'] = traj_id
                traj['filepath'] = traj_filepath
                traj['length'] = traj_length

                trajectory_list.append(traj)

            # Write number of dimensions.
            record['number_dimensions'] = self.number_dimensions
            record['stride'] = self.stride

            yaml.dump(record, file_handle)

        
    # ================================
    # Accessor Methods.

    # Notes:
    # - self.trajectory_listing :: [(Integer traj_id, String traj_filepath, Integer traj_length)]

    def get_trajectory_lengths(self):
        return [ traj_length for (traj_id, traj_filepath, traj_length) in self.trajectory_listing ]

    def get_trajectory_filepaths(self):
        return [ traj_filepath for (traj_id, traj_filepath, traj_length) in self.trajectory_listing ]

    def get_trajectory_ids(self):
        return [ traj_id for (traj_id, traj_filepath, traj_length) in self.trajectory_listing ]

    def get_trajectory_type(self):
        return self.trajectory_type
        
    def get_gro_filepath(self):
        return self.gro_filepath
    def get_ndx_filepath(self):
        return self.ndx_filepath
    
    def get_tpr_filepath(self):
        return self.tpr_filepath

    def get_number_dimensions(self):
        return self.number_dimensions
    def get_stride(self):
        return self.stride

# helper functions
def all_not_none(*args):
    return all( x != None for x in args)

