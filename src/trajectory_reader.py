from gp_grompy import Gmxtc


# I don't think it should be a class at the moment
# so I'm exposing the functions at the module level.

# ================================
# Methods to load coordinates from trajectory files.

def read_coordinates(traj_type, traj_filepath, no_frames, stride):
    if traj_type == "xtc":
        return read_coordinates_xtc(traj_filepath, no_frames, stride)
    else:
        raise ValueError("Unsupported trajectory type '{0}'".format(traj_type))


def read_coordinates_xtc(traj_filepath, no_frames, stride):
    gmxtc = Gmxtc()
    # No periodic boundary conditions. i.e molecules must be whole
    bPBC = 0 
    coordinates = gmxtc.load_traj(traj_filepath, stride, bPBC, no_frames, isize=None, index=None)

    return coordinates


# ================================
# Methods to get only the length (in frames) of a trajectory file.

def read_frameLength(traj_type, traj_filepath):
    if traj_type == "xtc":
        return read_frameLength_xtc(traj_filepath)
    else:
        raise ValueError("Unsupported trajectory type '{0}'".format(traj_type))



def read_frameLength_xtc(traj_filepath):
    gmxtc = Gmxtc() # instantiate gp_tools xtc reader object
    return gmxtc.get_trajlength(traj_filepath)



if __name__ == "__main__":
    print "testing"

