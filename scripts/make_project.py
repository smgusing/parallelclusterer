from parallelclusterer.project import Project
import sys
argv = sys.argv

#import logging
#logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
#ch = logging.StreamHandler(sys.stdout)
#formatter = logging.Formatter(fmt='%(filename)s:%(funcName)s:%(message)s')
#ch.setFormatter(formatter)
#logger.addHandler(ch)
#logger.propagate = False

argv = argv[1:]

gro_filepath = argv[0]
ndx_filepath = argv[1]
tpr_filepath = argv[2]

number_dimensions = int(argv[3])
stride = int(argv[4])

trajectory_type = argv[5]
trajectory_filepaths = argv[6:]

my_project = Project(   trajectory_filepaths=trajectory_filepaths,
                        trajectory_type=trajectory_type,
                        gro_filepath=gro_filepath,
                        ndx_filepath=ndx_filepath,
                        tpr_filepath=tpr_filepath,
                        number_dimensions=number_dimensions,
                        stride=stride,  )
                        
my_project.write_project("./my_project.yaml")
