#! /bin/bash
source ~/.bashrc

msmbeta
gmx455

function make_project {

if [ ! -f ./my_project.yaml ]
then
    make_project.py --gro data/nr.gro --ndx data/nr.ndx --tpr data/nr.tpr --ndim 3 --trajtype xtc --stride 1 --trajlist trajlist
fi

}

function cluster {
echo 0 0 | mpiexec -np 2 cluster.py -projectfile my_project.yaml -cutoff 0.05
#echo 0 0 | cluster.py -projectfile my_project.yaml -cutoff 0.05
}

function ana {
#anaclust.py -projfn my_project.yaml -centers centers.txt \
#-clusters clusters.txt -timestep 2  --get_centers

### This will dump largest cluster aligned on to its center frame
echo 0 0| anaclust.py -projfn my_project.yaml -centers centers.txt \
-clusters clusters.txt -timestep 2  -clid 0 -nconf 600 --get_clusterno

}


make_project
#cluster
#ana
