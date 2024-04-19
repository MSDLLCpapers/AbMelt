#!/bin/bash -l
#PBS -l 
#PBS -N gmx_arr
#PBS -m bea
#PBS -j oe
#PBS -o gmx_arr.out
#PBS -J 1-400


# set tmpdir for ompi arr & set env variables
export TMPDIR=/tmp/
export OMP_NUM_THREADS=16
export GMX_ENABLE_DIRECT_GPU_COMM=true
export GMX_FORCE_UPDATE_DEFAULT_GPU=true

if [ ! -z "$PBS_O_WORKDIR" ]; then
cd $PBS_O_WORKDIR
fi

# extract JOBID for directed outputs
JOBID=`echo ${PBS_JOBID} | cut -d'[' -f1`

# select project directory
project="AbMelt"

# sample array index & map to execution directory
arrayid=$PBS_ARRAY_INDEX
sample=$(sed "${arrayid}q;d" project/$project/job.conf)

# activate conda environment
conda activate abmelt

# run MD, fix trajectories, analyze descriptors, cluster structures
python ./src/moe_gromacs.py --project $project --dir $sample --temp '300, 350, 400' --ff 'charmm27' --md 
python ./src/moe_gromacs.py --project $project --dir $sample --temp '300, 350, 400' --ff 'charmm27' --fix_trj 
python ./src/moe_gromacs.py --project $project --dir $sample --temp '300, 350, 400' --ff 'charmm27' --analyze 

