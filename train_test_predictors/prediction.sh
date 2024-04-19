#!/bin/bash

#PBS -j oe
#PBS -o results.out
#PBS -N prediction
#PBS -m bea
#PBS -V

cd $PBS_O_WORKDIR

#activate conda env
source activate rapids

python prediction.py -holdout="holdout.csv" -models="all" -rescaled=False
