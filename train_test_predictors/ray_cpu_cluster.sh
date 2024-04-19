#!/bin/bash
#PBS -l place=free,walltime=336:00:00,select=4:ncpus=16
#PBS -j oe
#PBS -o results.out
#PBS -N ray
#PBS -m bea
#PBS -W sandbox=PRIVATE
#PBS -k n
#PBS -V

ln -s $PWD $PBS_O_WORKDIR/$PBS_JOBID
cd $PBS_O_WORKDIR

mkdir -p log
mkdir -p cv
mkdir -p models
chmod +x ray_workers.sh

M=${NCPUS}
G=$(echo $CUDA_VISIBLE_DEVICES | grep -o "GPU" | wc -l)
jobnodes=`uniq -c ${PBS_NODEFILE} | awk -F. '{print $1 }' | awk '{print $2}' | paste -s -d " "`
thishost=`uname -n | awk -F. '{print $1.}'`
thishostip=`hostname -i`
rayport=6379
echo "pbs_o_workdir is: $PBS_O_WORKDIR"

thishostNport="${thishostip}:${rayport}"
echo "allocate to nodes = <$jobnodes>"

echo "setting up ray cluster with ${M} cpus per node... " 
echo "setting up ray cluster with ${G} gpus per node... "
for n in `echo ${jobnodes}`
do
        if [[ ${n} == "${thishost}" ]]
        then
                echo "locate ${thishostNport} node - use as headnode ..."
                source activate rapids
                ray start --head --num-cpus="${M}" --num-gpus="${G}"
                sleep 10
        else
                ssh ${n}  $PBS_O_WORKDIR/ray_workers.sh ${thishostNport} ${M} ${G}  
                sleep 10
        fi
done 

sleep 5
ray status
python rf_rfs.py -input="tagg_final.csv" -corr_max=0.95 -cv_splits=3 -cv_repeats=3
python rf_efs.py -input="rf_ranked_features.csv" -n_efs=5 -cv_splits=3 -cv_repeats=3
wait
python predictors.py -input="rf_rfs.csv" -mode="rfs" -scoring="mae" -cv_splits=18 -cv_repeats=3 -bopt_iters=50 -bopt_points=4
python predictors.py -input="rf_efs.csv" -mode="efs" -scoring="mae" -cv_splits=18 -cv_repeats=3 -bopt_iters=500 -bopt_points=4
python predictors.py -input="tagg_final.csv" -mode="afs" -scoring="mae" -cv_splits=18 -cv_repeats=3 -bopt_iters=50 -bopt_points=4
wait

rm $PBS_O_WORKDIR/$PBS_JOBID


