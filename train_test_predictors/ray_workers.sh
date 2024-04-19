#!/bin/bash -l
source $HOME/.bashrc
cd $PBS_O_WORKDIR
param1=$1
param2=$2
param3=$3
destnode=`uname -n`
localhostip=`hostname -i`
echo "initializing ${param2} cpus on worker node $destnode"
echo "initializing ${param3} gpus on worker node $destnode"
echo "ray address = [$param1]"

source activate rapids
ray start --address="${param1}" --node-ip-address="${localhostip}" --redis-password='' --num-cpus="${param2}" --num-gpus="${param3}"
