#!/bin/bash -l
#SBATCH --output=collect_trash.out
#SBATCH --get-user-env
#SBATCH --time=10:00:00
#SBATCH --partition=kcs_batch
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=32

# load the environment modules you would like to use:
module load intel/18.0 mpi.intel/2018

L=20
#
duration=1000
#
episode_length=100	
#
N_RUNS=10
#
lr=0.0001
#
gamma=1.0
#
dirName="output_"${SLURM_ARRAY_JOB_ID}
#
path="/home/mruslani/Skyrmions/New_Code/"$dirName
mkdir -v -p $path
path=$path"/"
#

python AC_exclude_dumm.py $L $duration $episode_length $N_RUNS $lr $gamma $path ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}