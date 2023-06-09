#!/bin/sh

# request bash as shell for job
#PBS -S /bin/bash

# queue, parallel environment and number of processors
#PBS -l select=1:ncpus=1:mem=1gb:ngpus=1

#PBS -l walltime=72:00:00

# joins error and standard outputs
#PBS -j oe

# keep error and standard outputs on the execution host
#PBS -k oe

# forward environment variables
#PBS -V

# define job name
#PBS -N FD

# main commands here

module load anaconda3/personal

cd /rds/general/user/mm2319/home/M4R/

python3 finite_diff_Baye_Reg_CSS_noise_000.py