#!/bin/sh

# request bash as shell for job
#PBS -S /bin/bash

# queue, parallel environment and number of processors
#PBS -l select=1:ncpus=12:mem=10gb

#PBS -l walltime=72:00:00

# joins error and standard outputs
#PBS -j oe

# keep error and standard outputs on the execution host
#PBS -k oe

# forward environment variables
#PBS -V

# define job name
#PBS -N Spline

# main commands here

module load anaconda3/personal

cd /rds/general/user/mm2319/home/M4R/

python3 spline_Baye_Reg_Denoise_noise_025.py