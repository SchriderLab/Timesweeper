#!/bin/bash
##SBATCH --partition=general
#SBATCH --mem=32G
##SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH -J gen_fits
#SBATCH -o gen_fits.%A.out
#SBATCH -e gen_fits.%A.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lswhiteh@email.unc.edu

cd /pine/scr/l/s/lswhiteh/timeSeriesSweeps/timesweeper

source activate blinx

python feder_method.py ../onePop-selectiveSweep-20Samp-10Int/