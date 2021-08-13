for i in /overflow/dschridelab/users/lswhiteh/timeSeriesSweeps/src/run_configs/sel_coeff/*
    do
        sbatch run_snakemake.sb $i 
    done