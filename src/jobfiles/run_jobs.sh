for i in /overflow/dschridelab/users/lswhiteh/timeSeriesSweeps/src/run_configs/uniform_vs_dense_schema_comparison/*
    do
        sbatch run_snakemake.sb $i 
    done