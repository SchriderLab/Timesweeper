for i in 400 600 800 1000 
    do
        sbatch run_snakemake.sb /pine/scr/l/s/lswhiteh/timeSeriesSweeps-code/src/run_configs/post_selection_timing/uniform_10pt_20samp_${i}gens_post_sweep.yaml
    done