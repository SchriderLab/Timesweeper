for i in $(ls -d /pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-selectiveSweep/sims/*)
    do 
        sbatch prep_data.sb $i
    done