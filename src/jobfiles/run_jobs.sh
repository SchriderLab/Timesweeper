for i in \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-selectiveSweep-10Samp-20Int \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-selectiveSweep-5Samp-40Int \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-selectiveSweep-2Samp-100Int \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-selectiveSweep-20Samp-10Int \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-selectiveSweep-40Samp-5Int
do
    python ../hap_networks.py prep ${i}
    python ../hap_networks.py prep ${i} --time-series

    sbatch train_hap_net.sb $i
    sbatch train_hap_net.sb $i --time-series
done