for i in \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-adaptiveIntrogression-10Samp-20Int \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-adaptiveIntrogression-5Samp-40Int \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-adaptiveIntrogression-2Samp-100Int \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-adaptiveIntrogression-20Samp-10Int \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-adaptiveIntrogression-40Samp-5Int \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/twoPop-adaptiveIntrogression-10Samp-20Int \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/twoPop-adaptiveIntrogression-5Samp-40Int \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/twoPop-adaptiveIntrogression-2Samp-100Int \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/twoPop-adaptiveIntrogression-20Samp-10Int \
    /pine/scr/l/s/lswhiteh/timeSeriesSweeps/twoPop-adaptiveIntrogression-40Samp-5Int
do
    #python ../hap_networks.py prep ${i}
    #python ../hap_networks.py prep ${i} --time-series

    sbatch train_hap_net.sb $i
    sbatch train_hap_net.sb $i --time-series
done