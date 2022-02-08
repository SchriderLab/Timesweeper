#!/bin/bash
#SBATCH --partition=general
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH -J classify
#SBATCH -o logfiles/classify.%A.%a.out
#SBATCH -e logfiles/classify.%A.%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lswhiteh@email.unc.edu
#SBATCH --array=1-5001:500

##SLURM_ARRAY_TASK_ID=1

conda activate blinx

for i in hard neut soft
do  
    for id in $(seq ${SLURM_ARRAY_TASK_ID} $((${SLURM_ARRAY_TASK_ID} + 500)))
    do
        indir=${1}/${i}/pops/${id}
        
        rm -f $indir/*.npy $indir/.csv

        python classify_windows.py \
        --afs-model /proj/dschridelab/lswhiteh/timesweeper/simple_sims/models/allele_freqs_TimeSweeper \
        --hfs-model /proj/dschridelab/lswhiteh/timesweeper/simple_sims/models/bighaps_TimeSweeper \
        -s $(printf '1%.s ' {1..40}) \
        -i $indir/merged.vcf.gz \
        -o $indir/results.csv
    done
done
