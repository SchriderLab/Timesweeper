#!/bin/bash
#SBATCH --partition=general
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH -J classify
#SBATCH -o logfiles/classify.%A.%a.out
#SBATCH -e logfiles/classify.%A.%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lswhiteh@email.unc.edu
#SBATCH --array=1-5001

##SLURM_ARRAY_TASK_ID=1

conda activate blinx
source activate blinx

for i in hard neut soft
do  
    indir=/proj/dschridelab/lswhiteh/timesweeper/simple_sims/vcf_sims/onePop-selectiveSweep-vcf.slim/${i}/pops/${SLURM_ARRAY_TASK_ID}
    
    rm $indir/*.npy $indir/.csv

    python classify_windows.py \
    --afs-model ../simple_sims/onepop/onePop-selectiveSweep/models/allele_freqs_TimeSweeper \
    --hfs-model ../simple_sims/onepop/onePop-selectiveSweep/models/haps_TimeSweeper \
    -s $(printf '5%.s ' {1..20}) \
    -i $indir/merged.vcf.gz \
    -o $indir/results.csv
done
