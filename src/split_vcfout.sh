#!/bin/bash
#SBATCH --partition=general
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH -J make_sims
#SBATCH -o vcf_process.%A.%a.out
#SBATCH -e vcf_process.%A.%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lswhiteh@email.unc.edu
#SBATCH --array=1-1001

conda activate blinx
source activate blinx

for i in hard neut soft
    do
        indir=/proj/dschridelab/lswhiteh/timesweeper/simple_sims/vcf_sims/${i}/pops/${SLURM_ARRAY_TASK_ID}
        #splitdir=${1}/${i%.*}
        #mkdir -p $splitdir
        #csplit -s -z ${indir}.vcf /##fileformat=VCFv4.2/ {*} -f ${splitdir}/ -b "%04d.vcf"

        #for vcf in $indir/*
        #    do
        #        bgzip -c $vcf > $vcf.vcf.gz
        #        bcftools index $vcf.vcf.gz 
        #    done
        #
        #Fix samp names
        #bcftools merge -Oz --force-samples -0 ${indir}/*.vcf.gz > ${indir}/merged.vcf.gz

        python classify_windows.py \
            -i $indir/merged.vcf.gz \
            -o $indir/Timesweeper_predictions.csv \
            --afs-model /proj/dschridelab/lswhiteh/timesweeper/simple_sims/onepop/onePop-selectiveSweep/models/allele_freqs_TimeSweeper \
            --hfs-model /proj/dschridelab/lswhiteh/timesweeper/simple_sims/onepop/onePop-selectiveSweep/models/haps_TimeSweeper \
            -s $(printf '5 %.s' {1..20}) 
    
    done
