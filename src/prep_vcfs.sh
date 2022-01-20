#!/bin/bash
#SBATCH --partition=general
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH -J make_sims
#SBATCH -o logfiles/vcf_process.%A.%a.out
#SBATCH -e logfiles/vcf_process.%A.%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lswhiteh@email.unc.edu
#SBATCH --array=1-5001

conda activate blinx
source activate blinx

for i in hard neut soft
    do
        indir=/proj/dschridelab/lswhiteh/timesweeper/simple_sims/vcf_sims/onePop-selectiveSweep-vcf.slim/${i}/pops/${SLURM_ARRAY_TASK_ID}

        for vcf in $indir/*.vcf
            do
                bgzip -c $vcf > $vcf.gz
                bcftools index $vcf.gz 
            done
        
        #Fix samp names
        bcftools merge -Oz --force-samples -0 ${indir}/*.vcf.gz > ${indir}/merged.vcf.gz
    
    done
