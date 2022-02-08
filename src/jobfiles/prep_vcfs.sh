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
#SBATCH --array=1-5001:100

conda activate blinx
source activate blinx

for i in hard neut soft
do
    base_dir=${1}/${i}/pops
    
    for id in $(seq ${SLURM_ARRAY_TASK_ID} $((${SLURM_ARRAY_TASK_ID} + 100)))
    do
        indir=${base_dir}/${id} 

        rm $indir/*.vcf.gz*

        for vcf in $indir/*.vcf
        do
                bgzip -c $vcf > $vcf.gz
                bcftools index $vcf.gz 
        done
        
        #Fix samp names
        bcftools merge -Oz --force-samples -0 ${indir}/*.vcf.gz > ${indir}/merged.vcf.gz
    done
done
