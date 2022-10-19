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
#SBATCH --array=1-2501:100

conda activate blinx
source activate blinx

for i in sdn neut ssv
do
    base_dir=/proj/dschridelab/lswhiteh/timesweeper-experiments/empirical_model/OoA_stdpopsim/sims/vcfs/${i}/
    
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
        bcftools merge -Oz --force-samples -0 $(seq -f ${indir}/%01.0f.vcf.gz 0 19) > ${indir}/merged.vcf.gz
    done
done
