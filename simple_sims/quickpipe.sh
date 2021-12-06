#!/bin/bash
#SBATCH --partition=general
#SBATCH --mem=16G
#SBATCH --ntasks=64
#SBATCH --time=06:00:00
#SBATCH -J run_pipelines
#SBATCH -o logfiles/run_pipelines.%A.out
#SBATCH -e logfiles/run_pipelines.%A.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lswhiteh@email.unc.edu

conda activate blinx

cd ${1}
echo $pwd

for i in hard neut soft; do python /proj/dschridelab/lswhiteh/timesweeper/src/get_allele_freqs.py -i ${i}/pops; done
for i in neut hard soft; do python /proj/dschridelab/lswhiteh/timesweeper/src/allele_freq_mat.py ${i}/freqs; done
python /proj/dschridelab/lswhiteh/timesweeper/src/merge_npzs.py merged_freqs.npz */*_freqmats.npz
python /proj/dschridelab/lswhiteh/timesweeper/src/allele_freq_net.py train -i merged_freqs.npz -n allele_freqs
#python /proj/dschridelab/lswhiteh/timesweeper/src/summarize_3class.py . Allele_freqs_1DCNN allele_freqs_TimeSweeper_predictions.csv
python /proj/dschridelab/lswhiteh/timesweeper/src/plotting/plot_freq_spec.py merged_freqs.npz

for i in neut hard soft; do python /proj/dschridelab/lswhiteh/timesweeper/src/haplotypes.py -i ${i}/pops -s haps -o ${i}; done
python /proj/dschridelab/lswhiteh/timesweeper/src/merge_npzs.py merged_haps.npz */hfs_haps.npz
python /proj/dschridelab/lswhiteh/timesweeper/src/hap_networks.py train -i merged_haps.npz -n haps
#python /proj/dschridelab/lswhiteh/timesweeper/src/summarize_3class.py . Haps *_predictions.csv
python /proj/dschridelab/lswhiteh/timesweeper/src/plotting/plot_hap_spec.py merged_haps.npz

python /proj/dschridelab/lswhiteh/timesweeper/src/summarize_3class.py . ROC_Curves *_predictions.csv
