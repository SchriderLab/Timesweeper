for i in hard neut soft; do python ../../src/get_allele_freqs.py -i slimulations/${i}/pops; done
for i in neut hard soft; do python ../../src/allele_freq_mat.py slimulations/${i}/freqs; done
python ../../src/merge_npzs.py merged_freqs.npz ${1}/*/*_freqmats.npz
python ../../src/allele_freq_net.py train -i merged_freqs.npz -n allele_freqs
python ../../src/summarize_3class.py . OoAAllele_freqs_1DCNN allele_freqs_TimeSweeperHaps_predictions.csv
python ../../src/plotting/plot_hap_spec.py merged_freqs.npz

#for i in neut hard soft; do python ../../src/haplotypes.py -i slimulations/${i}/pops -s OoA -o slimulations/${i}
#python ../../src/merge_npzs.py merged_haps.npz ${1}/*/hfs_*.npz
#python ../../src/hap_net.py train -i merged_haps.npz -n haps
#python ../../src/summarize_3class.py . OoA_hfs_1DCNNhaps_TimeSweeper_predictions.csv
#python ../../src/plotting/plot_hap_spec.py merged_haps.npz

