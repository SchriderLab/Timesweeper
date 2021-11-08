#for i in hard neut soft; do python ../../src/FIt.py -i slimulations/${i}/pops; done
#for i in neut hard soft; do python ../../src/allele_freq_mat.py slimulations/${i}/freqs; done
python ../../src/merge_npzs.py merged_freqs.npz slimulations/*/*.npz
python ../../src/allele_freq_net.py train -i merged_freqs.npz -n allele_freqs
python ../../src/summarize_3class.py . Allele_freqs_1DCNN allele_freqs_TimeSweeperHaps_predictions.csv
python ../../src/plotting/plot_hap_spec.py merged_freqs.npz
