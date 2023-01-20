import argparse
import multiprocessing as mp
import os
import sys


def ts_main():
    agp = argparse.ArgumentParser(description="Timesweeper CLI")
    subparsers = agp.add_subparsers(help="Timesweeper modes", dest="mode")

    # simulate_stdpopsim.py
    sim_s_parser = subparsers.add_parser(
        name="sim_stdpopsim",
        description="Injects time-series sampling into stdpopsim SLiM output script.",
    )
    sim_s_parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        action="store_true",
        dest="verbose",
        help="Print verbose logging during process.",
    )
    sim_s_parser.add_argument(
        "--threads",
        required=False,
        type=int,
        default=mp.cpu_count() - 1,
        dest="threads",
        help="Number of processes to parallelize across.",
    )
    sim_s_parser.add_argument(
        "--rep-range",
        required=False,
        dest="rep_range",
        nargs=2,
        help="<start, stop>. If used, only range(start, stop) will be simulated for reps. \
            This is to allow for easy SLURM parallel simulations.",
    )
    sim_s_parser.add_argument(
        metavar="YAML CONFIG",
        dest="yaml_file",
        help="YAML config file with all options defined.",
    )

    # simulate_custom.py
    sim_c_parser = subparsers.add_parser(
        name="sim_custom",
        description="Simulates selection for training Timesweeper using a pre-made SLiM script.",
    )
    sim_c_parser.add_argument(
        "--threads",
        required=False,
        type=int,
        default=mp.cpu_count(),
        dest="threads",
        help="Number of processes to parallelize across. Defaults to all.",
    )
    sim_c_parser.add_argument(
        "--rep-range",
        required=False,
        dest="rep_range",
        nargs=2,
        help="<start, stop>. If used, only range(start, stop) will be simulated for reps. \
            This is to allow for easy SLURM parallel simulations.",
    )
    sim_c_parser.add_argument(
        "-y",
        "--yaml",
        metavar="YAML_CONFIG",
        required=True,
        dest="yaml_file",
        help="YAML config file with all required options defined.",
    )

    # make_training_features.py
    mtf_parser = subparsers.add_parser(
        name="condense",
        description="Creates training data from simulated merged vcfs after process_vcfs.py has been run.",
    )
    mtf_parser = subparsers.add_parser(
        name="condense",
        description="Creates training data from simulated merged vcfs after process_vcfs.py has been run.",
    )
    mtf_parser.add_argument(
        "--threads",
        required=False,
        type=int,
        default=mp.cpu_count() - 1,
        dest="threads",
        help="Number of processes to parallelize across.",
    )
    mtf_parser.add_argument(
        "-o",
        "--outfile",
        required=False,
        type=str,
        default="training_data.pkl",
        dest="outfile",
        help="Pickle file to dump dictionaries with training data to. Should probably end with .pkl.",
    )
    mtf_parser.add_argument(
        "--subsample-inds",
        required=False,
        type=int,
        dest="subsample_inds",
        help="Number of individuals to subsample if using a larger simulation than needed for efficiency. NOTE: If you use this, keep the sample_sizes entry in the yaml config identical to the original simulations. This is mostly for the paper experiments and as such isn't very cleanly implemented.",
    )
    mtf_parser.add_argument(
        "--subsample-tps",
        required=False,
        type=int,
        dest="subsample_tps",
        help="Number of timepoints to subsample if using a larger simulation than needed for efficiency. NOTE: If you use this, keep the sample_sizes entry in the yaml config identical to the original simulations. This is mostly for the paper experiments and as such isn't very cleanly implemented.",
    )
    mtf_parser.add_argument(
        "--og-tps",
        required=False,
        type=int,
        dest="og_tps",
        help="Number of timepoints taken in the original data to be subsetted if using --subample-tps.",
    )
    mtf_parser.add_argument(
        "-m",
        "--missingness",
        metavar="MISSINGNESS",
        dest="missingness",
        type=float,
        required=False,
        default=0.0,
        help="Missingness rate in range of [0,1], used as the parameter of a binomial distribution for randomly removing known values.",
    )
    mtf_parser.add_argument(
        "-f",
        "--freq-increase-threshold",
        metavar="FREQ_INC_THRESHOLD",
        dest="freq_inc_thr",
        type=float,
        required=False,
        default=0.0,
        help="If given, only include sim replicates where the sweep site has a minimum increase of <freq_inc_thr> from the first timepoint to the last.",
    )
    mtf_parser.add_argument(
        "-a",
        "--allow-shoulders",
        metavar="SAMPLING_OFFSET",
        dest="allow_shoulders",
        type=int,
        required=False,
        help="If value given: sample a uniform distribution with bounds of given value to offset training sample from central SNP.",
    )
    mtf_parser.add_argument(
        "--hft",
        required=False,
        action="store_true",
        dest="hft",
        help="Whether to calculate HFT alongside AFT. Computationally more expensive.",
    )
    mtf_parser.add_argument(
        "--no-progress",
        action="store_true",
        dest="no_progress",
        help="Turn off progress bar.",
    )
    mtf_parser.add_argument(
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Raise warnings from issues usually stemming from bad replicates.",
    )
    mtf_parser.add_argument(
        "-y",
        "--yaml",
        metavar="YAML_CONFIG",
        required=True,
        dest="yaml_file",
        help="YAML config file with all required options defined.",
    )

    # train_nets.py
    nets_parser = subparsers.add_parser(
        name="train",
        description="Handler script for neural network training and prediction for TimeSweeper Package.\
            Will train two models: one for the series of timepoints generated using the hfs vectors over a timepoint and one ",
    )
    nets_parser.add_argument(
        "-i",
        "--training-data",
        metavar="TRAINING_DATA",
        dest="training_data",
        type=str,
        required=True,
        help="Pickle file containing data formatted with make_training_features.py.",
    )
    nets_parser.add_argument(
        "-d",
        "--data-type",
        required=True,
        metavar="DATA_TYPE",
        dest="data_type",
        help="AFT or HFT data preparation.",
    )
    nets_parser.add_argument(
        "-s",
        "--subsample-amount",
        metavar="SUBSAMPLE_AMOUNT",
        dest="subsample_amount",
        type=int,
        required=False,
        help="Amount of data to subsample for each class to test for sample size effects.",
    )
    nets_parser.add_argument(
        "-n",
        "--experiment-name",
        metavar="EXPERIMENT_NAME",
        dest="experiment_name",
        type=str,
        required=False,
        default="ts_experiment",
        help="Identifier for the experiment used to generate the data. Optional, but helpful in differentiating runs.",
    )
    nets_parser.add_argument(
        "-y",
        "--yaml",
        metavar="YAML_CONFIG",
        required=True,
        dest="yaml_file",
        help="YAML config file with all required options defined.",
    )
    nets_parser.add_argument(
        "--single-tp",
        dest="single_tp",
        required=False,
        action="store_true",
        help="Whether to use the tp1_model module for a special case experiment.",
    )

    # find_sweeps.py
    sweeps_parser = subparsers.add_parser(
        name="detect",
        description="Module for iterating across windows in a time-series vcf file and predicting whether a sweep is present at each snp-centralized window.",
    )
    sweeps_parser.add_argument(
        "-i",
        "--input-vcf",
        dest="input_vcf",
        help="Merged VCF to scan for sweeps. Must be merged VCF where files are merged in order from earliest to latest sampling time, -0 flag must be used.",
        required=True,
    )
    sweeps_parser.add_argument(
        "--benchmark",
        dest="benchmark",
        action="store_true",
        help="If testing on simulated data and would like to report the mutation \
            type stored by SLiM during outputVCFSample as well as neutral predictions, use this flag. \
            Otherwise the mutation type will not be looked for in the VCF entry nor reported with results.",
        required=False,
    )
    sweeps_parser.add_argument(
        "--aft-model",
        dest="aft_model",
        help="Path to Keras2-style saved model to load for aft prediction.",
        required=True,
    )
    sweeps_parser.add_argument(
        "--hft-model",
        dest="hft_model",
        help="Path to Keras2-style saved model to load for hft prediction.",
        required=False,
    )
    sweeps_parser.add_argument(
        "-o",
        "--out-dir",
        dest="outdir",
        help="Directory to write output to.",
        required=True,
    )
    sweeps_parser.add_argument(
        "-y",
        "--yaml",
        metavar="YAML_CONFIG",
        required=True,
        dest="yaml_file",
        help="YAML config file with all required options defined.",
    )

    # plot_training_data.py
    input_plot_parser = subparsers.add_parser(
        name="plot_training",
        description="Plots central SNPs from simulations to visually inspect mean trends over replicates.",
    )
    input_plot_parser.add_argument(
        "-i",
        "--input-pickle",
        dest="input_pickle",
        metavar="INPUT PICKLE",
        type=str,
        required=True,
        help="Pickle file containing dictionary of structure dict[sweep][rep]['aft'] created by make_training_features.py.",
    )
    input_plot_parser.add_argument(
        "-n",
        "--schema-name",
        metavar="SCHEMA NAME",
        dest="schema_name",
        required=False,
        default="simulation_center_means",
        type=str,
        help="Experiment label to use for output file naming.",
    )
    input_plot_parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT DIR",
        dest="output_dir",
        required=False,
        default=".",
        type=str,
        help="Directory to write images to.",
    )
    input_plot_parser.add_argument(
        "--save-example",
        dest="save_example",
        required=False,
        action="store_true",
        help="Will create a directory with example input matrices.",
    )

    freq_plot_parser = subparsers.add_parser(
        name="plot_freqs",
        description="Create a bedfile of major and minor allele frequency changes over time.",
    )
    freq_plot_parser.add_argument(
        "-i",
        "--input",
        dest="input",
        metavar="INPUT VCF FILE",
        type=str,
        required=True,
        help="Merged time-series VCF file to pull SNPs and frequencies from.",
    )
    freq_plot_parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT FILE PREFIX",
        dest="output",
        required=False,
        default=sys.stdout,
        type=str,
        help="""Bedgraph file prefix, two files will be written using this prefix + '{.major,.minor}.bedGraph.
        The '.minor' file denotes the 4th column is the frequency change from last to first timepoints of the allele with the largest change over those epochs.
        The '.major' file denotes the 4th column is the frequency change from last to first timepoints of 1-minor allele at each SNP, with the 'minor' allele being described above as the highest-velocity allele across timepoints in a given SNP.
        Example: 'ts_output/d_simulans_experiment_1'.
        """,
    )
    freq_plot_parser.add_argument(
        "-y",
        "--yaml",
        metavar="YAML_CONFIG",
        required=True,
        dest="yaml_file",
        help="YAML config file with all required options defined.",
    )

    # parse_sim_logs.py
    summarize_parser = subparsers.add_parser(
        name="summarize",
        description="Creates a CSV of data parsed from slim log files.",
    )
    summarize_parser.add_argument(
        "--threads",
        required=False,
        type=int,
        default=mp.cpu_count(),
        dest="threads",
        help="Number of processes to parallelize across. Defaults to all.",
    )
    summarize_parser.add_argument(
        "-n",
        "--experiment-name",
        metavar="EXPERIMENT_NAME",
        dest="experiment_name",
        type=str,
        required=False,
        default="ts_experiment",
        help="Identifier for the experiment used to generate the data. Optional, but helpful in differentiating runs.",
    )
    summarize_parser.add_argument(
        "-y",
        "--yaml",
        metavar="YAML_CONFIG",
        required=True,
        dest="yaml_file",
        help="YAML config file with all required options defined.",
    )

    # merge_locs_acc.py
    merge_parser = subparsers.add_parser(
        name="merge_logs",
        description="Merges the summary TSV from SLiM logs with test data predictions.",
    )
    merge_parser.add_argument(
        "-s",
        "--summary-tsv",
        dest="summary_tsv",
        help="TSV created using `parse_slim_logs.py`.",
        required=True,
    )
    merge_parser.add_argument(
        "-i",
        "--input-test-file",
        dest="test_file",
        help="Files to merge ",
        required=True,
    )
    merge_parser.add_argument(
        "-o",
        "--output-dir",
        metavar="OUTPUT DIR",
        dest="output_dir",
        required=True,
        type=str,
        help="Directory to write merged results and plots to.",
    )

    ua = agp.parse_args()

    # fmt: off
    if ua.mode == "sim_stdpopsim":
        from . import simulate_stdpopsim
        simulate_stdpopsim.main(ua)

    elif ua.mode == "sim_custom":
        from . import simulate_custom
        simulate_custom.main(ua)

    elif ua.mode == "condense":
        from . import make_training_features
        make_training_features.main(ua)    

    elif ua.mode == "train":
        if ua.single_tp:
            from . import tp1_model
            tp1_model.main(ua)   
        
        else:
            from . import train_shoulder_nets
            train_shoulder_nets.main(ua)  
        
    elif ua.mode == "detect":
        from . import find_sweeps_vcf as find_sweeps_vcf
        find_sweeps_vcf.main(ua)   

    elif ua.mode == "plot_training":
        from .plotting import plot_training_data as plot_training
        plot_training.main(ua)   

    elif ua.mode == "plot_freqs":
        from .plotting import create_freq_track as cf
        cf.main(ua)
        
    elif ua.mode == "summarize":
        from . import parse_slim_logs as psl
        psl.main(ua)
    
    elif ua.mode == "merge_logs":
        from . import merge_logs_acc as mla
        mla.main(ua)
        
    elif ua.mode == None:
        agp.print_help()


if __name__ == "__main__":
    ts_main()
