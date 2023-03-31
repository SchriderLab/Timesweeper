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
        help="Injects time-series sampling into stdpopsim SLiM output script.",
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
        help="Simulates selection for training Timesweeper using a pre-made SLiM script.",
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

    # process_vcfs.py
    process_vcf_parser = subparsers.add_parser(
        name="process",
        help="Module for splitting multivcfs (vertically concatenated vcfs) into merged (horizontally concatenated vcfs) if simulating without the sim module.",
    )
    process_vcf_parser.add_argument(
        "-i",
        "--in-dir",
        dest="in_dir",
        help="Top-level directory containing subidrectories labeled with the names of each scenario and then replicate numbers inside those containing each multivcf.",
        required=True,
    )
    process_vcf_parser.add_argument(
        "-t",
        "--threads",
        dest="threads",
        help="Threads to use for multiprocessing.",
        required=True,
    )   
     
    # make_training_features.py
    mtf_parser = subparsers.add_parser(
        name="condense",
        help="Creates training data from simulated merged vcfs after process_vcfs.py has been run.",
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
        dest="allow_shoulders",
        action="store_true",
        help="1/3 of samples from sweep classes will be offset to be used as neutral shoulders.",
    )
    mtf_parser.add_argument(
        "--paramsfile",
        dest="paramsfile",
        help="Use a params file from the summarize module to make up for erroneous VCFs that don't correctly report selection coefficients.",
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
        help="Handler script for neural network training and prediction for TimeSweeper Package.\
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
        "-s",
        "--subsample-amount",
        metavar="SUBSAMPLE_AMOUNT",
        dest="subsample_amount",
        type=int,
        required=False,
        help="Amount of data to subsample for each class to test for sample size effects.",
    )
    nets_parser.add_argument(
        "-m",
        "--model-type",
        metavar="MODEL TYPE",
        dest="model_type",
        type=str,
        required=False,
        default="1dcnn",
        choices=['1dcnn', '2dcnn', 'chonk', 'rnn'],
        help="Architecture to use when training the model.",
    )
    nets_parser.add_argument(
        "--hft",
        required=False,
        action="store_true",
        dest="hft",
        help="Whether to train HFT alongside AFT. Computationally more expensive.",
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
        "--shic",
        dest="shic",
        action="store_true",
        help="Whether to use the shic module for a special case experiment.",
    )
    
    # find_sweeps.py
    sweeps_parser = subparsers.add_parser(
        name="detect",
        help="Module for iterating across windows in a time-series vcf file and predicting whether a sweep is present at each snp-centralized window.",
    )
    sweeps_parser.add_argument(
        "-i",
        "--input-vcf",
        dest="input_vcf",
        help="Merged VCF to scan for sweeps. Must be merged VCF where files are merged in order from earliest to latest sampling time, -0 flag must be used.",
        required=True,
    )
    sweeps_parser.add_argument(
        "--hft",
        required=False,
        action="store_true",
        dest="hft",
        help="Whether to train HFT alongside AFT. Computationally more expensive.",
    )
    sweeps_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Directory to write results to.",
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
        "-y",
        "--yaml",
        metavar="YAML_CONFIG",
        required=True,
        dest="yaml_file",
        help="YAML config file with all required options defined.",
    )


    # find_sweeps_npz.py
    npz_sweeps_parser = subparsers.add_parser(
        name="detect-npz",
        help="Module for iterating across windows in a time-series vcf file and predicting whether a sweep is present at each snp-centralized window.",
    )
    npz_sweeps_parser.add_argument(
        "-i",
        "--input-npz",
        dest="input_file",
        help="Merged VCF to scan for sweeps. Must be merged VCF where files are merged in order from earliest to latest sampling time, -0 flag must be used.",
        required=True,
    )
    npz_sweeps_parser.add_argument(
        "-o",
        "--output-dir",
        dest="outdir",
        help="Directory to write results to.",
        required=True,
    )
    npz_sweeps_parser.add_argument(
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
        help="Plots central SNPs from simulations to visually inspect mean trends over replicates.",
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
        "--save-example",
        dest="save_example",
        required=False,
        action="store_true",
        help="Will create a directory with example input matrices.",
    )
    input_plot_parser.add_argument(
        "-y",
        "--yaml",
        metavar="YAML_CONFIG",
        required=True,
        dest="yaml_file",
        help="YAML config file with all required options defined.",
    )

    #input plotter
    freq_plot_parser = subparsers.add_parser(
        name="plot_freqs",
        help="Create a bedfile of major and minor allele frequency changes over time.",
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
        help="Creates a CSV of data parsed from slim log files.",
    )
    summarize_parser.add_argument(
        "--threads",
        required=False,
        type=int,
        default=16,
        dest="threads",
        help="Number of processes to parallelize across. Defaults to all.",
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
        help="Merges the summary TSV from SLiM logs with test data predictions.",
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
        from timesweeper import simulate_stdpopsim
        simulate_stdpopsim.main(ua)

    elif ua.mode == "sim_custom":
        from timesweeper import simulate_custom
        simulate_custom.main(ua)
    
    elif ua.mode == "process":
        from timesweeper import process_vcfs
        process_vcfs.main(ua)    

    elif ua.mode == "condense":
        from timesweeper import make_training_features
        make_training_features.main(ua)    

    elif ua.mode == "train":
        if ua.shic:
            from timesweeper import train_nets_shic
            train_nets_shic.main(ua)
        else:
            from timesweeper import train_nets
            train_nets.main(ua)
        
    elif ua.mode == "detect":
        from timesweeper import find_sweeps_vcf as find_sweeps_vcf
        find_sweeps_vcf.main(ua)   

    elif ua.mode == "detect-npz":
        from timesweeper import find_sweeps_npz as find_sweeps_npz
        find_sweeps_npz.main(ua)   

    elif ua.mode == "plot_training":
        from timesweeper.plotting import plot_training_data as plot_training
        plot_training.main(ua)   

    elif ua.mode == "plot_freqs":
        from timesweeper.plotting import create_freq_track as cf
        cf.main(ua)
        
    elif ua.mode == "summarize":
        from timesweeper import parse_slim_logs as psl
        psl.main(ua)
    
    elif ua.mode == "merge_logs":
        from timesweeper import merge_logs_acc as mla
        mla.main(ua)
        
    elif ua.mode == None:
        agp.print_help()


if __name__ == "__main__":
    ts_main()
