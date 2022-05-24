import argparse
import multiprocessing as mp
import os


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
    sim_s_subparsers = sim_s_parser.add_subparsers(dest="config_format")
    sim_s_subparsers.required = True
    sim_s_yaml_parser = sim_s_subparsers.add_parser("yaml")
    sim_s_yaml_parser.add_argument(
        metavar="YAML CONFIG",
        dest="yaml_file",
        help="YAML config file with all cli options defined.",
    )

    sim_s_cli_parser = sim_s_subparsers.add_parser("cli")
    sim_s_cli_parser.add_argument(
        "-i",
        "--slim-file",
        required=True,
        type=str,
        help="SLiM Script output by stdpopsim to add time-series sampling to.",
        dest="slim_file",
    )
    sim_s_cli_parser.add_argument(
        "--reps",
        required=True,
        type=int,
        help="Number of replicate simulations to run. If using rep_range can just fill with random int.",
        dest="reps",
    )
    sim_s_cli_parser.add_argument(
        "--pop",
        required=False,
        type=str,
        default="p2",
        dest="pop",
        help="Label of population to sample from, will be defined in SLiM Script. Defaults to p2.",
    )
    sim_s_cli_parser.add_argument(
        "--sample_sizes",
        required=True,
        type=int,
        nargs="+",
        dest="sample_sizes",
        help="Number of individuals to sample without replacement at each sampling point. Must match the number of entries in the -y flag.",
    )
    sim_s_cli_parser.add_argument(
        "--years-sampled",
        required=True,
        type=int,
        nargs="+",
        dest="years_sampled",
        help="Years BP (before 1950) that samples are estimated to be from. Must match the number of entries in the --sample-sizes flag.",
    )
    sim_s_cli_parser.add_argument(
        "--selection-generation",
        required=False,
        type=int,
        default=200,
        dest="sel_gen",
        help="Number of gens before first sampling to introduce selection in population. Defaults to 200.",
    )
    sim_s_cli_parser.add_argument(
        "--selection-coeff-bounds",
        required=False,
        type=float,
        default=[0.005, 0.5],
        dest="sel_coeff_bounds",
        action="append",
        nargs=2,
        help="Bounds of log-uniform distribution for pulling selection coefficient of mutation being introduced. Defaults to [0.005, 0.5]",
    )
    sim_s_cli_parser.add_argument(
        "--mut-rate",
        required=False,
        type=str,
        default="1.29e-8",
        dest="mut_rate",
        help="Mutation rate for mutations not being tracked for sweep detection. Defaults to 1.29e-8 as defined in stdpopsim for OoA model.",
    )
    sim_s_cli_parser.add_argument(
        "--work-dir",
        required=False,
        type=str,
        default="./ts_experiment",
        dest="work_dir",
        help="Directory to start workflow in, subdirs will be created to write simulation files to. Will be used in downstream processing as well.",
    )
    sim_s_cli_parser.add_argument(
        "--slim-path",
        required=False,
        type=str,
        default="slim",
        dest="slim_path",
        help="Path to SLiM executable.",
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
    sim_c_subparsers = sim_c_parser.add_subparsers(dest="config_format")
    sim_c_subparsers.required = True
    sim_c_yaml_parser = sim_c_subparsers.add_parser("yaml")
    sim_c_yaml_parser.add_argument(
        metavar="YAML_CONFIG",
        dest="yaml_file",
        help="YAML config file with all cli options defined.",
    )

    sim_c_cli_parser = sim_c_subparsers.add_parser("cli")
    sim_c_cli_parser.add_argument(
        "-w",
        "--work-dir",
        dest="work_dir",
        type=str,
        help="Directory used as work dir for simulate modules. Should contain simulated vcfs processed using process_vcf.py.",
        required=False,
        default=os.getcwd(),
    )
    sim_c_cli_parser.add_argument(
        "-i",
        "--slim-file",
        required=True,
        type=str,
        help="SLiM Script to simulate with. Must output to a single VCF file. ",
        dest="slim_file",
    )
    sim_c_cli_parser.add_argument(
        "--slim-path",
        required=False,
        type=str,
        default="slim",
        dest="slim_path",
        help="Path to SLiM executable.",
    )
    sim_c_cli_parser.add_argument(
        "--reps",
        required=False,
        type=int,
        help="Number of replicate simulations to run if not using rep-range.",
        dest="reps",
    )

    # process_vcfs.py
    vcfproc_parser = subparsers.add_parser(
        name="process",
        description="Splits and re-merges VCF files to prepare for fast feature creation.",
    )
    vcfproc_parser.add_argument(
        "--vcf-header",
        required=False,
        type=str,
        default="##fileformat=VCFv4.2",
        dest="vcf_header",
        help="String that tops VCF header, used to split entries to new files.",
    )
    vcfproc_parser.add_argument(
        "--threads",
        required=False,
        type=int,
        default=mp.cpu_count() - 1,
        dest="threads",
        help="Number of processes to parallelize across.",
    )

    vcf_subparsers = vcfproc_parser.add_subparsers(dest="config_format")
    vcf_subparsers.required = True
    vcf_yaml_parser = vcf_subparsers.add_parser("yaml")
    vcf_yaml_parser.add_argument(
        metavar="YAML CONFIG",
        dest="yaml_file",
        help="YAML config file with all cli options defined.",
    )

    vcf_cli_parser = vcf_subparsers.add_parser("cli")
    vcf_cli_parser.add_argument(
        "-w",
        "--work-dir",
        dest="work_dir",
        type=str,
        help="Directory used as work dir for simulate modules. Should contain simulated vcfs processed using process_vcf.py.",
        required=False,
        default=os.getcwd(),
    )
    vcf_cli_parser.add_argument(
        "--sample_sizes",
        required=True,
        type=int,
        nargs="+",
        dest="sample_sizes",
        help="Number of individuals to sample without replacement at each sampling point. Must match the number of entries in the -y flag.",
    )

    # make_training_features.py
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
        help="If given, only include sim replicates where the sweep site has a minimum increase of <freq_inc_thr> from the first timepoint to the last.",
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
        help="Whether to print error messages, usually from VCF loading errors.",
    )
    mtf_subparsers = mtf_parser.add_subparsers(dest="config_format")
    mtf_subparsers.required = True

    mtf_yaml_parser = mtf_subparsers.add_parser("yaml")
    mtf_yaml_parser.add_argument(
        metavar="YAML CONFIG",
        dest="yaml_file",
        help="YAML config file with all cli options defined.",
    )

    mtf_cli_parser = mtf_subparsers.add_parser("cli")
    mtf_cli_parser.add_argument(
        "-w",
        "--work-dir",
        dest="work_dir",
        type=str,
        help="Directory used as work dir for simulate modules. Should contain simulated vcfs processed using process_vcf.py.",
        required=False,
        default=os.getcwd(),
    )
    mtf_cli_parser.add_argument(
        "-s",
        "--sample-sizes",
        dest="samp_sizes",
        help="Number of individuals from each timepoint sampled. Used to index VCF data from earliest to latest sampling poinfs.",
        required=True,
        nargs="+",
        type=int,
    )
    mtf_cli_parser.add_argument(
        "-p",
        "--ploidy",
        dest="ploidy",
        help="Ploidy of organism being sampled.",
        default="2",
        type=int,
    )
    # nets.py
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
        "-n",
        "--experiment-name",
        metavar="EXPERIMENT_NAME",
        dest="experiment_name",
        type=str,
        required=False,
        default="ts_experiment",
        help="Identifier for the experiment used to generate the data. Optional, but helpful in differentiating runs.",
    )

    nets_subparsers = nets_parser.add_subparsers(dest="config_format")
    nets_subparsers.required = True
    nets_yaml_parser = nets_subparsers.add_parser("yaml")
    nets_yaml_parser.add_argument(
        metavar="YAML CONFIG",
        dest="yaml_file",
        help="YAML config file with all cli options defined.",
    )

    nets_cli_parser = nets_subparsers.add_parser("cli")
    nets_cli_parser.add_argument(
        "-w",
        "--work-dir",
        dest="work_dir",
        type=str,
        help="Directory used as work dir for simulate modules. Should contain pickled training data from simulated vcfs processed using process_vcf.py.",
        required=False,
        default=os.getcwd(),
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
        "-o",
        "--out-dir",
        dest="outdir",
        help="Directory to write output to.",
        required=True,
    )
    sweeps_subparsers = sweeps_parser.add_subparsers(dest="config_format")
    sweeps_subparsers.required = True

    sweeps_yaml_parser = sweeps_subparsers.add_parser("yaml")
    sweeps_yaml_parser.add_argument(
        metavar="YAML CONFIG",
        dest="yaml_file",
        help="YAML config file with all cli options defined.",
    )

    sweeps_cli_parser = sweeps_subparsers.add_parser("cli")
    sweeps_cli_parser.add_argument(
        "-s",
        "--sample-sizes",
        dest="samp_sizes",
        help="Number of individuals from each timepoint sampled. Used to index VCF data from earliest to latest sampling points.",
        required=True,
        nargs="+",
        type=int,
    )
    sweeps_cli_parser.add_argument(
        "-p",
        "--ploidy",
        dest="ploidy",
        help="Ploidy of organism being sampled.",
        default="2",
        type=int,
    )
    sweeps_cli_parser.add_argument(
        "-w",
        "--work-dir",
        metavar="WORKING_DIR",
        dest="work_dir",
        type=str,
        help="Working directory for workflow, should be identical to previous steps.",
    )

    sweeps_cli_parser.add_argument(
        "--years-sampled",
        required=False,
        type=int,
        nargs="+",
        dest="years_sampled",
        default=None,
        help="Years BP (before 1950) that samples are estimated to be from. Only used for FIT calculations, and is optional if you don't care about those.",
    )
    sweeps_cli_parser.add_argument(
        "--gen-time",
        required=False,
        type=int,
        dest="gen_time",
        default=None,
        help="Generation time to multiply years_sampled by. Similarly to years_sampled, only used for FIT calculation and is optional.",
    )

    #plot_training_data.py
    input_plot_parser = subparsers.add_parser(
        name="plot_training",
        description="Plots central SNPs from simulations to visually inspect mean trends over replicates."
    )

    input_plot_parser.add_argument(
        "-i",
        "--input-pickle",
        dest="input_pickle",
        metavar="INPUT PICKLE",
        type=str,
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

    ua = agp.parse_args()

    #fmt: off
    if ua.mode == "sim_stdpopsim":
        from . import simulate_stdpopsim
        simulate_stdpopsim.main(ua)

    elif ua.mode == "sim_custom":
        from . import simulate_custom
        simulate_custom.main(ua)

    elif ua.mode == "process":
        from . import process_vcfs
        process_vcfs.main(ua)
        
    elif ua.mode == "condense":
        from . import make_training_features
        make_training_features.main(ua)    

    elif ua.mode == "train":
        from . import nets
        nets.main(ua)  
        
    elif ua.mode == "detect":
        from . import find_sweeps_vcf as find_sweeps_vcf
        find_sweeps_vcf.main(ua)   

    elif ua.mode == "plot_training":
        from plotting import plot_training_data as plot_training
        plot_training.main(ua)   

    elif ua.mode == None:
        agp.print_help()


if __name__ == "__main__":
    ts_main()
