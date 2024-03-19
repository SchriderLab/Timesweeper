# TimeSweeper

Timesweeper is a package for detecting positive selective sweeps from time-series genomic sampling using convolutional neural networks.

The associated manuscript can be found here: https://www.biorxiv.org/content/10.1101/2022.07.06.499052v1
Experiments, figures, and trained networks for the Timesweeper manuscript can be found here: https://github.com/SchriderLab/timesweeper-experiments

## Table of Contents
- [TimeSweeper](#timesweeper)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Workflow Overview](#workflow-overview)
  - [Installation and Requirements](#installation-and-requirements)
  - [Timesweeper Configuration](#timesweeper-configuration)
    - [Configs required for both types of simulation:](#configs-required-for-both-types-of-simulation)
    - [Additional configs needed for stdpopsim simulation:](#additional-configs-needed-for-stdpopsim-simulation)
  - [Timesweeper Modules Details](#timesweeper-modules-details)
    - [Overview](#overview-1)
    - [Custom Simulation (`simulate_custom`)](#custom-simulation-simulate_custom)
    - [stpopsim Simulation (`simulate_stdpopsim`)](#stpopsim-simulation-simulate_stdpopsim)
    - [Process VCF Files (`process`)](#process-vcf-files-process)
    - [Make Training Data (`condense`)](#make-training-data-condense)
    - [Neural Networks (`train`)](#neural-networks-train)
    - [Detect Sweeps (`detect`)](#detect-sweeps-detect)
  - [Preparing Input Data for Timesweeper to Predict On](#preparing-input-data-for-timesweeper-to-predict-on)
  - [Example Usage](#example-usage)
  - [Using Non-Timesweeper Simulations](#using-non-timesweeper-simulations)

## Overview

Timesweeper is built as a series of modules that are chained together to build a workflow for detecting signatures of selective sweeps using simulated demographic models to train a 1D convolutional neural network (1DCNN). Some modules can be swapped out or modified for different use cases (such as simulating from a custom SLiM script), but the general workflow order is assumed to be consistent. 

## Workflow Overview

1. Create a SLiM script
   1. Either based on the `example_demo_model.slim` example 
   2. Or by using stdpopsim to generate a SLiM script
2. Simulate demographic model with time-series sampling
   1. `sim_custom` if using custom SLiM script
   2. `sim_stdpopsim` if using a SLiM script output by stdpopsim
   - Note: If available, we suggest using a job submission platform such as SLURM to parallelize simulations. This is the most resource and time-intensive part of the module by far.
   - Optional: Preprocess VCFs simulated without timesweepers simulation modules by merging with `process_vcfs`
4. Create features for the neural network with `condense`
5. Train networks with `train`
6. Run `detect` on VCF of interest using trained models and input data

---

## Installation and Requirements

The only required utility not packaged with the PyPi installation of Timesweeper is SLiM, which can be easily installed with conda through the conda-forge channel (see below) or as a binary (https://messerlab.org/slim/).

Otherwise see either [setup.cfg](setup.cfg) or [requirements.txt](requirements.txt) for general and specific requirements respectively.

Timesweeper and all requirements can be installed from pip, I recommend doing so inside a virtual environment along with SLiM for easy access when simulating:
```{bash}
conda create -n blinx -c conda-forge python slim=3.7
conda activate blinx
pip install timesweeper
```

---

## Timesweeper Configuration

For any given experiment run you will need a YAML configuration file (see `example_timesweeper_config.yaml` for template). The requirements for configuration are different between using stdpopsim and custom SLiM script simulations.

### Configs required for both types of simulation:

Example config file for a custom simulation run:

```
work dir: win_size_sims/
slimfile: /slimfiles/onePop-selectiveSweep-vcf-noselcoeff.slim
slim path: slim
mut types: [2]
scenarios: ["neut", "sdn", "ssv"]
win_size: 51
num_sample_points : 20
inds_per_tp : 10  # Diploid inds
sample sizes: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
ploidy: 2
physLen : 5000000
reps: 30000
```

Example config file for a stdpopsim simulation run:

```
work dir: OoA
slimfile: OoA3pop.slim
pop: p2
sample sizes: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
years sampled: [4750, 4500, 4250, 4000, 3750, 3500, 3250, 3000, 2750, 2500, 2250, 2000, 1750, 1500, 1250, 1000, 750, 500, 250, 0]
ploidy: 2
scenarios: ["neut", "sdn", "ssv"]
mut types: [2]
win_size: 51
inds_per_tp: 10
num_sample_points : 20
reps: 10000
selection gen: 50 # Gens before first sampling
selection coeff bounds: [0.00025, 0.25]
mut rate: 1.29e-8
slim path: slim
gen time: 25
```

- **Working Directory** (`work dir`) - this serves as the "home base" of an experiment. This is where all intermediate and final data will be written to.
- **SLiM Script** (`slimfile`) - either generated by stdpopsim (see "Using stdpopsim to generate a SLiM template" below) or from a custom SLiM script (see "Using a custom SLiM script").
- - **Path to SLiM Executable** (`slim path`) - if you use the Makefile will be `/<path>/<to>/<timesweeper>/SLiM/build/slim`
- **Mutation types** (`mut types`) - target mutation types output by SLiM in outputVCFSample. Typically MT=1 is for neutral mutations and MT=2 is for selected alleles.
- **Scenarios** (`scenarios`) - Names of the types of simulation scenarios input into Timesweeper. The standard (and maybe only?) option for current implementation is `["neut", "ssv", "sdn"]`.
- **Window size** (`win_size`) - Number of SNPs to use to create feature vectors.
- **Number of Sample Points** (`num_sample_points`) - Number of timepoints sampled, identical to `len(sample_sizes)`.
- **Individuals per timepoint** (`inds_per_tp`) - Another lazy variable to do some quick calculations, variable sampling sizes is supported but this is used for some quick calculations. Will probably be removed at some point when it causes errors.
- **Sample sizes of each timepoint** (`sample sizes`) - sampled in your data (i.e. how many individuals are included in timepoint 1, timepoint 2, etc)
- **Ploidy** (`ploidy`) - ploidy of your samples.
- **Physical size** (`physLen`) - Size of the chromosome to simulate. Will be overwritten by stdpopsim if used.
- **Simulation Replicates** (`reps`) - for each scenario: neutral, selection on de novo mutation, selection on standing variation. Will be overwritten with `--rep-range` argument if doing parallelized sims.


### Additional configs needed for stdpopsim simulation: 
- **Target Population ID** (`pop`) - will be something like "p1" or "p2", will be identified in the stdpopsim model catalog.
- **Years Sampled** (`years sampled`) - typically in years Before Present (BP - which is years prior to 1950 apparently). This is only needed if you are injecting sampling into a stdpopsim model or if you'd like to calculate FIT values along with the AFT predictions with Timesweeper. If you don't have any idea what these values should be you can still run Timesweeper and get AFT predictions no problem, just use a custom SLiM script and remove this line from the config file.
- **Selection Generation Before Sampling** (`selection gen`) - Number of generations before the __first__ sample timepoint. We've found that you can be relatively flexible (see the manuscript), but 50 is a good default value.
- **Selection Coefficient Bounds** (`selection coeff bounds`) - to improve robustness we use a uniform distribution to draw selection coefficients with lower and upper bounds specified here. If you want a non-stochastic selection coefficient simply use the same number twice.
- **Mutation Rate** (`mut rate`) - just overwrites the stdpopsim mutation rate in case you'd like to fiddle with it. 
- **Generation Time** (`gen time`) - allows conversions between generations and continuous time. 
- **Selection Generation** (`selection gen`) - Number of generations to start sampling (with a random offset each replicate) before selection.


---

## Timesweeper Modules Details

### Overview
```
$ timesweeper
usage: timesweeper [-h] {sim_stdpopsim,sim_custom,process,condense,train,detect,plot_training,plot_freqs,summarize,merge_logs} ...

Timesweeper CLI

positional arguments:
  {sim_stdpopsim,sim_custom,process,condense,train,detect,plot_training,plot_freqs,summarize,merge_logs}
                        Timesweeper modes
    sim_stdpopsim       Injects time-series sampling into stdpopsim SLiM output script.
    sim_custom          Simulates selection for training Timesweeper using a pre-made SLiM script.
    process             Module for splitting multivcfs (vertically concatenated vcfs) into merged (horizontally concatenated vcfs) if simulating
                        without the sim module.
    condense            Creates training data from simulated merged vcfs after process_vcfs.py has been run.
    train               Handler script for neural network training and prediction for TimeSweeper Package. Will train two models: one for the series
                        of timepoints generated using the hfs vectors over a timepoint and one
    detect              Module for iterating across windows in a time-series vcf file and predicting whether a sweep is present at each snp-
                        centralized window.
    plot_training       Plots central SNPs from simulations to visually inspect mean trends over replicates.
    plot_freqs          Create a bedfile of major and minor allele frequency changes over time.
    summarize           Creates a CSV of data parsed from slim log files.
    merge_logs          Merges the summary TSV from SLiM logs with test data predictions.

options:
  -h, --help            show this help message and exit
```

### Custom Simulation (`simulate_custom`) 

A flexible wrapper for a SLiM script that assumes you have a demographic model already defined in SLiM and would like to use it for Timesweeper. This module is meant to be modified with utility functions and other constants to feed into simulation replicates using the string variable `d_block`. This set of constants will be fed to SLiM at runtime but can be modified in the module as needed prior. The module on GitHub is the version used to generate simulations for the manuscript, and is meant to provide ideas/context for how it's possible to modify the `d_block` and write utility functions for it.

  There are some basic requirements for using a custom SLiM script:
  1. Each timepoint must be output using a the outputVCFSample function like so: `<pop_id>.outputVCFSample(<num_individuals>, replace=T, filePath=outFile, append=T);`
  2. The constants `sweep`, `outFile`, and `dumpFile` must be used in your simulation script
      - `sweep`: one of "neut"/"sdn"/"soft", equivalent to neutral, selection on *de novo* mutation, and selection on standing variation respectively. This identifier is used both in the SLiM script to condition on scenarios but also in the output file naming.
      - `outFile`: is the VCF file that will be used as output for samples for a given replicate. Will be set as `<work_dir>/vcfs/<sweep>/<rep>.multivcf`
      - `dumpFile`: similarly to outFile this is where the intermediate simulation state is saved to in case of mutation loss or other problems with a replicate.

  There is an example SLiM simulation in [constant_population.slim](timesweeper/constant_population.slim) that has been used and templated/modified for many experiments for the manuscript and is ready to be used with the current `d_block` setup in `sim_custom.py`.

```
$ timesweeper sim_custom -h
usage: timesweeper sim_custom [-h] [--threads THREADS]
                              [--rep-range REP_RANGE REP_RANGE] -y YAML_CONFIG

Simulates selection for training Timesweeper using a pre-made SLiM script.

options:
  -h, --help            show this help message and exit
  --threads THREADS     Number of processes to parallelize across. Defaults to
                        all.
  --rep-range REP_RANGE REP_RANGE
                        <start, stop>. If used, only range(start, stop) will
                        be simulated for reps. This is to allow for easy SLURM
                        parallel simulations.
  -y YAML_CONFIG, --yaml YAML_CONFIG
                        YAML config file with all required options defined.

```


### stpopsim Simulation (`simulate_stdpopsim`) 

For use with SLiM scripts that have been generated using stdpopsim's `--slim-script` option to output the model. This allows for out of the box demographic models downloaded straight from the catalog stdpopsim adds to regularly. Some information needs to be gotten from the model definition so that the wrapper knows which population to sample from, how to scale values if rescaling the simulation, and more. These are described in detail both in the help message of the module and in the above doc section [Configs required for both types of simulation](#configs-required-for-both-types-of-simulation).

```
$ timesweeper sim_stdpopsim -h
usage: timesweeper sim_stdpopsim [-h] [-v] [--threads THREADS]
                                 [--rep-range REP_RANGE REP_RANGE]
                                 YAML CONFIG

Injects time-series sampling into stdpopsim SLiM output script.

positional arguments:
  YAML CONFIG           YAML config file with all options defined.

options:
  -h, --help            show this help message and exit
  -v, --verbose         Print verbose logging during process.
  --threads THREADS     Number of processes to parallelize across.
  --rep-range REP_RANGE REP_RANGE
                        <start, stop>. If used, only range(start, stop) will
                        be simulated for reps. This is to allow for easy SLURM
                        parallel simulations.
```

### Process VCF Files (`process`) 

This module splits the multivcf files (which are just multiple concatenated VCF entries) generated by SLiM and then merges them in order from most ancient to most current timepoints. **NOTE:** This is already integrated into `sim_custom` and `sim_stdpopsim` but can be used separately if you would like to simulate without the sim_custom or sim_stdpopsim modules.

```
$ timesweeper process -h
usage: timesweeper process [-h] -i IN_DIR -t THREADS

Module for splitting multivcfs (vertically concatenated vcfs) into merged
(horizontally concatenated vcfs) if simulating without the sim module.

options:
  -h, --help            show this help message and exit
  -i IN_DIR, --in-dir IN_DIR
                        Top-level directory containing subidrectories labeled
                        with the names of each scenario and then replicate
                        numbers inside those containing each multivcf.
  -t THREADS, --threads THREADS
                        Threads to use for multiprocessing.

```

### Make Training Data (`condense`) 

VCFs merged using `timesweeper process` are read in as allele frequencies using scikit-allel, and depending on the scenario (neut/sdn/soft) the central or locus under selection is pulled out and aggregated for all replicates. This labeled ground-truth data from simulations is then saved as a dictionary in a pickle file for easy access and low disk usage. 

This module also allows for adding missingness to the training data in the case of missingness in the real data Timesweeper is going to be used on. To do this add the `-m <val>` flag where `val` is in [0,1] and is used as the parameter of a binomial draw for each allele per timestep to set as present/missing. We show in the manuscript that some missingness is viable (e.g. `val=0.2`), however high missingness (e.g. `val=0.5`) will result in terrible performance and should be avoided. Optimally this value should reflect the missingness present in the real data input to Timesweeper so as to parameterize the network to be better prepared for it.

Note: the process of retrieving known-selection sites is based on the mutation type labels contained in VCF INFO fields output by SLiM. It currently assumes the mutation type where selection is being introduced is identified as "m2", but if you use a custom SLiM model and happen to change mutation type this module should be modified to properly scan for that.

Warning: This module is the primary user of the `win_size` parameter. We recommend a window size of 51 based on our testing, but note that the current implementation of the 1DCNN fails to classify anything properly at sizes larger than k=101 in our testing, so user discretion and testing is advised when attempting to use larger windows.

```
$ timesweeper condense -h
usage: timesweeper condense [-h] [--threads THREADS] [-o OUTFILE]
                            [--subsample-inds SUBSAMPLE_INDS]
                            [--subsample-tps SUBSAMPLE_TPS] [--og-tps OG_TPS]
                            [-m MISSINGNESS] [-f FREQ_INC_THRESHOLD] [-a]
                            [--hft] [--no-progress] [--verbose] -y YAML_CONFIG

Creates training data from simulated merged vcfs after process_vcfs.py has
been run.

options:
  -h, --help            show this help message and exit
  --threads THREADS     Number of processes to parallelize across.
  -o OUTFILE, --outfile OUTFILE
                        Pickle file to dump dictionaries with training data
                        to. Should probably end with .pkl.
  --subsample-inds SUBSAMPLE_INDS
                        Number of individuals to subsample if using a larger
                        simulation than needed for efficiency. NOTE: If you
                        use this, keep the sample_sizes entry in the yaml
                        config identical to the original simulations. This is
                        mostly for the paper experiments and as such isn't
                        very cleanly implemented.
  --subsample-tps SUBSAMPLE_TPS
                        Number of timepoints to subsample if using a larger
                        simulation than needed for efficiency. NOTE: If you
                        use this, keep the sample_sizes entry in the yaml
                        config identical to the original simulations. This is
                        mostly for the paper experiments and as such isn't
                        very cleanly implemented.
  --og-tps OG_TPS       Number of timepoints taken in the original data to be
                        subsetted if using --subample-tps.
  -m MISSINGNESS, --missingness MISSINGNESS
                        Missingness rate in range of [0,1], used as the
                        parameter of a binomial distribution for randomly
                        removing known values.
  -f FREQ_INC_THRESHOLD, --freq-increase-threshold FREQ_INC_THRESHOLD
                        If given, only include sim replicates where the sweep
                        site has a minimum increase of <freq_inc_thr> from the
                        first timepoint to the last.
  -a, --allow-shoulders
                        1/3 of samples from sweep classes will be offset to be
                        used as neutral shoulders.
  --hft                 Whether to calculate HFT alongside AFT.
                        Computationally more expensive.
  --no-progress         Turn off progress bar.
  --verbose             Raise warnings from issues usually stemming from bad
                        replicates.
  -y YAML_CONFIG, --yaml YAML_CONFIG
                        YAML config file with all required options defined.

```

### Neural Networks (`train`) 

Timesweeper's neural network architecture is a shallow 1DCNN implemented in Keras2 with a Tensorflow backend that trains extremely fast on CPUs with very little RAM needed. Assuming all previous steps were run it can be trained and evaluated on hold-out test data with a single line invocation.

```
$ timesweeper train -h
usage: timesweeper train [-h] -i TRAINING_DATA -d DATA_TYPE
                         [-s SUBSAMPLE_AMOUNT] [-n EXPERIMENT_NAME] -y
                         YAML_CONFIG [--single-tp]

Handler script for neural network training and prediction for TimeSweeper
Package. Will train two models: one for the series of timepoints generated
using the hfs vectors over a timepoint and one

options:
  -h, --help            show this help message and exit
  -i TRAINING_DATA, --training-data TRAINING_DATA
                        Pickle file containing data formatted with
                        make_training_features.py.
  -d DATA_TYPE, --data-type DATA_TYPE
                        AFT or HFT data preparation.
  -s SUBSAMPLE_AMOUNT, --subsample-amount SUBSAMPLE_AMOUNT
                        Amount of data to subsample for each class to test for
                        sample size effects.
  -n EXPERIMENT_NAME, --experiment-name EXPERIMENT_NAME
                        Identifier for the experiment used to generate the
                        data. Optional, but helpful in differentiating runs.
  -y YAML_CONFIG, --yaml YAML_CONFIG
                        YAML config file with all required options defined.
  --single-tp           Whether to use the tp1_model module for a special case
                        experiment.

```

### Detect Sweeps (`detect`) 

Finally, the main module of the package is for detecting sweeps in a given VCF. This loads in the prepared VCF (see "Preparing Input Data for Timesweeper" below) in chunks, converts allele data to time-series allele velocity data, and predicts using the 1DCNN trained on simulated data. Each prediction represents a 51-SNP window with the focal allele being the actual target.

Timesweeper outputs predictions as both a csv file and a bedfile. The BED file allows for easy intersections using bedtools and can be cross-referenced back to the CSV for score filtering.

Here are the details on the headers:
- Chrom: Chromosome/contig, identical to VCF file name of it
- BP: location of central allele in the window being predicted on
- Class: Neut/Hard/Soft, class identified by the maximum score in 3-class softmax output from the model
- Neut/Hard/Soft Prob: raw score from softmax final layer of 1DCNN
- SDN/SSV_sval: Predicted selection coefficient by each (ssv, sdn) regression model
- Win_Start/End: left and right-most locations of the SNPs on each side of the window being predicted on

Note: By default Timesweeper only outputs sites with a minimum sweep (sdn+soft) score of 0.66 to prevent massive amounts of neutral outputs. This value could easily be modified in the module but we find it better to filter after the fact for more flexibility.

Timesweeper also has a `--benchmark` flag that will allow for testing accuracy on simulated data if wanted. This will search the input data for the mutation type identifier flags allowing a benchmark of detection accuracy on data that has a ground truth.

```
$ timesweeper detect -h
usage: timesweeper detect [-h] -i INPUT_VCF [--benchmark] --aft-class-model
                          AFT_CLASS_MODEL --aft-reg-model AFT_REG_MODEL
                          [--hft-class-model HFT_CLASS_MODEL] --hft-reg-model
                          HFT_REG_MODEL -o OUTFILE -y YAML_CONFIG -s SCALAR

Module for iterating across windows in a time-series vcf file and predicting
whether a sweep is present at each snp-centralized window.

options:
  -h, --help            show this help message and exit
  -i INPUT_VCF, --input-vcf INPUT_VCF
                        Merged VCF to scan for sweeps. Must be merged VCF
                        where files are merged in order from earliest to
                        latest sampling time, -0 flag must be used.
  --benchmark           If testing on simulated data and would like to report
                        the mutation type stored by SLiM during
                        outputVCFSample as well as neutral predictions, use
                        this flag. Otherwise the mutation type will not be
                        looked for in the VCF entry nor reported with results.
  --aft-class-model AFT_CLASS_MODEL
                        Path to Keras2-style saved model to load for
                        classification aft prediction.
  --aft-reg-model AFT_REG_MODEL
                        Path to Keras2-style saved model to load for
                        regression aft prediction. Either SDN or SSV work,
                        both will be loaded.
  --hft-class-model HFT_CLASS_MODEL
                        Path to Keras2-style saved model to load for
                        classification hft prediction.
  --hft-reg-model HFT_REG_MODEL
                        Path to Keras2-style saved model to load for
                        regression hft prediction.
  -o OUTFILE, --outfile-prefix OUTFILE
                        Prefix to use for results csvs. Will have
                        '_[aft/hft].csv' appended to it.
  -y YAML_CONFIG, --yaml YAML_CONFIG
                        YAML config file with all required options defined.
  -s SCALAR, --scalar SCALAR
                        Minmax scalar saved as a pickle file during training.

```

---

## Preparing Input Data for Timesweeper to Predict On

Timesweeper needs a specific format of input data to work, namely a VCF file merged to the superset of all samples and polymorphisms that is merged *in order* of oldest to most recent. That last point is extremely important and easily missed.

VCFs of all samples will need to be merged using the `bcftools merge -Oz --force-samples -0 <inputs_earliest.vcf ... inputs_latest.vcf> > merged.vcf.gz` options.

---

## Example Usage

Below is an example run of Timesweeper using the example config and simulation script provided in the repo. This will go through the entire process of simulating, training the model, processing input and calling predicted sweeps on a VCF of interest called `foo.vcf`.

```
conda activate blinx
cd timesweeper

#Simulate training data
timesweeper sim_custom yaml examples/example_config.yaml

#Create feature vectors for both AFT and HFT
timesweeper condense --hft yaml examples/example_config.yaml

#Train network
timesweeper train -n example_ts_run yaml examples/example_config.yaml

#Predict on input VCF
timesweeper detect -i foo.vcf \
  --aft-class-model ts_experiment/trained_models/example_ts_run_Timesweeper_aft \
  --hft-class-model ts_experiment/trained_models/example_ts_run_Timesweeper_hft \
  --aft-reg-model ts_experiment/trained_models/REG_example_ts_run_Timesweeper_aft_sdn \
  --hft-reg-model ts_experiment/trained_models/REG_example_ts_run_Timesweeper_hft_sdn \
  
```


## Using Non-Timesweeper Simulations
