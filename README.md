# TimeSweeper

Workflow that generates SLiM simulations for multiple timepoints, parses them into msprime-style output, and trains an R-CNN to recognize selective sweeps.

## Empirical Data
- Data from the empirical experiments in this study come from: https://www.nature.com/articles/s41586-021-03336-2
  - Direct link to data repo: https://www.ebi.ac.uk/ena/browser/view/PRJEB42781?show=reads (Project ID: PRJEB42781)

## Installation
Pretty straightforward install and setup process:

```{bash}
git clone git@github.com:SchriderLab/timeSeriesSweeps.git

cd timeSeriesSweeps
make install
```

This will:
    - Create the conda environment for the TimeSweeper (Blinx) with all necessary packages
    - Install and make SLiM, a forwards-in-time simulation engine
    - Install and make diploSHIC, a pacakge for detecting sweeps from simulation data
  
All of this is done within the conda environment, so no privileges are needed.

And that's it, you're good to go. I'll be making this a bona-fide package sooner or later so we can use setuputils and all the fancy stuff.


## Necessary Input Preprocessing
- VCFs of all samples will need to be merged using the `bcftools merge -Oz --force-samples -0 <inputs.vcf> > merged.vcf.gz` options
  - Note: VCFs supplied to this merge must be in order from earliest to latest in terms of time sampled

## Workflow Overview
1. Create a SLiM script
   1. Either based on the `simple_model.slim` example 
   2. Or by using stdpopsim to generate a SLiM script and injecting time-series sampling into it with `inject_slim.py`
2. Simulate replicates of each sweep type using make_sims.sh
   1. If available, we suggest using a job submission platform such as SLURM to parallelize simulations
3. Preprocess simulated vcfs by merging with `process_vcfs.sh`
4. Create features for AFS and HFS models with `make_training_features.py`
5. Train networks with `nets.py`
6. Run `timesweeper.py` on VCF of interest using trained models




## Simulated scenarios and sampling

We simulated two types of adaptation: selective sweeps on mutations that arose in the focal population, and those that originated in a donor population and introduced into the focal population via introgression (i.e. migration/hybridization).

In addition, we simulated these scenarios with several sampling schemes. First, each we constructed both one-population and two-population samples for each scenario (see the onePop/ and twoPop/ directories). Thus, we have one-population samples with selective sweeps (onePop/selectiveSweep), one-population samples with adaptive introgression (i.e. the donor population is not sampled onePop/adaptiveIntrogression), two-population samples with selective sweeps (one population is under selection and the other is not; twoPop/selectiveSweep), and two-population samples with adaptive introgression (sampling both the donor and recipient population; twoPop/adaptiveIntrogression).

Second, for each scenario we constructed both time-series samples (sampling 20 individuals across each of 10 timepoints), and single-timepoint samples (i.e. 200 individuals all sampled from the same time). This allows us to see how much power we gain by using time-series data in comparison to the same amount of data all gathered from the same timepoint. The scripts for simulating the data and merging the resultant files are 01_launchSims.py and 01b_combineSims.py respectively (found in all four of the directories mentioned in the previous paragraph).

