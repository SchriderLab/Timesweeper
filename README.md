# TimeSweeper

Workflow that generates SLiM simulations for multiple timepoints, parses them into msprime-style output, and trains an R-CNN to recognize selective sweeps.

## Installation

Pretty straightforward install and setup process:

```{bash}
git clone git@github.com:SchriderLab/timeSeriesSweeps.git
git checkout shic_input #Only until we PR and merge

cd timeSeriesSweeps #Really want to rename this TimeSweeper
make install
```

This will:
    - Create the conda environment for the TimeSweeper (Blinx) with all necessary packages
    - Install and make SLiM, a backwards-in-time simulation engine
    - Install and make diploSHIC, a pacakge for detecting sweeps from simulation data
  
All of this is done within the conda environment, so no privelages are needed.

And that's it, you're good to go. I'll be making this a bona-fide package sooner or later so we can use setuputils and all the fancy stuff.

## Running

Pretty simple to run the different modules too. I'd streamline this if I could but since large parts of it require SLURM parallelization it's separate procedures.

```{bash}
make sims #Runs slim on the slimfile I've set in the code, make this an argument?
make combine #Combines simulations done in batches
make format #Formats sims to msprime output
make plot #Makes plots, duh
make train #Runs the old training scripts, will be updated
```

---

Now for the science-y stuff.

## Simulated scenarios and sampling

We simulated two types of adaptation: selective sweeps on mutations that arose in the focal population, and those that originated in a donor population and introduced into the focal population via introgression (i.e. migration/hybridization).

In addition, we simulated these scenarios with several sampling schemes. First, each we constructed both one-population and two-population samples for each scenario (see the onePop/ and twoPop/ directories). Thus, we have one-population samples with selective sweeps (onePop/selectiveSweep), one-population samples with adaptive introgression (i.e. the donor population is not sampled onePop/adaptiveIntrogression), two-population samples with selective sweeps (one population is under selection and the other is not; twoPop/selectiveSweep), and two-population samples with adaptive introgression (sampling both the donor and recipient population; twoPop/adaptiveIntrogression).

Second, for each scenario we constructed both time-series samples (sampling 20 individuals across each of 10 timepoints), and single-timepoint samples (i.e. 200 individuals all sampled from the same time). This allows us to see how much power we gain by using time-series data in comparison to the same amount of data all gathered from the same timepoint. The scripts for simulating the data and merging the resultant files are 01_launchSims.py and 01b_combineSims.py respectively (found in all four of the directories mentioned in the previous paragraph).

## Summarizing patterns of diversity within sampled regions

Once we have our data in hand we need to turn it into a tensor to input to keras. I have begun experimenting with this in there different ways: 1) using the site frequency spectrum (SFS) at each time point, 2) tracking the frequencies of each haplotype over time, and 3) including the entire sequence alignment at each time point as our input. The third is definitely the most ambitious. If I recall, I have the first two working or close to it.

For each of these methods, we have fixed the number of polymorphisms (i.e. the number of sites that are variable within the population, also called segregating sites) to something like 200 I think. The number of polymorphisms differs from simulation run to simulation run because mutations are stochastic, so we simply take the centermost 200 of these to get a fixed-size sequence alignment. Input data in these representations are created by running the respective 02_formatAll.py scripts.

## Training and testing neural networks.

So far I am experimenting with fairly simple convolutional neural networks (and even simpler fully connected neural networks for our single-time data which is unidimensional for the SFS and haplotype frequency data). There is probably a great deal of room for experimening with different/better neural network architectures to improve our performance. To train the neural networks and test them on an independent test set, run the 03_trainCNNs.py scripts which currently use Longleaf's CPU nodes because our data size and neural network architectures are small enough that there is no need for GPUs, but this may change.

---

## Misc

This repo assumes that your environment's PYTHONPATH variable points to a directory that includes runCmdAsJob.py (which you will have to modify if you are using an HPC scheduler other than SLURM or are running things locally). So, after cloning the repo you may want to add its base directory to your PYTHONPATH