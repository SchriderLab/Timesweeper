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

## Generating and preparing data

1. Run slim on the slim parameterizations and slim script defined in the inititializeVars method in Blinx. This will submit SLURM jobs to run a bunch of simulation replicates.

   ```{bash}
   $ python timesweeper/blinx.py -f launch_sims -s slimfiles/onePop-adaptiveIntrogression.slim #Any slimfile works
   ```

   Then wait for a bit...

<br>

2. Once simulations are done, separate out each MS entry into its own file (timepoint) within folders of replicates (samples). This will also clean up any non-relevant SLiM output so that the only thing in each file is the SHIC-required header and the MS entry.

   ```{bash}
   $ python timesweeper/blinx.py -f clean_sims -s slimfiles/onePop-adaptiveIntrogression.slim
   ```

Then wait again, but a little less long this time...

<br>

3. Now create feature vectors using the diploSHIC fvecSim module.

   ```{bash}
   $ python timesweeper/blinx.py -f create_feat_vecs -s slimfiles/onePop-adaptiveIntrogression.slim
   ```

---

Now for the science-y stuff.

## Simulated scenarios and sampling

We simulated two types of adaptation: selective sweeps on mutations that arose in the focal population, and those that originated in a donor population and introduced into the focal population via introgression (i.e. migration/hybridization).

In addition, we simulated these scenarios with several sampling schemes. First, each we constructed both one-population and two-population samples for each scenario (see the onePop/ and twoPop/ directories). Thus, we have one-population samples with selective sweeps (onePop/selectiveSweep), one-population samples with adaptive introgression (i.e. the donor population is not sampled onePop/adaptiveIntrogression), two-population samples with selective sweeps (one population is under selection and the other is not; twoPop/selectiveSweep), and two-population samples with adaptive introgression (sampling both the donor and recipient population; twoPop/adaptiveIntrogression).

Second, for each scenario we constructed both time-series samples (sampling 20 individuals across each of 10 timepoints), and single-timepoint samples (i.e. 200 individuals all sampled from the same time). This allows us to see how much power we gain by using time-series data in comparison to the same amount of data all gathered from the same timepoint. The scripts for simulating the data and merging the resultant files are 01_launchSims.py and 01b_combineSims.py respectively (found in all four of the directories mentioned in the previous paragraph).

