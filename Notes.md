# Time-Series Sweep Detection using CNNs

---

### This is a general notebook of progress updates, ideas, and planning for the TimeSweeper manuscript.

---

#### Large Experiment Ideas
- Phased vs unphased

Adapt to microbiome longitudinal studies?

#### Planned Figures
- Explanatory figure describing the sampling process and simulation pipeline into sampling
- ROC curves and confusion matrices for all sampling schemes of eac set of parameters
- ROC comparisons for multiple parameters and the same sampling scheme (inverted onto conf mat)
- PR curves

---

- Demo model for East Asians - read https://www.nature.com/articles/s41586-021-03336-2 (https://www.ebi.ac.uk/ena/browser/view/PRJEB42781?show=reads)

#### TODO
- Need to add argparser to all scripts
- Evo rescue?
- Snakefile/Nextflow for sims/preprocessing and another for training basic model
  
---
#### Updates

#### 2/16/2022
- Bugfix for bash globbing the vcf merges
- Curioius how bad the soft sweep detection is with this model

#### 2/14/2022
- Wrote new module for processing multiVCFs into numerically-sorted vcf files for merging

#### 2/9/2022
- Running training with all the new updated scripts

#### 2/8/2022
- Should provide a simple simulation example script, not just the stdpopsim one

#### 2/7/2022
- More cleanup/doc writing
- Typehints are gone from repo
- Need to think of workflow, will probably write readme first and then match the software to that
  - One thing to think about is having the option to write out all the data to npz after you collect it in the network training module
- Streamlined basically all of the modules
  - Including making a separate module for training nets using the same methods of extraction as the window classifier.
  
#### 2/4/2022
- Today is cleanup day. Items on the docket:
  - Go through each script, remove typestrings if there, add docstrings if not
  - Write unit tests, or at least get test module started
  - Fix bugs with haps and sampling
  - Clarify and streamline sampling
  - Rewrite network code to allow for vcf input

#### 1/31/2022
- TODO Today
  - Update plot script to do 3x3 means across methods where 3 lines are for each class prob - fraction of sims that are assigned at each bin to each of the three classes
  - Get mean plots for all sims that ran over the weekend
  - Organize workflow for training and running
  - Get SLIM injection script updated to reflect new sim updates

#### 1/27/2022
- Mean probs in 3x3

#### 1/25/2022
- Finally got windows classification figure done
- Making violin plot and mean line plot
- Need to:
  - Download BAMS 
  - Edit stdpopsim injector to do vcf output
  - Rerun analyses for all other setups using current setup
  - Think of a way to condense data after sim and preprocessing (npz? hdf5?)

#### 1/20/2022
- Catch up on documentation
- Need to make snakefile for the paper stuff, convoluted to take out stuff for making figures as it stands
- Also need to fix process for collecting sweeps and plotting 
- Check over and fix bam downloads
- Will have to rewrite a lot of the stdpopsim -> analysis pipeline to accept vcfs

#### 1/13/2022
- Assign major/minor allele based on which allele is most prevalent in last timepoint (major allele at final timepoint)

#### 1/12/2022
- GenotypeCounts only works for diploid, need to adjust for haplotypearrays for MAF calcs

#### 1/10/2022
- Why are some samples failing in split_arr?
- Running test to make average image 
- Checkk where ploidy is assumed in afs calc

#### 1/6/2022
- Need to check vcf sims and performance
- Streamline model training
- Throw out SNPs that don't have enough neighbors on either side (first 25)

#### 12/10/2021
- Merging vcf files gets us what we want for simulated chroms
  - Have to feed them to merge in order of time correctly
  - Must use "-0" flag with merge so it doesn't count them as missing
  - Solves the haplotype problem, everything should be same size now
  
#### 12/9/2021
- Need to match all possible snps for haps
  - Can just assume 0 for not present, right?
- Should use joint genotyper, more sensitive genotyping
  - Will result in single VCF
  
#### 12/7/2021
- Finished first draft of window classifier for AFS
- Needs to be sped up, only 40 iters/sec
  - Probably end up multi-processing along batches
- Maybe consider swapping to sk-allel moving_window stats
  
#### 12/2/2021
- Some pretty drastic overfitting happening on the haps net

#### 11/30/2021
- Replaced sorting algo for haps 
  
#### 11/16/2021
- Ran sims and pipelines for selection coefficients and windows
  - High selection coeff degredation in alelle freqs

#### 11/15/2021
- Finally fixed haplotypes module, sample sizes were being misrepresented in some of the loops
- Wow haplotypes need a ton more data
- Simulated 50k samples for comparison of training set size
- Need to figure out what "Couldn't process onepop/soft/freqs/909.pop.freqs because of list index out of range" is in freqmat

#### 11/9/2021
- Clean up simple sims to remove logging
- Binary classifier
- Figure out how to remove the first gen sampled
- OVERFITTING

#### 11/8/2021
- Fixed all the sampling issues, FIT now totally works
- Rewrote the snakefile for simple sims
- Rerunning some simple sims to test out allele freq tracking
- Need to look at simple sims and refactor, figure out why extra one is tacked on

#### 11/7/2021
- Why is it still being padded?

#### 11/6/2021
- Check over FIT DF binning again, might not be needed
- Binning is now done pre-sim based on the same logic as before.
  - Years BP are converted to generations, sample size and generation cutoffs are applied during the process
  - After binning, gens are converted back into years for slim injection process
  - This should ensure that there are no events where a given generation is sampled multiple times, while still reporting years for plugging into the slim code
- Think this means that Q should realistically only be 1 because otherwise you'll squash gens

#### 11/5/2021
- Change it so that we're binning before we do simulation
- Use the mean value of the distributions to bin, but still pull from random distribution for timing
- Plot a range of confidence scores to see if the more easily visible ones are better predicted
- Saliency plot
- Need to refactor FIT to be less convoluted, especially since part of it will now be integral

#### 11/4/2021
- Swap out HFS for allele matrix where we track each allele frequency at every timepoint - take the FIT code and throw it into the network - PRIORITY

#### 10/29/2021
- FIt binning is causing the improper freq calcs
  
#### 10/20/2021
- Fixed FIt module for binning
  
#### 10/20/2021
- ~~sanitizer function for restarts~~
- FIt on our sims

#### 10/14/2021
- Finished prelim testing of OoA model, ~65% accuracy on TS, SP is a nightmare 
- ~~Plot individual examples of HFS from each class~~
- ~~implement binning function to make sampling points more than 1 indiv per hfs~~
- DO FIt ON THE EMPIRICAL SIMS

#### 10/7/2021
- Testing haps creation with updated model
  - Fixed bug with max size of samples 
  
#### 10/6/2021
- Need to figure out how to handle hfs creation when different number of samples per timepoint
  - Could sample to max number in simulations, but what about the actual data?

#### 10/1/2021
- Troubleshooting inject_slim, basically done just need to run the sims

#### 9/28/2021
- Master ID for each sample is listed as Sample Alias in SRA
- Need to filter out low coverage samples and potentially combine those which have multiple library preps
- Keep all "Yes" or "Contamination" in column K of Table 1
  - Make histogram of combined coverages that aren't related and pass coverage filter
- Might have to adjust binning post-generation
  - Will also have to think about what we need to do to make FIt compatible

- Need to figure out how burn time affects all other stuff happening, scheduling is wack

#### 9/28/2021
- Too low coverage for us to get reliable genotypes, will need to find different data structure
  - Reads supporting ancestral/derived alleles? How to sort?
    - By total number of reads supporting derived?
    - Fraction of reads supporting derived -> sum
      - Gets total expected number of derived alleles on that haplotype
- Need to simulate read counts on data
  - Depth distribution at each snp from simulated individuals for each real individual
  - Hets need to add binomial sampling to get counts for both alleles

#### 9/28/2021
- Been working on getting SLiM output from stdpopsim to make sense and create script to inject code

#### 9/21/2021
- Adjustable SLiM params for timing of sweep and other code we need to add
- Run timing variety and train model on all of them
- Compare test results from each on whole training
  
#### 9/16/2021
- Finish MSMC tests to confirm failure
- Use stdpopsim to gen up OoA admix 3 pop demos
- Fit a guassian to confidence intervals on carbon dating for replicates
- Go ahead and use all samples
- Fit timing of sampling from demo sims to collection times of mongolian samples

#### 9/15/2021
- Launched all sims for sel coefficient experiment
- Launched jobs to finish empirical study testing from msmc

#### 9/14/2021
- Attempted to move simulations into snakemake, gave up because it was too much of a pain. Will probably try again some other time but for now it's gonna have to stay like this so we can get results.

#### 9/13/2021
- Tried to test out empirical analysis snakemake files but slurm wasn't playing nice
- Started migrating simulation launches into snakemake but need to double check everything
- Need to read dl2ai tomorrow

##### 8/19/2021
- Met with Dan
  - Need to test JPT/CHB (bc OoA) 1KG sample with workflow to confirm MSMC is working as intended
  - More recent than 20kya not very good, so make that the cutoff
  - Graph of the one Mongolian genome we tested looks shitty, see `mongolian_model/msmc_results.png`
  - Calculate pi per site before masks are applied to get rough estimate of theta

##### 8/13/2021
- Finally finished Snakefile for the ancient genome samples
- Can now go from empirical data to SLiM script for single-pop using MSMC
- Used the pipeline to gen files needed for single Mongolian sample
- Need to mess around with variant filtering to see how it affects inferred mutation and recomb rate
  

##### 8/12/2021
- Trying to wrangle this data from the Wang paper
  - Looks like it'll need to go:
    - EIGENSTRAT -> PED (EIGENSOFT CONVERTF)
    - PED -> VCF (PLINK)
    - VCF -> MSMC
  - Should check and see if any info is being lost/messed up during this process

##### 8/3/2021
- Fixed `haplotypes.py` bug where new simulation format (batches inside pops dirs) wasn't being read properly
  - Will need to remember that if I need to do any more stuff with old sim folders, or just regen
- Ran everything needed for sample size diffs and selection coefficient experiments
- Genned up a bunch of results figures and sent over to Dan. Should discuss next steps.

##### 7/27/2021
- For some reason during some sampling schema haplotypes module isnt sampling the correct number of haps nor timepoints. Debugging.
  - Found the bug, the samplesSkipped weren't being subtracted from the samplesSeen during the check for generations matching.
  - Rerunning entire set of experiments because that means most of the HFS are messed up from badly-sampled sims.
- Adjust hfs npz creation so that it's a single job per simtype per schema, more efficient job submission
- Shapes are now consistently documented throughout entire process.
  - Shape is consistently (samples, timepoints, haps) which is the same shape necessary for Conv1D inputs for Keras. 
  - This only changes during the final plotting step, where the timepoints and haps are transposed for the vertical plot.

##### 7/26/2021
- Troubleshooting pipeline all day, weird that it's having so many issues now.

##### 7/23/2021
- Should put it somewhere in the docs that the shape of the final haps featvecs has to be (num_tp, samp_size*num_tp [200]).
  - Should we force 200? Seems clunky, maybe just multiply it in the Snakemake pipeline and pass it in when needed.
- Realized we could run multiple Snakemake workflows in parallel if we separate out the configfiles for the run.
  - No big deal to generate these, could even do it programmatically if really wanted to do a ton.
  - For now just going to hand-write the ones we're interested in and throw them in a separate folder.
  - Will have to use the --configfile flag for the Snakemake call.
- !RERUN THE 20SAMP SETUPS, THINK THE ONE-HOT ENCODING WAS MESSED UP

##### 7/22/2021
- Spent all day troubleshooting various parts of the workflow after discovering haps weren't being properly generated.
- Eventually fixed hap part, but need to fix single-point data in the network and plotting scripts.
- Might need to revise how we're shaping arrays to make sure it's consistent across the board.

##### 7/21/2021
- Finished up running Snakemake workflow for uniform and dense modern sampling. 
- Not looking good, results show that scores are way worse than they used to be.
  - Need to test and confirm that this is because of the new way of calculating HFS featvecs rather than some bug in the workflow.
- Re-implemented HFS tracking at last timepoint rather than entire thing, rerunning workflow.

##### 7/20/2021
- Snakemake now handles model running as well as feature creation and aggregation.
- Hap_models module updated to run both time-series as well as single-point models every time it's run.
  - This allows us to use the exact same samples for each in the train/test split, as all we're doing is using the last timepoint from the ts hfs.
- Still haven't integrated the simulation rule into Snakemake yet, spent too long today debugging.
- TODO: Generate confmats and ROCs of the sweep vs non-sweep just like before

##### 7/19/2021
- Finalized Snakemake pipeline and tested, working great as of now.
- Going to add in simulation and model training steps to it after I get results for Dan this week.
- Modified the hap_networks module to train both the time-series and single-point models in the same process.
  - This achieves two things, 1) makes it easier to integrate into a snakemake workflow and 2) allows us to use exactly the same samples between TS and SP training paradigms by sampling from the same train/val/test splits.


##### 7/19/2021
- Spent all day working on Snakefile and getting that up and running for haplotype modules.
  - Seems to be working, spent quite a while getting the glob_wildcards function to work properly. Should remember how that works for the next time so I don't waste a ton of time on it again.
- Will need to check out the Profiles to the Snakemake cluster modules rather than the cluster config YAMLs since it's deprecated. It's what I went with first since it's easier but I'd rather it be up to date.
- Will also need to implement the simulations into the Snakefile, should be relatively straightforward now that I understand how it works. One button pipeline is gonna be great!
  
##### 7/16/2021
- Trying to work out how these population sizes need to be laid out in the SLiM scripts.
  - Right now each subpop is 500 diploid individuals.
  - This means for each type of sweep:
    - Onepop-selsweep: 500 ind (1000 chroms) throughout entire process, all sampled.
    - Twopop-selsweep: 2 x 500 ind (2 x 1000 chroms), 500 chroms sampled from each then concatenated, all sampled from pool. ! THIS IS BEING CHANGED BY DAN TO BE CLOSER TO LOCAL ADAPTATION - DONE
    - Onepop-adaptiveIntrogression: 1000 ind split into two 500 ind pops, all (1000 chroms) of *introgressed pop* sampled.
    - Twopop-adaptiveIntrogression: 1000 ind split into two 500 ind pops, concat half-size samples together (2 x 500 chroms) and output.
  - Updated all scripts to now use subpop sizes for sampling along with proper starting population sizes to be consistent across resulting samples.
  - ***THIS MEANS THAT THE ONLY TIME WE'RE RANDOMLY SAMPLING BEFORE HAPLOTYPES IS ADAPTIVE INTROGRESSION WHEN CONCATENATING THE TWO POPS BACK TOGETHER***
  
##### 7/15/2021
- Haplotype module updated to utilize multiprocessing for feature vector generation. Not sure this is the best way to go but it seems to work decently for our setup.
  - ~4 hours with 16 cores processing 50,000 files
- Started 20samp_10int test run, still processing as of writing.
  - Will train a model on this to make sure things are as we expect after eyeball tests.
- Once confirmed working will start experiments for manuscript as outlined above.
  
##### 7/14/2021
- Tested and fixed haplotype length bug by calculating frequency for entire population before building haplotypes.
- Optimized the frequency calculation process (which is now very slow because it has to do it for all genomes) by converting to flat Numpy array and summing over instances.
- Implemented multiprocessing to better optimize speed during haplotype creation.
- Going to need to figure out some way around memory requirement. Might need to write to temp files and then concat at the end of the process.

##### 7/13/2021
- Tested haplotypes module, edits to fix small bugs.
- Currently trying to figure out a solution for the case where segsites are consistent across entire set of timepoints.
  - Not accounting for this leads to errors when trying to compute seqDist across two haplotypes during the frequency sorting phase.

##### 7/12/2021
- Refactored SLiM runner script and blinx launcher function to only simulate a single set of sims for an entire experiment.
- Implemented flexible sampling in haplotypes module, all sampling and adjustments are now done post-sim.
  - This does not include the total number of timepoints being taken (the most possible, 40 in our default) nor does it include the largest number of chromosomes being sampled for output.
  - These are both controlled within blinx.py in the main() function.
- Looks like SLiM is stopping a run before it actually finishes, or the sampling schema needs to be checked out. Either way, it's not actually outputting samples, just the logs. Will check tomorrow.

##### 7/9/2021
- Finished OO refactor of haplotype frequency spectrum feature prep module.
- Documentation for above module also completed today.
- Will suggested it would be useful to have examples of each function's output, I agree. Have added to low-priority issues.

