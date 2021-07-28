# Time-Series Sweep Detection using CNNs

---

### This is a general notebook of progress updates, ideas, and planning for the TimeSweeper manuscript.

---

#### Large Experiment Ideas
- Try some alternative methods for feature representation (jointSFS?)
- Training set size vs accuracy, how low can you go?
- General comparison of standard parameters for single/two pop selective sweeps and adaptive introgression
- How does spacing of sampling affect detection power?
    - Could do left/right skewed distributions and random intervals for all number of samples
    - Downsampling from most dense sampling scheme
- How much does placement of single sampling affect power of detection?
- Bottlenecks and non-equilibrium demos
- How does number of samples at each timepoint affect detection? 
- Comparison to benchmark methods (should we do this across a basic set of parameters or a bunch?)
    - FiT?
    - Adapt Graham’s method and test that
    - Other stuff we can find
- How does adjusting the length of the sweep period affect accuracy?
- Strength of selection (0.01 and 0.005 in addition to 0.05)?
- iHS neural net, sort data by highest score


#### Planned Figures
- Explanatory figure describing the sampling process and simulation pipeline into sampling
- ROC curves and confusion matrices for all sampling schemes of eac set of parameters
- ROC comparisons for multiple parameters and the same sampling scheme (inverted onto conf mat)
- PR curves

---

#### TODO by 7/29/21
- Uniform vs decay all timepoints figures
- Look at distribution of selection coefficients for pop files
- Grid search for model
- Selection strength adj
- Window-after selection adjust
- Sample size within the same number of timepoints


---
#### Updates

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

