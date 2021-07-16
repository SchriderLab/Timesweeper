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

#### Planned Figures
- Explanatory figure describing the sampling process and simulation pipeline into sampling
- ROC curves and confusion matrices for all sampling schemes of eac set of parameters
- ROC comparisons for multiple parameters and the same sampling scheme (inverted onto conf mat)
- PR curves

---

#### TODO by 7/22/21
- ~~Fix SLiM Scripts and make sure sampling sizes for runner script passing is correct~~
- Replicate ROC curves
- Uniform vs Decay sampling patterns

---
#### Updates

##### 7/16/2021
- Trying to work out how these population sizes need to be laid out in the SLiM scripts.
  - Right now each subpop is 500 diploid individuals.
  - This means for each type of sweep:
    - Onepop-selsweep: 500 ind (1000 chroms) throughout entire process, all sampled.
    - Twopop-selsweep: 2 x 500 ind (2 x 1000 chroms), 500 chroms sampled from each then concatenated, all sampled from pool. ! THIS IS BEING CHANGED BY DAN TO BE CLOSER TO LOCAL ADAPTATION
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

