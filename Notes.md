# Time-Series Sweep Detection using CNNs

---

### This is a general notebook of progress updates, ideas, and planning for the TimeSweeper manuscript.

---

#### Large Experiment Ideas
- Try some alternative methods for feature representation (jointSFS?)
- Training set size vs accuracy, how low can you go?
- General comparison of standard parameters for single/twopop selective sweeps and adaptive introgression
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

#### Planned Figures
- Explanatory figure describing the sampling process and simulation pipeline into sampling
- ROC curves and confusion matrices for all sampling schemes of eac set of parameters
- ROC comparisons for multiple parameters and the same sampling scheme (inverted onto conf mat)
- PR curves

---

#### Updates

##### 7/9/2021
- Finished OO refactor of haplotype frequency spectrum feature prep module.
- Documentation for above module also completed today.
- Will suggested it would be useful to have examples of each function's output, I agree. Have added to low-priority issues.
- Next: