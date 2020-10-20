import os
import sys

import timeseriessweeps.plotting_utils as pu
import timeseriessweeps.utils as ut
from timeseriessweeps.dataprepper import (AliPrepper, HapsPrepper, JSFSPrepper,
                                          SFSPrepper)
from timeseriessweeps.initializeVar import *
from timeseriessweeps.utils import run_batch_job

sys.path.insert(1, '/pine/scr/e/m/emae/timeSeriesSweeps')

# Testing vars #############################################################
#TODO set these as argparse args OR as file naming in slimfiles? JSON?
# Time Series
sampleSizePerStepTS = 20  # individuals in sample for each time interval in time series
numSamplesTS = 2  # number of time points sampled in time series
samplingIntervalTS = 100  # spacing between time points

# 1 Sample at 1 One Time Point
# size of population sampled at one time point so that it is same size as time series data
sampleSizePerStep1Samp = 40
numSamples1Samp = 1  # number of time points sampled
samplingInterval1Samp = 200  # spacing between time points
# End Testing vars #########################################################

#TODO Also these in argparse
baseDir = '/proj/dschridelab/timeSeriesSweeps'
maxSnps = 200


