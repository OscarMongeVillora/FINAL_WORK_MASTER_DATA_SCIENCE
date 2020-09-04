# TIMES SERIES SPLIT (METHOD)
#n_sample_subset= 28
#percentage_validation = 0.3
#overlap = 0.2

# This module allows to divide the train part of the data in different subsets of train and validation
# in a cross-validation way, respecting the temporal order of the events.
# There are 2 methods to choose: Times Series Split or Blocking Time Series Split

import numpy as np
def times_split(opt_train_size, n_sample_subset = 28, percentage_validation = 0.3, overlap = 0.2, timesplitmethod = 0,
                iteration= 0):

    n_subsets = int(round(opt_train_size / n_sample_subset))            #Number of subsets for cross-validating
    n_validation = int(round(n_sample_subset * percentage_validation))  #Number of subsets of validation
    n_subtrain = n_sample_subset - n_validation                         #Number of subsets of subtrain
    idx = np.array(range(opt_train_size))                               #Inizialize the vector of index samples chosen
    i = iteration

    if timesplitmethod == 0:
        # Times Series Split

        start = 0
        split = start + (i + 1) * n_subtrain
        stop = split + (i + 1) * n_validation

        idx_subset_train = idx[start: split]
        idx_subset_valid = idx[split: stop]

        if i == (n_subsets - 1):
            idx_subset_valid = idx[split:]

    elif timesplitmethod == 1:
        # Blocking Time Series Split

        start = int(round(i * n_sample_subset * (1-overlap)))
        split = start + n_subtrain
        stop = split + n_validation

        idx_subset_train = idx[start: split]
        idx_subset_valid = idx[split: stop]

        if i == (n_subsets - 1):
            idx_subset_valid = idx[split:]

    return [idx_subset_train, idx_subset_valid]


#It is a testing function to show the indexes chosen and demonstrate is working well
def testing_timesplit(opt_train_size_ = 50, n_sample_subset_ = 14, percentage_validation_ = 0.3, overlap_ = 0.2, timesplitmethod_ = 1):

    n_subsets = int(round(opt_train_size_ / n_sample_subset_))
    for i_ in range(0, n_subsets):
        idx_subset_train, idx_subset_validation = times_split(opt_train_size_, n_sample_subset_, percentage_validation_,
                                                            overlap_, timesplitmethod_, iteration= i_)
        print((idx_subset_train, idx_subset_validation))
