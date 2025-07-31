import torch
from paradigm_setting import paradigm_setting
from simulate_adaptation import simulate_adaptation
from repeffects_fig4_ZP_empirical import produce_slopes
from ExperimentalData import create_pattern
import numpy as np
import matplotlib.pyplot as plt


faceData = 'face_data.mat' #We need to change the data
gratingData = 'grating_data.mat' #We need to change the data

def process_empirical_subject(sub, paradigm):
    print("processing empirical subject")
    cond1 = np.pi/4
    cond2 = np.pi*3/4
    j, ind, reset_after, winning_params = paradigm_setting(paradigm, cond1, cond2)
    full_pattern = create_pattern(paradigm)
    pattern = np.array(full_pattern[sub])
    v = pattern.shape[1]
    if paradigm == 'face':
    
        cond1_p = {
            1: pattern[::4, :v],
            2: pattern[1::4, :v]
        }
        cond2_p = {
            1: pattern[2::4, :v],
            2: pattern[3::4, :v]
        }
        y_sub = np.vstack([cond1_p[1], cond1_p[2], cond2_p[1], cond2_p[2]])
    elif paradigm == 'grating':
        cond1_p = {
            1: pattern[ind['cond1_p1'], :v],  # Using the indices for condition 1, part 1
            2: pattern[ind['cond1_p3'], :v]  # Using the indices for condition 1, part 2
        }
        cond2_p = {
            1: pattern[ind['cond2_p1'], :v],  # Using the indices for condition 2, part 1
            2: pattern[ind['cond2_p3'], :v]  # Using the indices for condition 2, part 2
        }
        y_sub = np.vstack([cond1_p[1], cond1_p[2], cond2_p[1], cond2_p[2]])
    return y_sub

def produce_slopes_empirical(paradigm, sub_num):
    y = {}
    for sub in range(sub_num):
        y[sub] = process_empirical_subject(sub, paradigm)
    # print(y)
    if paradigm == 'face':
        return produce_slopes(y, 1, 10)
    elif paradigm == 'grating':
        return produce_slopes(y, 1, 8)

def empirical_data(paradigm):
    sub_num = 18
    results = produce_slopes_empirical(paradigm, sub_num)
    return results

results = empirical_data('face')
print(results)