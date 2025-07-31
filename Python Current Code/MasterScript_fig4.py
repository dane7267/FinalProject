import numpy as np
from paradigm_setting import paradigm_setting
from simulate_adaptation import simulate_adaptation
from repeffects_fig4 import produce_slopes
import sys
from ExperimentalData import create_pattern
from itertools import product
np.set_printoptions(threshold=sys.maxsize)
from joblib import Parallel, delayed
import os
import scipy as sp
import matplotlib.pyplot as plt

def simulate_subject(sub, v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind):
    """Produces the voxel pattern for one simulation for one parameter combination of one paradigm"""
    out = simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N)
    #out currently is 32x200 but we want to do this 18 times over. 
    pattern = (out.T + np.random.randn(v, len(j)) * noise).T
    v = pattern.shape[1]
    if paradigm == 'face':
        cond1_p = {1: pattern[::4, :v], 2: pattern[1::4, :v]}
        cond2_p = {1: pattern[2::4, :v], 2: pattern[3::4, :v]}
    elif paradigm == 'grating':
        cond1_p = {1: pattern[ind['cond1_p1'], :v], 2: pattern[ind['cond1_p3'], :v]}
        cond2_p = {1: pattern[ind['cond2_p1'], :v], 2: pattern[ind['cond2_p3'], :v]}
    return np.vstack([cond1_p[1], cond1_p[2], cond2_p[1], cond2_p[2]])

def produce_slopes_one_simulation(paradigm, model_type, sigma, a, b, n_jobs, n_simulations):
    """Produces the slope of each data feature for one parameter combination for one simulation"""
    v, X = 200, np.pi
    cond1, cond2 = X/4, 3*X/4
    sub_num = 18
    noise = 0.03
    N = 8

    j, ind, reset_after, _ = paradigm_setting(paradigm, cond1, cond2)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_subject)(sub, v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind)
        for sub in range(sub_num)
    )
    #results = 18 x 32 x 200
    y = np.array([results[sub] for sub in range(sub_num)])

    return produce_slopes(y, 1)

def produce_confidence_intervals(paradigm, model_type, sigma, a, b, n_jobs, n_simulations):
    """Produces a dictionary of whether a data feature increases, decreases, or does not change significantly for the average of n_simulations simulations
    for one parameter combinations"""
    print("done one simulation set")

    slopes = Parallel(n_jobs=n_jobs)(
        delayed(produce_slopes_one_simulation)(paradigm, model_type, sigma, a, b, n_jobs, n_simulations)
        for _ in range(n_simulations)
    )
    
    #Finding overall confidence interval for all simulations
    slopes = np.array(slopes)
    means, stds = slopes.mean(axis=0), slopes.std(axis=0)
    sems = stds / np.sqrt(n_simulations)
    t_critical = sp.stats.t.ppf(0.995, df=n_simulations-1)
    mega_sci = np.column_stack([means - t_critical * sems, means + t_critical * sems]).flatten()

    results_dict = {
        key: 3 if x[0] < 0 and x[1] < 0 else 
             1 if x[0] > 0 and x[1] > 0 else 
             2 if x[0] <0 < x[1] else 
             4
        for key, x in zip(['AM', 'WC', 'BC', 'CP', 'AMS', 'AMA'], mega_sci.reshape(-1, 2))
        }
    #3 means CI below zero so decreasing
    #1 means CI above zero so increasing
    #2 means CI overlaps zero so flat
    return [
        results_dict['AM'], results_dict['WC'], results_dict['BC'],
        results_dict['CP'], results_dict['AMS'], results_dict['AMA']
    ]

def producing_fig_4(parameters, models, paradigm, n_jobs, n_simulations):
    """Master code for producing figure 4 using a parameter dictionary, dictionary of all models, paradigm, parallel jobs, and a set number of simulations per parameter"""
    fig_4 = np.empty((6, 12), dtype=object)
    param_combinations = list(product(*parameters.values()))
    tasks = [
        (paradigm, model_type, sigma, a, b, n_jobs, n_simulations) 
        for sigma, a, b in param_combinations
        for model_type in models.values()
    ]

    # Run simulations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(produce_confidence_intervals)(*task) for task in tasks
    )
    idx = 0

    for sigma, a, b in param_combinations:
        for model_name, model_type in models.items():
            sim_results = results[idx]
            for row in range(6):
                if fig_4[row][model_type - 1] is None:
                    fig_4[row][model_type - 1] = []
                fig_4[row][model_type - 1].append(sim_results[row])
            idx += 1
    
    for row in range(6):
        for col in range(12):
            unique_elements = set(fig_4[row][col])
            if 4 in unique_elements:
                print("invalid confidence interval at")
                print("row")
                print(row)
                print("column")
                print(col)
            elif unique_elements == {3}:
                fig_4[row][col] = 'blue'
            elif unique_elements == {1}:
                fig_4[row][col] = 'red'
            elif unique_elements == {1, 2}:
                fig_4[row][col] = 'white-red'
            elif unique_elements == {2, 3}:
                fig_4[row][col] = 'white-blue'
            elif unique_elements == {2}:
                fig_4[row][col] = 'white'
            else:
                fig_4[row][col] = 'any'

    print(fig_4)
    
    #Turns figure into colour-coded diagram
    x = np.zeros(fig_4.shape)
    y = np.zeros(fig_4.shape)
    for i in range(0, fig_4.shape[1]):
        for j in range(0, fig_4.shape[0]):
            x[j, i] = i
            y[j, i] = fig_4.shape[1]-j-1
    colours =  {"blue":"blue", "any":"purple", "red":"red", "white-red":"pink", "white-blue":"cyan", "white":"white"}
    fig_4 = fig_4.flatten()
    for i in range(len(fig_4)):
        fig_4[i] = colours[fig_4[i]]
    plt.scatter(x.flatten(), y.flatten(), c=fig_4, s=550)
    plt.axis("off")
    plt.show()

parameters = {
    'sigma' : [0.1, 0.9],
    'a' : [0.6],
    'b' : [1.5]
}

good_spread_parameters = {
    'sigma' : [0.1, 0.9, 11],
    'a' : [0.1, 0.5, 0.9],
    'b' : [0.1, 0.7, 1.5]
}

actual_parameters = {
    #Alink et al. 648 parameter combinations
    'sigma' : [0.1, 0.3, 0.5, 0.7, 0.9, 2, 5, 8, 11],
    'a' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'b' : [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
}

model = {
    'global scaling' : 1
}

models = {
    #Dictionary of all models
    'global scaling' : 1,
    'local scaling' : 2,
    'remote scaling' : 3,
    'global sharpening' : 4,
    'local sharpening' : 5,
    'remote sharpening' : 6,
    'global repulsion' : 7,
    'local repulsion' : 8,
    'remote repulsion' : 9,
    'global attraction' : 10,
    'local attraction' : 11,
    'remote attraction' : 12
}

n_jobs = 4 #Number of parallel jobs
n_simulations = 50 #Number of simulations per parameter combination

producing_fig_4(good_spread_parameters, models, 'face', n_jobs, n_simulations)
#n_simulations is the number of simulations for each parameter combination. 