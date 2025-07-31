

import numpy as np
import scipy as sp
from paradigm_setting_PMZP import paradigm_setting
from simulate_adaptation import simulate_adaptation
from repeffects_fig4_ZP import produce_slopes
import sys
from ExperimentalData import create_pattern
from itertools import product
np.set_printoptions(threshold=sys.maxsize)
from joblib import Parallel, delayed
import os
n_jobs = 2

def simulate_subject(v, X, j, cond1, cond2, a, b, sigma, k, model_type, reset_after, paradigm, N, ind, sub_num, voxel_correlated_noise, independent_noise):
    """Produces the voxel pattern for one simulation for one parameter combination of one paradigm"""
    out = simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, k, model_type, reset_after, paradigm, N, sub_num)
    
    if paradigm == 'face':
        out = np.tile(out, (49, 1))

        if voxel_correlated_noise > 0:
            cov_voxels = (1 - voxel_correlated_noise) * np.eye(v) + voxel_correlated_noise * np.ones((v, v))
            mv_noise = np.random.multivariate_normal(mean=np.zeros(v), cov=cov_voxels, size=(196, sub_num))
            mv_noise = mv_noise.transpose(2, 0, 1)
        else:
            mv_noise = 0
        noise = np.random.randn(v, 196, sub_num) * independent_noise + mv_noise

        noisy_pattern = (out.T + noise).T
        pattern_split = np.stack([
            noisy_pattern[:, ind['cond1_p1'], :],
            noisy_pattern[:, ind['cond1_p2'], :],
            noisy_pattern[:, ind['cond2_p1'], :],
            noisy_pattern[:, ind['cond2_p2'], :]
        ], axis=1)

        reshaped = pattern_split.reshape(sub_num, 196, v)  # shape: (sub_num, 32, v)

        return reshaped

    elif paradigm == 'grating':
        out = np.tile(out, (4, 1))
        
        if voxel_correlated_noise > 0:
            cov_voxels = (1 - voxel_correlated_noise) * np.eye(v) + voxel_correlated_noise * np.ones((v, v))
            mv_noise = np.random.multivariate_normal(mean=np.zeros(v), cov=cov_voxels, size=(48, sub_num))
            mv_noise = mv_noise.transpose(2, 0, 1)
        else:
            mv_noise = 0

        noise = np.random.randn(v, 48, sub_num) * independent_noise + mv_noise

        noisy_pattern = (out.T + noise).T
        pattern_split = np.stack([
            noisy_pattern[:, ind['cond1_p1'], :],
            noisy_pattern[:, ind['cond1_p3'], :],
            noisy_pattern[:, ind['cond2_p1'], :],
            noisy_pattern[:, ind['cond2_p3'], :]
        ], axis=1)
        reshaped = pattern_split.reshape(sub_num, 32, v)
        return reshaped

def produce_slopes_one_simulation(paradigm, model_type, sigma, a, b, k, voxel_correlated_noise, n_jobs, independent_noise):
    """Produces the slope of each data feature for one parameter combination for one simulation"""
    # print("one simulation done")
    v, X = 200, np.pi
    cond1, cond2 = X/4, 3*X/4
    sub_num = 1
    # noise = 0.03
    N = 8
    if paradigm == 'face':
        Kfold = 10
    elif paradigm == 'grating':
        Kfold = 8

    j, ind, reset_after, _ = paradigm_setting(paradigm, cond1, cond2)
    y = simulate_subject(v, X, j, cond1, cond2, a, b, sigma, k, model_type, reset_after, paradigm, N, ind, sub_num, voxel_correlated_noise, independent_noise)
    return produce_slopes(y, 1, Kfold)
    
def produce_confidence_intervals(paradigm, model_type, sigma, a, b, k, n_jobs, n_simulations, voxel_correlated_noise, independent_noise):
    """Produces a dictionary of whether a data feature increases, decreases, or does not change significantly for the average of n_simulations simulations
    for one parameter combinations"""
    print("done one parameter combination for one model")
    primary_data_features = []
    CPPM = []
    CPZP = []
    ZPMINUSPM = []
    for sim in range(n_simulations):
        # print(sim)
        slopes = produce_slopes_one_simulation(paradigm, model_type, sigma, a, b, k, voxel_correlated_noise, n_jobs, independent_noise)
        # print(sim)
        # print(slopes)
        primary_data_features.append(slopes[0].tolist())
        CPPM.append([slopes[1]])
        CPZP.append([slopes[2]])
        ZPMINUSPM.append([slopes[3]])
    primary_data_features = np.array(primary_data_features)
    CPPM = np.array(CPPM)
    CPZP = np.array(CPZP)
    ZPMINUSPM = np.array(ZPMINUSPM)
    total = np.hstack((primary_data_features, CPPM, CPZP, ZPMINUSPM))
    # print(total)
    
    
    #Finding overall confidence interval for all simulations
    means, stds = total.mean(axis=0), total.std(axis=0)
    sems = stds / np.sqrt(n_simulations)
    t_critical = sp.stats.t.ppf(0.995, df=n_simulations-1)
    mega_sci = np.column_stack([means - t_critical * sems, means + t_critical * sems])
    result = {}
    for key, ci in zip(['AM', 'WC', 'BC', 'CP', 'AMS', 'AMA', 'CPPM', 'CPZP', 'ZPMINUSCP'], mega_sci):
        if key == 'CPPM' or key == 'CPZP':
            if ci[0] < 0.5 and ci[1] < 0.5:
                result[key] = 3
            elif ci[0] > 0.5 and ci[1] > 0.5:
                result[key] = 1
            elif ci[0] < 0.5 < ci[1]:
                result[key] = 2
            else:
                result[key] = 4
        else:
            if ci[0] < 0 and ci[1] < 0:
                result[key] = 3
            elif ci[0] > 0 and ci[1] > 0:
                result[key] = 1
            elif ci[0] < 0 < ci[1]:
                result[key] = 2
            else:
                result[key] = 4
    # print(result)
    return result
    #3 means CI below zero so decreasing
    #1 means CI above zero so increasing
    #2 means CI overlaps zero so flat


def produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs, voxel_correlated_noise, independent_noise):
    """Evaluate each param combo for one model and return comparison results"""
    param_grid = list(product(*parameters.values()))
    index_map = {
        combo: [parameters['sigma'].index(combo[0]), parameters['a'].index(combo[1]), parameters['b'].index(combo[2]), parameters['k'].index(combo[3])]
        for combo in param_grid
    }

    def evaluate_combo(index, combo):
        sigma, a, b, k = combo
        result_dict = produce_confidence_intervals(paradigm, model, sigma, a, b, k, n_jobs, n_simulations, voxel_correlated_noise, independent_noise)
        #This is the thing that was changed!
        match = [1 if result_dict[k] == experimental_results[k] else 0 for k in result_dict]
        return index, match

    results_comparison = np.zeros((len(param_grid), 9))
    max_same, parameters_of_max = [], []
    no_max_same = 0

    for idx, match in Parallel(n_jobs=n_jobs)(
        delayed(evaluate_combo)(i, combo) for i, combo in enumerate(param_grid)
    ):
        results_comparison[idx] = match
        score = sum(match)
        if score > no_max_same:
            no_max_same = score
            max_same = [match]
            parameters_of_max = [index_map[param_grid[idx]]]
        elif score == no_max_same:
            max_same.append(match)
            parameters_of_max.append(index_map[param_grid[idx]])

    return {
        # 'model_comparison': results_comparison,
        'max_same': np.unique(max_same, axis=0),
        'no_max_same': no_max_same,
        'parameters_of_max': parameters_of_max
    }

def producing_fig_5(parameters, paradigm, n_simulations, n_jobs, voxel_correlated_noise, independent_noise):
    """Outputs the final figure as well as all the possible maximum results"""
    experimental_results = experimental_face_results if paradigm == 'face' else experimental_grating_results
    # fig_5_dict = {model: produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs) for model in range(1, 13)}
    fig_5_dict = {
        model: produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs, voxel_correlated_noise, independent_noise)
        for model in range(3,4)
    }
    fig_5_array = np.zeros((9, 12))
    fig_5_sets = []
    for model, model_data in fig_5_dict.items():
        fig_5_sets.append(model_data['max_same'])
        # for i, max_match in enumerate(model_data['max_same'][0]):
        #     fig_5_array[i, model - 1] = max_match
        fig_5_array[:, model - 1] = model_data['max_same'][0]
    print(fig_5_array, fig_5_sets)
    print(fig_5_dict)
    return fig_5_dict

parameters = {
    'sigma' : [0.08],
    'a' : [0.95],
    'b' : [1.03],
    'k' : [4.38]
}

good_spread_parameters = {
    'sigma' : [0.1, 0.3, 0.5],
    'a' : [0.1, 0.3, 0.5, 0.7, 0.9],
    'b' : [0.3, 0.5, 0.7, 0.9, 1.1, 1.3],
    'k' : [0.4, 4, 10]
}

actual_parameters = {
    #648 parameter combinations
    'sigma' : [0.1, 0.3, 0.5, 0.7, 0.9, 2, 5, 8, 11],
    'a' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'b' : [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
    'k' : [4]
} 


experimental_face_results = {
    'AM' : 3,
    'WC' : 3,
    'BC' : 3,
    'CP' : 3,
    'AMS' : 1,
    'AMA' : 1,
    'CPPM' : 1,
    'CPZP' : 1,
    'ZPMINUSCP' : 1
}

experimental_grating_results = {
    'AM' : 3,
    'WC' : 3,
    'BC' : 3,
    'CP' : 1,
    'AMS' : 3,
    'AMA' : 1,
    'CPPM' : 1,
    'CPZP' : 1,
    'ZPMINUSCP' : 1
}


n_simulations = 50
# voxel_correlated_noise = 0.5

voxel_correlated_noise = [0.4]
independent_noise = [0.15]
results = []

for i in voxel_correlated_noise:
    for j in independent_noise:
        print("amount of voxel correlated noise:")
        print(i)
        print("amount of independent noise:")
        print(j)
        print("new vcor noise")
        results.append([i, j])
        results.append(producing_fig_5(parameters, 'face', n_simulations, n_jobs, i, j))

print(results)