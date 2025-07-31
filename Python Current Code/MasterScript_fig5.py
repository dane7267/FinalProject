import numpy as np
import scipy as sp
from paradigm_setting import paradigm_setting
from simulate_adaptation import simulate_adaptation
from repeffects_fig4 import produce_slopes
import sys
from ExperimentalData import create_pattern
from itertools import product
np.set_printoptions(threshold=sys.maxsize)
from joblib import Parallel, delayed
import os
n_jobs = 3

def simulate_subject(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num):
    """Produces the voxel pattern for one simulation for one parameter combination of one paradigm"""
    batch_size=3
    T = len(j)
    noisy_pattern = np.empty((sub_num, T, v))
    for i in range(0, sub_num, batch_size):
        current_batch_size = min(batch_size, sub_num - i)
        out = simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, current_batch_size)
        noisy_pattern[i:i+current_batch_size] = (
            out.transpose(0, 2, 1) + np.random.randn(current_batch_size, v, len(j)) * noise
        ).transpose(0, 2, 1)  # shape: (batch, time, voxel)
    if paradigm == 'face':
        # Build condition indices using np.arange
        cond1_p1 = np.arange(0, 32, 4)
        cond1_p2 = np.arange(1, 32, 4)
        cond2_p1 = np.arange(2, 32, 4)
        cond2_p2 = np.arange(3, 32, 4)

        # Stack condition slices
        pattern_split = np.stack([
            noisy_pattern[:, cond1_p1, :],
            noisy_pattern[:, cond1_p2, :],
            noisy_pattern[:, cond2_p1, :],
            noisy_pattern[:, cond2_p2, :]
        ], axis=1)  # shape: (sub_num, 4, 8, v)

        reshaped = pattern_split.reshape(sub_num, 32, v)  # shape: (sub_num, 32, v)

        return reshaped
    elif paradigm == 'grating':
        pattern_split = np.stack([
            noisy_pattern[:, ind['cond1_p1'], :],
            noisy_pattern[:, ind['cond1_p3'], :],
            noisy_pattern[:, ind['cond2_p1'], :],
            noisy_pattern[:, ind['cond2_p3'], :]
        ], axis=1)
        reshaped = pattern_split.reshape(sub_num, 32, v)
        return reshaped

def produce_slopes_one_simulation(paradigm, model_type, sigma, a, b, noise, n_jobs):
    """Produces the slope of each data feature for one parameter combination for one simulation"""
    # print("one simulation done")
    v, X = 200, np.pi
    cond1, cond2 = X/4, 3*X/4
    sub_num = 18
    # noise = 0.03
    N = 8

    j, ind, reset_after, _ = paradigm_setting(paradigm, cond1, cond2)
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(simulate_subject)(sub, v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num)
    #     for sub in range(sub_num)
    # )
    #y is an 18 x 32 x 200
    # y = np.array([results[sub] for sub in range(sub_num)])
    y = simulate_subject(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num)
    return produce_slopes(y, 1)
    
def produce_confidence_intervals(paradigm, model_type, sigma, a, b, n_jobs, n_simulations, noise):
    """Produces a dictionary of whether a data feature increases, decreases, or does not change significantly for the average of n_simulations simulations
    for one parameter combinations"""
    print("done one parameter combination for one model")

    slopes = Parallel(n_jobs=n_jobs)(
        delayed(produce_slopes_one_simulation)(paradigm, model_type, sigma, a, b, noise, n_jobs)
        for _ in range(n_simulations)
    )
    
    #Finding overall confidence interval for all simulations
    slopes = np.array(slopes)
    means, stds = slopes.mean(axis=0), slopes.std(axis=0)
    sems = stds / np.sqrt(n_simulations)
    t_critical = sp.stats.t.ppf(0.995, df=n_simulations-1)
    mega_sci = np.column_stack([means - t_critical * sems, means + t_critical * sems])

    return {
        key: 3 if ci[0] < 0 and ci[1] < 0 else
             1 if ci[0] > 0 and ci[1] > 0 else
             2 if ci[0] < 0 < ci[1] else
             4
        for key, ci in zip(['AM', 'WC', 'BC', 'CP', 'AMS', 'AMA'], mega_sci)
    }
    #3 means CI below zero so decreasing
    #1 means CI above zero so increasing
    #2 means CI overlaps zero so flat

def produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs, noise):
    """Evaluate each param combo for one model and return comparison results"""
    param_grid = list(product(*parameters.values()))
    index_map = {
        combo: [parameters['sigma'].index(combo[0]), parameters['a'].index(combo[1]), parameters['b'].index(combo[2])]
        for combo in param_grid
    }

    def evaluate_combo(index, combo):
        sigma, a, b = combo
        result_dict = produce_confidence_intervals(paradigm, model, sigma, a, b, n_jobs, n_simulations, noise)
        #This is the thing that was changed!
        match = [1 if result_dict[k] == experimental_results[k] else 0 for k in result_dict]
        return index, match

    results_comparison = np.zeros((len(param_grid), 6))
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
        'model_comparison': results_comparison,
        'max_same': np.unique(max_same, axis=0),
        'no_max_same': no_max_same,
        'parameters_of_max': parameters_of_max
    }

def producing_fig_5(parameters, paradigm, n_simulations, n_jobs, noise):
    """Outputs the final figure as well as all the possible maximum results"""
    experimental_results = experimental_face_results if paradigm == 'face' else experimental_grating_results
    # fig_5_dict = {model: produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs) for model in range(1, 13)}
    fig_5_dict = {
        model: produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs, noise)
        for model in range(1, 7)
    }
    fig_5_array = np.zeros((6, 12))
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
    'sigma' : [0.2],
    'a' : [0.7],
    'b' : [0.2]
}

good_spread_parameters = {
    'sigma' : [0.1, 0.2, 0.3, 0.7, 1.0],
    'a' : [0.1, 0.4, 0.7, 0.9],
    'b' : [0.1, 0.5, 0.9, 1.5]
}

actual_parameters = {
    #648 parameter combinations
    'sigma' : [0.1, 0.3, 0.5, 0.7, 0.9, 2, 5, 8, 11],
    'a' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'b' : [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
}    


experimental_face_results = {
    'AM' : 3,
    'WC' : 3,
    'BC' : 3,
    'CP' : 3,
    'AMS' : 1,
    'AMA' : 1
}

experimental_grating_results = {
    'AM' : 3,
    'WC' : 3,
    'BC' : 3,
    'CP' : 1,
    'AMS' : 3,
    'AMA' : 1
}


n_simulations = 50
noise = 0.1

results = producing_fig_5(good_spread_parameters, 'face', n_simulations, n_jobs, noise)

with open("fig5_face_proper_noise_1to7.txt", "w") as f:
    from pprint import pformat
    f.write(pformat(results))




















def simulate_subject(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num):
    """Produces the voxel pattern for one simulation for one parameter combination of one paradigm"""
    batch_size=3
    T = len(j)
    noisy_pattern = np.empty((sub_num, T, v))
    for i in range(0, sub_num, batch_size):
        current_batch_size = min(batch_size, sub_num - i)
        out = simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, current_batch_size)
        noisy_pattern[i:i+current_batch_size] = (
            out.transpose(0, 2, 1) + np.random.randn(current_batch_size, v, len(j)) * noise
        ).transpose(0, 2, 1)  # shape: (batch, time, voxel)
    if paradigm == 'face':
        # Build condition indices using np.arange
        cond1_p1 = np.arange(0, 32, 4)
        cond1_p2 = np.arange(1, 32, 4)
        cond2_p1 = np.arange(2, 32, 4)
        cond2_p2 = np.arange(3, 32, 4)

        # Stack condition slices
        pattern_split = np.stack([
            noisy_pattern[:, cond1_p1, :],
            noisy_pattern[:, cond1_p2, :],
            noisy_pattern[:, cond2_p1, :],
            noisy_pattern[:, cond2_p2, :]
        ], axis=1)  # shape: (sub_num, 4, 8, v)

        reshaped = pattern_split.reshape(sub_num, 32, v)  # shape: (sub_num, 32, v)

        return reshaped
    elif paradigm == 'grating':
        pattern_split = np.stack([
            noisy_pattern[:, ind['cond1_p1'], :],
            noisy_pattern[:, ind['cond1_p3'], :],
            noisy_pattern[:, ind['cond2_p1'], :],
            noisy_pattern[:, ind['cond2_p3'], :]
        ], axis=1)
        reshaped = pattern_split.reshape(sub_num, 32, v)
        return reshaped

def produce_slopes_one_simulation(paradigm, model_type, sigma, a, b, noise, n_jobs):
    """Produces the slope of each data feature for one parameter combination for one simulation"""
    # print("one simulation done")
    v, X = 200, np.pi
    cond1, cond2 = X/4, 3*X/4
    sub_num = 18
    # noise = 0.03
    N = 8

    j, ind, reset_after, _ = paradigm_setting(paradigm, cond1, cond2)
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(simulate_subject)(sub, v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num)
    #     for sub in range(sub_num)
    # )
    #y is an 18 x 32 x 200
    # y = np.array([results[sub] for sub in range(sub_num)])
    y = simulate_subject(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num)
    return produce_slopes(y, 1)
    
def produce_confidence_intervals(paradigm, model_type, sigma, a, b, n_jobs, n_simulations, noise):
    """Produces a dictionary of whether a data feature increases, decreases, or does not change significantly for the average of n_simulations simulations
    for one parameter combinations"""
    print("done one parameter combination for one model")

    slopes = Parallel(n_jobs=n_jobs)(
        delayed(produce_slopes_one_simulation)(paradigm, model_type, sigma, a, b, noise, n_jobs)
        for _ in range(n_simulations)
    )
    
    #Finding overall confidence interval for all simulations
    slopes = np.array(slopes)
    means, stds = slopes.mean(axis=0), slopes.std(axis=0)
    sems = stds / np.sqrt(n_simulations)
    t_critical = sp.stats.t.ppf(0.995, df=n_simulations-1)
    mega_sci = np.column_stack([means - t_critical * sems, means + t_critical * sems])

    return {
        key: 3 if ci[0] < 0 and ci[1] < 0 else
             1 if ci[0] > 0 and ci[1] > 0 else
             2 if ci[0] < 0 < ci[1] else
             4
        for key, ci in zip(['AM', 'WC', 'BC', 'CP', 'AMS', 'AMA'], mega_sci)
    }
    #3 means CI below zero so decreasing
    #1 means CI above zero so increasing
    #2 means CI overlaps zero so flat

def produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs, noise):
    """Evaluate each param combo for one model and return comparison results"""
    param_grid = list(product(*parameters.values()))
    index_map = {
        combo: [parameters['sigma'].index(combo[0]), parameters['a'].index(combo[1]), parameters['b'].index(combo[2])]
        for combo in param_grid
    }

    def evaluate_combo(index, combo):
        sigma, a, b = combo
        result_dict = produce_confidence_intervals(paradigm, model, sigma, a, b, n_jobs, n_simulations, noise)
        #This is the thing that was changed!
        match = [1 if result_dict[k] == experimental_results[k] else 0 for k in result_dict]
        return index, match

    results_comparison = np.zeros((len(param_grid), 6))
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
        'model_comparison': results_comparison,
        'max_same': np.unique(max_same, axis=0),
        'no_max_same': no_max_same,
        'parameters_of_max': parameters_of_max
    }

def producing_fig_5(parameters, paradigm, n_simulations, n_jobs, noise):
    """Outputs the final figure as well as all the possible maximum results"""
    experimental_results = experimental_face_results if paradigm == 'face' else experimental_grating_results
    # fig_5_dict = {model: produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs) for model in range(1, 13)}
    fig_5_dict = {
        model: produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs, noise)
        for model in range(7, 13)
    }
    fig_5_array = np.zeros((6, 12))
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
    'sigma' : [0.2],
    'a' : [0.7],
    'b' : [0.2]
}

good_spread_parameters = {
    'sigma' : [0.1, 0.2, 0.3, 0.7, 1.0],
    'a' : [0.1, 0.4, 0.7, 0.9],
    'b' : [0.1, 0.5, 0.9, 1.5]
}

actual_parameters = {
    #648 parameter combinations
    'sigma' : [0.1, 0.3, 0.5, 0.7, 0.9, 2, 5, 8, 11],
    'a' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'b' : [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
}    


experimental_face_results = {
    'AM' : 3,
    'WC' : 3,
    'BC' : 3,
    'CP' : 3,
    'AMS' : 1,
    'AMA' : 1
}

experimental_grating_results = {
    'AM' : 3,
    'WC' : 3,
    'BC' : 3,
    'CP' : 1,
    'AMS' : 3,
    'AMA' : 1
}


n_simulations = 50
noise = 0.1

results = producing_fig_5(good_spread_parameters, 'face', n_simulations, n_jobs, noise)

with open("fig5_face_proper_noise_7to12.txt", "w") as f:
    from pprint import pformat
    f.write(pformat(results))




















def simulate_subject(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num):
    """Produces the voxel pattern for one simulation for one parameter combination of one paradigm"""
    batch_size=3
    T = len(j)
    noisy_pattern = np.empty((sub_num, T, v))
    for i in range(0, sub_num, batch_size):
        current_batch_size = min(batch_size, sub_num - i)
        out = simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, current_batch_size)
        noisy_pattern[i:i+current_batch_size] = (
            out.transpose(0, 2, 1) + np.random.randn(current_batch_size, v, len(j)) * noise
        ).transpose(0, 2, 1)  # shape: (batch, time, voxel)
    if paradigm == 'face':
        # Build condition indices using np.arange
        cond1_p1 = np.arange(0, 32, 4)
        cond1_p2 = np.arange(1, 32, 4)
        cond2_p1 = np.arange(2, 32, 4)
        cond2_p2 = np.arange(3, 32, 4)

        # Stack condition slices
        pattern_split = np.stack([
            noisy_pattern[:, cond1_p1, :],
            noisy_pattern[:, cond1_p2, :],
            noisy_pattern[:, cond2_p1, :],
            noisy_pattern[:, cond2_p2, :]
        ], axis=1)  # shape: (sub_num, 4, 8, v)

        reshaped = pattern_split.reshape(sub_num, 32, v)  # shape: (sub_num, 32, v)

        return reshaped
    elif paradigm == 'grating':
        pattern_split = np.stack([
            noisy_pattern[:, ind['cond1_p1'], :],
            noisy_pattern[:, ind['cond1_p3'], :],
            noisy_pattern[:, ind['cond2_p1'], :],
            noisy_pattern[:, ind['cond2_p3'], :]
        ], axis=1)
        reshaped = pattern_split.reshape(sub_num, 32, v)
        return reshaped

def produce_slopes_one_simulation(paradigm, model_type, sigma, a, b, noise, n_jobs):
    """Produces the slope of each data feature for one parameter combination for one simulation"""
    # print("one simulation done")
    v, X = 200, np.pi
    cond1, cond2 = X/4, 3*X/4
    sub_num = 18
    # noise = 0.03
    N = 8

    j, ind, reset_after, _ = paradigm_setting(paradigm, cond1, cond2)
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(simulate_subject)(sub, v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num)
    #     for sub in range(sub_num)
    # )
    #y is an 18 x 32 x 200
    # y = np.array([results[sub] for sub in range(sub_num)])
    y = simulate_subject(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num)
    return produce_slopes(y, 1)
    
def produce_confidence_intervals(paradigm, model_type, sigma, a, b, n_jobs, n_simulations, noise):
    """Produces a dictionary of whether a data feature increases, decreases, or does not change significantly for the average of n_simulations simulations
    for one parameter combinations"""
    print("done one parameter combination for one model")

    slopes = Parallel(n_jobs=n_jobs)(
        delayed(produce_slopes_one_simulation)(paradigm, model_type, sigma, a, b, noise, n_jobs)
        for _ in range(n_simulations)
    )
    
    #Finding overall confidence interval for all simulations
    slopes = np.array(slopes)
    means, stds = slopes.mean(axis=0), slopes.std(axis=0)
    sems = stds / np.sqrt(n_simulations)
    t_critical = sp.stats.t.ppf(0.995, df=n_simulations-1)
    mega_sci = np.column_stack([means - t_critical * sems, means + t_critical * sems])

    return {
        key: 3 if ci[0] < 0 and ci[1] < 0 else
             1 if ci[0] > 0 and ci[1] > 0 else
             2 if ci[0] < 0 < ci[1] else
             4
        for key, ci in zip(['AM', 'WC', 'BC', 'CP', 'AMS', 'AMA'], mega_sci)
    }
    #3 means CI below zero so decreasing
    #1 means CI above zero so increasing
    #2 means CI overlaps zero so flat

def produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs, noise):
    """Evaluate each param combo for one model and return comparison results"""
    param_grid = list(product(*parameters.values()))
    index_map = {
        combo: [parameters['sigma'].index(combo[0]), parameters['a'].index(combo[1]), parameters['b'].index(combo[2])]
        for combo in param_grid
    }

    def evaluate_combo(index, combo):
        sigma, a, b = combo
        result_dict = produce_confidence_intervals(paradigm, model, sigma, a, b, n_jobs, n_simulations, noise)
        #This is the thing that was changed!
        match = [1 if result_dict[k] == experimental_results[k] else 0 for k in result_dict]
        return index, match

    results_comparison = np.zeros((len(param_grid), 6))
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
        'model_comparison': results_comparison,
        'max_same': np.unique(max_same, axis=0),
        'no_max_same': no_max_same,
        'parameters_of_max': parameters_of_max
    }

def producing_fig_5(parameters, paradigm, n_simulations, n_jobs, noise):
    """Outputs the final figure as well as all the possible maximum results"""
    experimental_results = experimental_face_results if paradigm == 'face' else experimental_grating_results
    # fig_5_dict = {model: produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs) for model in range(1, 13)}
    fig_5_dict = {
        model: produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs, noise)
        for model in range(1, 7)
    }
    fig_5_array = np.zeros((6, 12))
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
    'sigma' : [0.2],
    'a' : [0.7],
    'b' : [0.2]
}

good_spread_parameters = {
    'sigma' : [0.1, 0.2, 0.3, 0.7, 1.0],
    'a' : [0.1, 0.4, 0.7, 0.9],
    'b' : [0.1, 0.5, 0.9, 1.5]
}

actual_parameters = {
    #648 parameter combinations
    'sigma' : [0.1, 0.3, 0.5, 0.7, 0.9, 2, 5, 8, 11],
    'a' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'b' : [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
}    


experimental_face_results = {
    'AM' : 3,
    'WC' : 3,
    'BC' : 3,
    'CP' : 3,
    'AMS' : 1,
    'AMA' : 1
}

experimental_grating_results = {
    'AM' : 3,
    'WC' : 3,
    'BC' : 3,
    'CP' : 1,
    'AMS' : 3,
    'AMA' : 1
}


n_simulations = 50
noise = 0.1

results = producing_fig_5(good_spread_parameters, 'grating', n_simulations, n_jobs, noise)

with open("fig5_grating_proper_noise_1to6.txt", "w") as f:
    from pprint import pformat
    f.write(pformat(results))






























def simulate_subject(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num):
    """Produces the voxel pattern for one simulation for one parameter combination of one paradigm"""
    batch_size=3
    T = len(j)
    noisy_pattern = np.empty((sub_num, T, v))
    for i in range(0, sub_num, batch_size):
        current_batch_size = min(batch_size, sub_num - i)
        out = simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, current_batch_size)
        noisy_pattern[i:i+current_batch_size] = (
            out.transpose(0, 2, 1) + np.random.randn(current_batch_size, v, len(j)) * noise
        ).transpose(0, 2, 1)  # shape: (batch, time, voxel)
    if paradigm == 'face':
        # Build condition indices using np.arange
        cond1_p1 = np.arange(0, 32, 4)
        cond1_p2 = np.arange(1, 32, 4)
        cond2_p1 = np.arange(2, 32, 4)
        cond2_p2 = np.arange(3, 32, 4)

        # Stack condition slices
        pattern_split = np.stack([
            noisy_pattern[:, cond1_p1, :],
            noisy_pattern[:, cond1_p2, :],
            noisy_pattern[:, cond2_p1, :],
            noisy_pattern[:, cond2_p2, :]
        ], axis=1)  # shape: (sub_num, 4, 8, v)

        reshaped = pattern_split.reshape(sub_num, 32, v)  # shape: (sub_num, 32, v)

        return reshaped
    elif paradigm == 'grating':
        pattern_split = np.stack([
            noisy_pattern[:, ind['cond1_p1'], :],
            noisy_pattern[:, ind['cond1_p3'], :],
            noisy_pattern[:, ind['cond2_p1'], :],
            noisy_pattern[:, ind['cond2_p3'], :]
        ], axis=1)
        reshaped = pattern_split.reshape(sub_num, 32, v)
        return reshaped

def produce_slopes_one_simulation(paradigm, model_type, sigma, a, b, noise, n_jobs):
    """Produces the slope of each data feature for one parameter combination for one simulation"""
    # print("one simulation done")
    v, X = 200, np.pi
    cond1, cond2 = X/4, 3*X/4
    sub_num = 18
    # noise = 0.03
    N = 8

    j, ind, reset_after, _ = paradigm_setting(paradigm, cond1, cond2)
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(simulate_subject)(sub, v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num)
    #     for sub in range(sub_num)
    # )
    #y is an 18 x 32 x 200
    # y = np.array([results[sub] for sub in range(sub_num)])
    y = simulate_subject(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind, sub_num)
    return produce_slopes(y, 1)
    
def produce_confidence_intervals(paradigm, model_type, sigma, a, b, n_jobs, n_simulations, noise):
    """Produces a dictionary of whether a data feature increases, decreases, or does not change significantly for the average of n_simulations simulations
    for one parameter combinations"""
    print("done one parameter combination for one model")

    slopes = Parallel(n_jobs=n_jobs)(
        delayed(produce_slopes_one_simulation)(paradigm, model_type, sigma, a, b, noise, n_jobs)
        for _ in range(n_simulations)
    )
    
    #Finding overall confidence interval for all simulations
    slopes = np.array(slopes)
    means, stds = slopes.mean(axis=0), slopes.std(axis=0)
    sems = stds / np.sqrt(n_simulations)
    t_critical = sp.stats.t.ppf(0.995, df=n_simulations-1)
    mega_sci = np.column_stack([means - t_critical * sems, means + t_critical * sems])

    return {
        key: 3 if ci[0] < 0 and ci[1] < 0 else
             1 if ci[0] > 0 and ci[1] > 0 else
             2 if ci[0] < 0 < ci[1] else
             4
        for key, ci in zip(['AM', 'WC', 'BC', 'CP', 'AMS', 'AMA'], mega_sci)
    }
    #3 means CI below zero so decreasing
    #1 means CI above zero so increasing
    #2 means CI overlaps zero so flat

def produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs, noise):
    """Evaluate each param combo for one model and return comparison results"""
    param_grid = list(product(*parameters.values()))
    index_map = {
        combo: [parameters['sigma'].index(combo[0]), parameters['a'].index(combo[1]), parameters['b'].index(combo[2])]
        for combo in param_grid
    }

    def evaluate_combo(index, combo):
        sigma, a, b = combo
        result_dict = produce_confidence_intervals(paradigm, model, sigma, a, b, n_jobs, n_simulations, noise)
        #This is the thing that was changed!
        match = [1 if result_dict[k] == experimental_results[k] else 0 for k in result_dict]
        return index, match

    results_comparison = np.zeros((len(param_grid), 6))
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
        'model_comparison': results_comparison,
        'max_same': np.unique(max_same, axis=0),
        'no_max_same': no_max_same,
        'parameters_of_max': parameters_of_max
    }

def producing_fig_5(parameters, paradigm, n_simulations, n_jobs, noise):
    """Outputs the final figure as well as all the possible maximum results"""
    experimental_results = experimental_face_results if paradigm == 'face' else experimental_grating_results
    # fig_5_dict = {model: produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs) for model in range(1, 13)}
    fig_5_dict = {
        model: produce_model_key_variables(model, parameters, paradigm, experimental_results, n_simulations, n_jobs, noise)
        for model in range(7, 13)
    }
    fig_5_array = np.zeros((6, 12))
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
    'sigma' : [0.2],
    'a' : [0.7],
    'b' : [0.2]
}

good_spread_parameters = {
    'sigma' : [0.1, 0.2, 0.3, 0.7, 1.0],
    'a' : [0.1, 0.4, 0.7, 0.9],
    'b' : [0.1, 0.5, 0.9, 1.5]
}

actual_parameters = {
    #648 parameter combinations
    'sigma' : [0.1, 0.3, 0.5, 0.7, 0.9, 2, 5, 8, 11],
    'a' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'b' : [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
}    


experimental_face_results = {
    'AM' : 3,
    'WC' : 3,
    'BC' : 3,
    'CP' : 3,
    'AMS' : 1,
    'AMA' : 1
}

experimental_grating_results = {
    'AM' : 3,
    'WC' : 3,
    'BC' : 3,
    'CP' : 1,
    'AMS' : 3,
    'AMA' : 1
}


n_simulations = 50
noise = 0.1

results = producing_fig_5(good_spread_parameters, 'grating', n_simulations, n_jobs, noise)

with open("fig5_grating_proper_noise_7to13.txt", "w") as f:
    from pprint import pformat
    f.write(pformat(results))