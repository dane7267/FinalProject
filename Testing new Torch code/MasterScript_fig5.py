import numpy as np
from paradigm_setting import paradigm_setting
from simulate_adaptation import simulate_adaptation
from repeffects_fig4 import repeffects_fig4
import sys
from ExperimentalData import create_pattern
from itertools import product
np.set_printoptions(threshold=sys.maxsize)
from joblib import Parallel, delayed
n_jobs = 1
# n_jobs = os.cpu_count() // 2

def produce_results_dictionary(paradigm, model_type, sigma, a, b):
    v = 200
    X = np.pi #Stimulus Dimension
    cond1 = 1/4*X #Condition 1 of paradigm
    cond2 = 3/4*X #Condition 2 of paradigm
    sub_num = 18 #The number of subjects. Keep at 18
    noise = 0.03 
    N = 8
    y = {}
    j, ind, reset_after, _ = paradigm_setting(paradigm, cond1, cond2) #We don't use winning_params, we use sigma, a, b
    for sub in range(sub_num):
        out = simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N)
        pattern = (out['pattern'].T + np.random.randn(v, len(j)) * noise).T
        v = pattern.shape[1]
        if paradigm == 'face':
            #print("Simulating face data")
            #The core function that generates the initial and repeated patterns - outputs: [presentation x voxel] matrix
            #Now group trials according to conditions and presentations
            cond1_p = {
                1: pattern[::4, :v],  # Rows 1:4:end (0-based index)
                2: pattern[1::4, :v]  # Rows 2:4:end (0-based index)
            }
            cond2_p = {
                1: pattern[2::4, :v],  # Rows 3:4:end (0-based index)
                2: pattern[3::4, :v]  # Rows 4:4:end (0-based index)
            }
        elif paradigm == 'grating':
            #print("Simulating grating data")
            cond1_p = {
                1: pattern[ind['cond1_p1'], :v],  # Using the indices for condition 1, part 1
                2:pattern[ind['cond1_p3'], :v]  # Using the indices for condition 1, part 2
            }
            cond2_p = {
                1: pattern[ind['cond2_p1'], :v],  # Using the indices for condition 2, part 1
                2: pattern[ind['cond2_p3'], :v]  # Using the indices for condition 2, part 2
            }
        y[sub] = np.vstack([cond1_p[1], cond1_p[2], cond2_p[1], cond2_p[2]])
    return repeffects_fig4(y,1)
    

def run_simulation(model, parameters, paradigm, experimental_results):
    num_combinations = np.prod([len(v) for v in parameters.values()])
    results_comparison = np.zeros((num_combinations, 6))
    max_same, parameters_of_max = [], []
    no_max_same = 0

    parameter_list = list(product(*parameters.values()))
    parameter_indices = {
        (sigma, a, b): [parameters['sigma'].index(sigma), parameters['a'].index(a), parameters['b'].index(b)]
        for sigma, a, b, in parameter_list
    }
    def process_combination(index, combination):
        sigma, a, b = combination
        results_dict = produce_results_dictionary(paradigm, model, sigma, a, b)
        results_match = [1 if results_dict[feature] == experimental_results[feature] else 0 for feature in results_dict]
        return index, results_match

    parallel_results = Parallel(n_jobs)(
        delayed(process_combination)(i, combination) for i, combination in enumerate(parameter_list)
    )

    for index, results_match in parallel_results:
        results_comparison[index] = results_match
        current_max = sum(results_match)

        if current_max > no_max_same:
            no_max_same = current_max
            max_same = [results_match]
            parameters_of_max = [parameter_indices[parameter_list[index]]]
        elif current_max == no_max_same:
            max_same.append(results_match)
            parameters_of_max.append(parameter_indices[parameter_list[index]])

    return {
        'model_comparison' : results_comparison,
        'max_same' : np.unique(max_same, axis=0),
        'no_max_same' : no_max_same,
        'parameters_of_max' : parameters_of_max
    }
def producing_fig_5(parameters, paradigm):
    experimental_results = experimental_face_results if paradigm == 'face' else experimental_grating_results
    fig_5_dict = {model: run_simulation(model, parameters, paradigm, experimental_results) for model in range(1, 13)}
    fig_5_array = np.zeros((6, 12))
    fig_5_sets = []
    for model, model_data in fig_5_dict.items():
        fig_5_sets.append(model_data['max_same'])
        for i, max_match in enumerate(model_data['max_same'][0]):
            fig_5_array[i, model - 1] = max_match
    print(fig_5_array, fig_5_sets)
    return fig_5_dict

parameters = {
    'sigma' : [0.2, 5],
    'a' : [0.2],
    'b' : [0.2, 0.4]
}

good_spread_parameters = {
    'sigma' : [0.1, 2, 11],
    'a' : [0.1, 0.5, 0.9],
    'b' : [0.1, 0.7, 1.5]
}

actual_parameters = {
    #648 parameter combinations
    'sigma' : [0.1, 0.3, 0.5, 0.7, 0.9, 2, 5, 8, 11],
    'a' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'b' : [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
}

model_1_2 = {
    'global scaling' : 1,
    'local scaling' : 2,
    'remote scaling' : 3,
    'global sharpening' : 4
}

models = {
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

paradigms = ['face', 'grating']        

# fig_5_top_dictionary = {
#     model : {
#         #'model_comparison' : np.zeros((648, 4)),
#         'model_comparison' : np.zeros((648, 6)),
#         'max_same' : [[]],
#         'no_max_same' : 0,
#         'parameters_of_max' : [[]]
#     }
#     for model in range(7, 13)
# }

# fig_5_bottom_dictionary = {
#     model : {
#         #'model_comparison' : np.zeros((648, 4)),
#         'model_comparison' : np.zeros((648, 6)),
#         'max_same' : [[]],
#         'no_max_same' : 0,
#         'parameters_of_max' : [[]]
#     }
#     for model in range(7, 8)
# }

experimental_face_results = {
    #This will be produced by other code in final version
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

# fig_5_top_array = np.zeros((6, 12))
# fig_5_bottom_array = np.zeros((6,12))
# fig_5_sets = []

# def producing_fig_5(parameters, paradigm):
#     #This would take about 15 hours to run all parameters and models!
#     if paradigm == 'face':
#         for model in fig_5_top_dictionary:
#             for count1, combination in enumerate(product(*parameters.values())):
#                 #print('new parameters')
#                 print(count1)
#                 sigma, a, b = combination
#                 parameter_indices = [parameters['sigma'].index(sigma), parameters['a'].index(a), parameters['b'].index(b)]
#                 results_dict = produce_results_dictionary(paradigm, model, sigma, a, b)
#                 results_match = [
#                     1 if results_dict[feature] == experimental_face_results[feature] else 0
#                     for feature in results_dict
#                 ]
#                 fig_5_top_dictionary[model]['model_comparison'][count1] = results_match
#                 current_max = sum(results_match)
#                 previous_max = fig_5_top_dictionary[model]['no_max_same']
#                 if current_max > previous_max:
#                     fig_5_top_dictionary[model]['no_max_same'] = current_max
#                     fig_5_top_dictionary[model]['max_same'] = [results_match]
#                     fig_5_top_dictionary[model]['parameters_of_max'] = [parameter_indices]
#                 elif current_max == previous_max:
#                     fig_5_top_dictionary[model]['max_same'].append(results_match)
#                     fig_5_top_dictionary[model]['parameters_of_max'].append(parameter_indices)    
#             print(fig_5_top_dictionary[model]['max_same'])
#             print(np.unique(fig_5_top_dictionary[model]['max_same'], axis=0))
#             fig_5_sets.append(np.unique(fig_5_top_dictionary[model]['max_same'], axis=0))
#             for i, max_match in enumerate(fig_5_top_dictionary[model]['max_same'][0]):
#                 fig_5_top_array[i, model - 1] = max_match
#             print(fig_5_top_array)
#         print(fig_5_top_array)
#         print(fig_5_sets)
#         #This would take about 15 hours to run all parameters and models!
#     if paradigm == 'grating':
#         for model in fig_5_bottom_dictionary:
#             for count1, combination in enumerate(product(*parameters.values())):
#                 #print('new parameters')
#                 print(count1)
#                 sigma, a, b = combination
#                 parameter_indices = [parameters['sigma'].index(sigma), parameters['a'].index(a), parameters['b'].index(b)]
#                 results_dict = produce_results_dictionary(paradigm, model, sigma, a, b)
#                 results_match = [
#                     1 if results_dict[feature] == experimental_grating_results[feature] else 0
#                     for feature in results_dict
#                 ]
#                 fig_5_bottom_dictionary[model]['model_comparison'][count1] = results_match
#                 current_max = sum(results_match)
#                 previous_max = fig_5_bottom_dictionary[model]['no_max_same']
#                 if current_max > previous_max:
#                     fig_5_bottom_dictionary[model]['no_max_same'] = current_max
#                     fig_5_bottom_dictionary[model]['max_same'] = [results_match]
#                     fig_5_bottom_dictionary[model]['parameters_of_max'] = [parameter_indices]
#                 elif current_max == previous_max:
#                     fig_5_bottom_dictionary[model]['max_same'].append(results_match)
#                     fig_5_bottom_dictionary[model]['parameters_of_max'].append(parameter_indices)    
#             print(fig_5_bottom_dictionary[model]['max_same'])
#             print(np.unique(fig_5_bottom_dictionary[model]['max_same'], axis=0))
#             fig_5_sets.append(np.unique(fig_5_bottom_dictionary[model]['max_same'], axis=0))
#             for i, max_match in enumerate(fig_5_bottom_dictionary[model]['max_same'][0]):
#                 fig_5_bottom_array[i, model - 1] = max_match
#             print(fig_5_bottom_array)
#         print(fig_5_bottom_array)
#         print(fig_5_sets)
#     return fig_5_top_dictionary

producing_fig_5(parameters, 'grating')