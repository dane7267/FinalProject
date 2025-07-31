import numpy as np
import scipy as sp
from paradigm_setting import paradigm_setting

np.set_printoptions(threshold=np.inf)

def simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, sub_num):
    """
    Parameters
    ----------
    v : integer
        The total number of voxels in the simulation
    X : float
        The length of the stimulus dimension (e.g angle range of grating)
    j : list
        A list of the condition orders (e.g [cond1, cond2, cond2, cond1])
    cond1 : float
        Stimulus value for condition 1 (1/4pi)
    cond2 : float
        Stimulus value for condition 2 (3/4pi)
    a : float
        The amount of adaptation
    b : float
        The extent of the domain adaptation
    sigma : float
        The width of the tuning curve
    model_type : integer
        The type of model demonstrated, represented by an integer from 1 to 12
    reset_after : integer
        The number of conditions to go through before needing to reset the adaptation factor c
    paradigm : string
        The paradigm being demonstrated: either 'face' or 'grating'
    N : integer
        The number of neuronal populations per voxel

    
    Returns
    -------
    out : dictionary
        A dictionary representing every run of the paradigm and the  """
    
    nt = len(j)
    res = 180
    dt = X/res
    x = np.arange(dt, X + dt, dt)
    tuning_curves_peaks = np.array([0, X/8, X*2/8, X*3/8, X*4/8, X*5/8, X*6/8, X*7/8])
    precomputed_gaussians = np.array([gaussian(x, u, sigma, paradigm) for u in tuning_curves_peaks])
    precomputed_gaussians /= np.max(precomputed_gaussians, axis=1, keepdims=True)
    pattern = np.zeros((sub_num, nt, v))
    # activity = np.zeros((nt, v, N)) 
    # rep = np.zeros((nt, v, N, res)) 

    #Randomly assign preferred tuning curves to neurons
    # u_vals = np.random.choice(tuning_curves_peaks, size=(sub_num, v,N), replace=True) #200x8
    u_vals = np.array([0, X/8, 2*X/8, X/8, 0, 5*X/8, 7*X/8, 4*X/8])
    u_vals = np.tile(u_vals[None, None, :], (sub_num, v,1))
    u_indices = np.searchsorted(tuning_curves_peaks, u_vals) #This maps the randomly selected values back to their positions in the original array
    init = precomputed_gaussians[u_indices] #init is 1 x 8 x 20
    
    #Create reset mask (this is True for every interval of reset after)
    reset_mask = np.mod(np.arange(nt), reset_after) == 0
    c = np.ones((sub_num, nt, v, N)) #Adaptation factor for every trial, voxel, and neuron

    #Compute adaptation for all trials at once using broadcasting
    d = u_vals[:, None, :, :] - np.array(j)[None, :, None, None]
    if paradigm == 'grating':
        d = np.minimum(np.abs(d), X-np.abs(d))

    scaling_factors = {
        2: np.minimum(1, (a + np.abs(d / b) * (1 - a))),  # Local Scaling
        3: np.maximum(a, (1 - np.abs(d / b) * (1 - a))),  # Remote Scaling
        5: np.minimum(1, (a + np.abs(d / b) * (1 - a))),  # Local Sharpening
        6: np.maximum(a, (1 - np.abs(d / b) * (1 - a))),  # Remote Sharpening
        7: a * np.sign(d), #Global Repulsion
        8: np.sign(d) *np.minimum(1, (a + np.abs(d / b) * (1 - a))),  # Local Repulsion
        9: np.sign(d) * np.maximum(a, (1 - np.abs(d / b) * (1 - a))),  # Remote Repulsion
        10: a * np.sign(d), #Global Attraction
        11: np.sign(d) * np.minimum(1, (a + np.abs(d / b) * (1 - a))),  # Local Attraction
        12: np.sign(d) * np.maximum(a, (1 - np.abs(d / b) * (1 - a))),  # Remote Attraction
    }

    shifting_models = {7, 8, 9, 10, 11, 12}  # Models that involve shifting
    if model_type in {1, 4}:
        e = a * np.ones_like(d)
    elif model_type in scaling_factors:
        e = scaling_factors[model_type]

    num_blocks = nt // reset_after
    e_reshaped = e.reshape(sub_num, num_blocks, reset_after, v, N)
    e_modified = np.ones_like(e_reshaped)
    e_modified[:, :, 1:, :, :] = e_reshaped[:, :, 1:, :, :]
    transformed_array = np.cumprod(e_modified, axis=2)
    transformed_array = transformed_array.reshape(sub_num, nt, v, N)


    if model_type in {4, 5, 6}:
        temp = gaussian(x[None, None, None, None, :], u_vals[:, None, :, :, None], transformed_array[..., None] * sigma, paradigm)
        temp /= np.max(temp, axis=-1, keepdims=True)
    elif model_type in shifting_models:  # Shift-based models
        shift_direction = 1 if model_type in {7, 8, 9} else -1  # Repulsive (+) vs. Attractive (-)
        shift_amount = transformed_array * X/2
        shift_amount[:, ::reset_after, :, :] = 1
        temp = gaussian(x[None, None, None, None, :], u_vals[:, None, :, :, None] + shift_direction * shift_amount[..., None], sigma, paradigm)
        temp /= np.max(temp, axis=-1, keepdims=True)
    elif model_type in {1, 2, 3}:  # Scaling models (1, 2, 3)
        temp = transformed_array[..., None] * init[:, None, :, :, :]
    rep = temp
    rep[:, ::reset_after, :, :, :] = init[:, None, :, :, :] #unsure about this line


    cond_indices = (int(cond1 / dt - 1), int(cond2 / dt - 1))

    cond1_mask = (np.array(j) == cond1)[None, :, None, None]
    cond2_mask = (np.array(j) == cond2)[None, :, None, None]

    act1 = np.take(rep, cond_indices[0], axis=-1)
    act2 = np.take(rep, cond_indices[1], axis=-1)

    activity = act1 * cond1_mask + act2 * cond2_mask
    pattern = np.mean(activity, axis=3)

    return {
        'pattern': pattern,
        'rep': rep,
        'activity': activity
    }

def gaussian(x, u, sigma, paradigm):
    if paradigm == 'face':
        return non_circular_g(x, sigma, u)
    elif paradigm == 'grating':
        return circular_g(2*x, 2*u, 1/sigma)
    

def non_circular_g(x, sigma, u):
    return np.exp(-((x-u)**2)/(2*sigma**2))

def circular_g(x, u, sigma):
    c = 1 / (2*np.pi*sp.special.i0(sigma))
    return c * np.exp(sigma * np.cos(x-u))

# v = 1
# X = np.pi
# cond1, cond2 = 1/4*X, 3/4*X
# a = 0.1
# b = 0.1
# sigma = 0.1
# model_type = 1
# paradigm = 'face'
# N = 8
# sub_num = 2
# j, ind, reset_after, winning_params = paradigm_setting('face', cond1, cond2)
# print(simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, sub_num))