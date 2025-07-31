import numpy as np
import scipy as sp
from paradigm_setting import paradigm_setting
import torch
np.set_printoptions(threshold=np.inf)

def simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, k, model_type, reset_after, paradigm, N, tuning_curves_indices, sub_num):
    # print(model_type)
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
    res = 150
    dt = X/res
    x = torch.arange(dt, X + dt, dt, dtype=torch.float32)

    tuning_curves_peaks = torch.tensor([0, X/8, X*2/8, X*3/8, X*4/8, X*5/8, X*6/8, X*7/8], dtype=torch.float32)
    tuning_curves_peaks_np = [0, X/8, X*2/8, X*3/8, X*4/8, X*5/8, X*6/8, X*7/8]
    precomputed_gaussians = torch.stack([gaussian(x, u, sigma, paradigm) for u in tuning_curves_peaks])
    # print(precomputed_gaussians.shape)
    # print(torch.max(precomputed_gaussians, axis=1, keepdims=True).shape)
    precomputed_gaussians = precomputed_gaussians / torch.max(precomputed_gaussians, dim=1, keepdims=True)[0]

    pattern = torch.zeros((sub_num, nt, v), dtype=torch.float32)
    activity = torch.zeros((sub_num, nt, v, N), dtype=torch.float32) 
    rep = torch.zeros((sub_num, nt, v, N, res), dtype=torch.float32) 

    # u = [3*X/8 X/8 3*X/8 X/8 X/8 X/8 5*X/8 X/8];
    #Randomly assign preferred tuning curves to neurons
    
    # u_vals = torch.tensor(np.random.choice(tuning_curves_peaks_np, size=(v,N), replace=True), dtype=torch.float32, requires_grad=True) #200x8
    # print(u_vals)
    # u_vals = torch.tensor(
    #     torch.multinomial(torch.ones(len(tuning_curves_peaks)), num_samples=(v, N), replacement=True),
    #     dtype=torch.float32,
    #     requires_grad=True
    # )
    u_vals = tuning_curves_peaks[tuning_curves_indices]
    """This should be size sub_num, v, N"""
    # u_vals.requires_grad_()
    # u = np.array([3*X/8, X/8, 3*X/8, X/8, X/8, X/8, 5*X/8, X/8])
    # u = np.tile(u, (v, 1))
    # u_vals = torch.tensor(u, dtype=torch.float32, requires_grad=True)
    u_indices = torch.searchsorted(tuning_curves_peaks, u_vals) #This maps the randomly selected values back to their positions in the original array
    init = precomputed_gaussians[u_indices] #init is 1 x 8 x 20
    #Create reset mask (as before, this is True every interval of reset after)
    # reset_mask = torch.mod(torch.arange(nt), reset_after) == 0
    # c = torch.ones((nt, v, N), dtype=torch.float32, requires_grad=True) #Adaptation factor for every trial, voxel, and neuron
    #Compute adaptation for all trials at once using broadcasting
    # d = np.abs(u_vals[None, :, :] - np.array(j)[:, None, None])
    # d = u_vals[None, :, :] - torch.tensor(j, dtype=torch.float32, requires_grad=True)[:, None, None]
    j_tens = torch.tensor(j, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    d = u_vals[:, None, :, :] - j_tens
    #u_vals[None, :, :] is 1 x 200 x 8
    #j is 48 so 48 x 1 x 1 meaning these can broadcast when subtracting!

    if paradigm == 'grating':
        d = torch.minimum(torch.abs(d), X-torch.abs(d))

    if model_type == 1:
        temp = produce_temp_1(a, nt, v, N, reset_after, init, d, sub_num)
    elif model_type == 2:
        temp = produce_temp_2(a, b, d, nt, reset_after, v, N, init, sub_num)
    elif model_type == 3:
        temp = produce_temp_3(a, b, d, nt, reset_after, v, N, init, sub_num)
    elif model_type == 4:
        temp = produce_temp_4(a, nt, v, N, reset_after, sigma, paradigm, u_vals, x, d, sub_num)
    elif model_type == 5:
        temp = produce_temp_5(a, nt, v, N, reset_after, sigma, paradigm, u_vals, x, d, b, sub_num)
    elif model_type == 6:
        temp = produce_temp_6(a, nt, v, N, reset_after, sigma, paradigm, u_vals, x, d, b, sub_num)
    elif model_type == 7:
        temp = produce_temp_7(a, d, nt, reset_after, v, N, x, u_vals, sigma, paradigm, X, sub_num)
    elif model_type == 8:
        temp = produce_temp_8(a, d, nt, reset_after, v, N, x, u_vals, sigma, paradigm, X, b, sub_num)
    elif model_type == 9:
        temp = produce_temp_9(a, d, nt, reset_after, v, N, x, u_vals, sigma, paradigm, X, b, sub_num)
    elif model_type == 10:
        temp = produce_temp_10(a, d, nt, reset_after, v, N, x, u_vals, sigma, paradigm, X, sub_num)
    elif model_type == 11:
        temp = produce_temp_11(a, d, nt, reset_after, v, N, x, u_vals, sigma, paradigm, X, b, sub_num)
    elif model_type == 12:
        temp = produce_temp_12(a, d, nt, reset_after, v, N, x, u_vals, sigma, paradigm, X, b, sub_num)
    
    rep = temp
    rep[:, ::reset_after, :, :, :] = init[:, None, :, :, :]


    cond_indices = (int(cond1 / dt - 1), int(cond2 / dt - 1))
    activity = rep[..., cond_indices[0]] * (j[None, :, None, None] == cond1) + \
               rep[..., cond_indices[1]] * (j[None, :, None, None] == cond2)

    pattern = torch.mean(activity, dim=3)
    pattern = pattern * k
    # print(f"simulate_adaptation complete for model {model_type}")

    return pattern

def produce_temp_1(a, nt, v, N, reset_after, init, d, sub_num):
    e = a * torch.ones_like(d, dtype=torch.float32, requires_grad=True)
    transformed_array = produce_transformed_array(nt, reset_after, v, N, e, sub_num)
    temp = transformed_array[..., None] * init[:, None, :, :, :]
    return temp

def produce_temp_2(a, b, d, nt, reset_after, v, N, init, sub_num):
    e = smooth_min(torch.ones_like(d), (a + smooth_abs(d / b) * (1 - a)))  # Local Scaling
    transformed_array = produce_transformed_array(nt, reset_after, v, N, e, sub_num)
    temp = transformed_array[..., None] * init[:, None, :, :, :]
    return temp

def produce_temp_3(a, b, d, nt, reset_after, v, N, init, sub_num):
    e = smooth_max(a*torch.ones_like(d), (1 - smooth_abs(d / b) * (1 - a)))  # Remote Scaling
    transformed_array = produce_transformed_array(nt, reset_after, v, N, e, sub_num)
    temp = transformed_array[..., None] * init[:, None, :, :, :]
    return temp

def produce_temp_4(a, nt, v, N, reset_after, sigma, paradigm, u_vals, x, d, sub_num):
    # print("Entered produce_temp_4")
    # print(f"a: {a}, sigma: {sigma}, paradigm: {paradigm}")
    # print(f"x shape: {x.shape}, u_vals shape: {u_vals.shape}")
    e = a * torch.ones_like(d, dtype=torch.float16)
    transformed_array = produce_transformed_array(nt, reset_after, v, N, e, sub_num)
    # print(f"transformed_array shape: {transformed_array.shape}")
    temp = gaussian(x[None, None, None, None, :], u_vals[:, None, :, :, None], transformed_array[..., None] * sigma, paradigm)
    # print(torch.max(temp, dim=-1, keepdims=True).values)
    # print(f"temp shape: {temp.shape}, temp max: {temp.max()}, temp min: {temp.min()}")
    
    # temp = temp/ (torch.max(temp, dim=-1, keepdims=True).values +1e-6)
    
    denom = torch.max(temp, dim=-1, keepdims=True).values
    # print("max per tuning curve stats — min:", denom.min().item(), "max:", denom.max().item())
    if torch.any(torch.isnan(denom)):
        raise ValueError("NaN detected in denominator during normalization!")

    temp = temp / (denom + 1e-6)

    return temp

def produce_temp_5(a, nt, v, N, reset_after, sigma, paradigm, u_vals, x, d, b, sub_num):
    e = torch.minimum(torch.ones_like(d), (a + torch.abs(d / b) * (1 - a)))  # Local Sharpening
    transformed_array = produce_transformed_array(nt, reset_after, v, N, e, sub_num)
    temp = gaussian(x[None, None, None, None, :], u_vals[:, None, :, :, None], transformed_array[..., None] * sigma, paradigm)
    temp = temp/ (torch.max(temp, dim=-1, keepdims=True).values +1e-6)
    return temp

def produce_temp_6(a, nt, v, N, reset_after, sigma, paradigm, u_vals, x, d, b, sub_num):
    e = torch.maximum(a*torch.ones_like(d), (1 - torch.abs(d / b) * (1 - a)))  # Remote Sharpening
    transformed_array = produce_transformed_array(nt, reset_after, v, N, e, sub_num)
    temp = gaussian(x[None, None, None, None, :], u_vals[:, None, :, :, None], transformed_array[..., None] * sigma, paradigm)
    temp = temp/ (torch.max(temp, dim=-1, keepdims=True).values +1e-6)
    return temp

def produce_temp_7(a, d, nt, reset_after, v, N, x, u_vals, sigma, paradigm, X, sub_num):
    e = a * smooth_sign(d)
    transformed_array = produce_transformed_array(nt, reset_after, v, N, e, sub_num)
    shift_direction = 1
    shift_amount = transformed_array * X/2
    shift_amount[:, ::reset_after, :, :] = 1
    temp = gaussian(x[None, None, None, None, :], u_vals[:, None, :, :, None] + shift_direction * shift_amount[..., None], sigma, paradigm)
    temp =temp / (torch.max(temp, dim=-1, keepdims=True).values +1e-6)
    return temp

def produce_temp_8(a, d, nt, reset_after, v, N, x, u_vals, sigma, paradigm, X, b, sub_num):
    e = smooth_sign(d) *torch.minimum(torch.ones_like(d), (a + torch.abs(d / (b + 1e-6)) * (1 - a)))  # Local Repulsion
    # e = torch.clamp(e, -1.0, 1.0)
    transformed_array = produce_transformed_array(nt, reset_after, v, N, e, sub_num)
    shift_direction = 1
    shift_amount = transformed_array * X/2
    shift_amount[:, ::reset_after, :, :] = 1
    temp = gaussian(x[None, None, None, None, :], u_vals[:, None, :, :, None] + shift_direction * shift_amount[..., None], sigma, paradigm)
    temp =temp / (torch.max(temp, dim=-1, keepdims=True).values +1e-6)
    # Functional replacement for in-place reset
    # init_shape = u_vals[:, None, :, :, None] + shift_direction * shift_amount[..., None]
    # temp = gaussian(x[None, None, None, None, :], init_shape, sigma, paradigm)

    # # Normalize to max=1
    # temp = temp / torch.max(temp, dim=-1, keepdim=True).values
    return temp


def produce_temp_9(a, d, nt, reset_after, v, N, x, u_vals, sigma, paradigm, X, b, sub_num):
    e = smooth_sign(d) * torch.maximum(a*torch.ones_like(d), (1 - torch.abs(d / b) * (1 - a)))  # Remote Repulsion
    transformed_array = produce_transformed_array(nt, reset_after, v, N, e, sub_num)
    shift_direction = 1
    shift_amount = transformed_array * X/2
    shift_amount[:, ::reset_after, :, :] = 1
    temp = gaussian(x[None, None, None, None, :], u_vals[:, None, :, :, None] + shift_direction * shift_amount[..., None], sigma, paradigm)
    temp =temp / (torch.max(temp, dim=-1, keepdims=True).values +1e-6)
    return temp

def produce_temp_10(a, d, nt, reset_after, v, N, x, u_vals, sigma, paradigm, X, sub_num):
    e = a * smooth_sign(d)
    transformed_array = produce_transformed_array(nt, reset_after, v, N, e, sub_num)
    shift_direction = -1
    shift_amount = transformed_array * X/2
    shift_amount[:, ::reset_after, :, :] = 1
    temp = gaussian(x[None, None, None, None, :], u_vals[:, None, :, :, None] + shift_direction * shift_amount[..., None], sigma, paradigm)
    temp =temp / (torch.max(temp, dim=-1, keepdims=True).values +1e-6)
    return temp

def produce_temp_11(a, d, nt, reset_after, v, N, x, u_vals, sigma, paradigm, X, b, sub_num):
    e = smooth_sign(d) * torch.minimum(torch.ones_like(d), (a + torch.abs(d / b) * (1 - a)))  # Local Attraction
    transformed_array = produce_transformed_array(nt, reset_after, v, N, e, sub_num)
    shift_direction = -1
    shift_amount = transformed_array * X/2
    shift_amount[:, ::reset_after, :, :] = 1
    # print(shift_amount)
    temp = gaussian(x[None, None, None, None, :], u_vals[:, None, :, :, None] + shift_direction * shift_amount[..., None], sigma, paradigm)
    # print("temp")
    # print(temp)
    temp =temp / (torch.max(temp, dim=-1, keepdims=True).values +1e-6)
    return temp

def produce_temp_12(a, d, nt, reset_after, v, N, x, u_vals, sigma, paradigm, X, b, sub_num):
    e = smooth_sign(d) * torch.maximum(a*torch.ones_like(d), (1 - torch.abs(d / b) * (1 - a)))  # Remote Attraction
    transformed_array = produce_transformed_array(nt, reset_after, v, N, e, sub_num)
    shift_direction = -1
    shift_amount = transformed_array * X/2
    shift_amount[:, ::reset_after, :, :] = 1
    temp = gaussian(x[None, None, None, None, :], u_vals[:, None, :, :, None] + shift_direction * shift_amount[..., None], sigma, paradigm)
    temp =temp / (torch.max(temp, dim=-1, keepdims=True).values +1e-6)
    return temp


def produce_transformed_array(nt, reset_after, v, N, e, sub_num):
    num_blocks = nt // reset_after
    # print("e shape:", e.shape)
    # print("reshaping to:", (sub_num, nt // reset_after, reset_after, v, N))
    e_reshaped = e.reshape(sub_num, num_blocks, reset_after, v, N)
    e_modified = torch.ones_like(e_reshaped)
    e_modified[:, :, 1:, :, :] = e_reshaped[:, :, 1:, :, :]
    transformed_array = torch.cumprod(e_modified, dim=2)
    transformed_array = transformed_array.reshape(sub_num, nt, v, N)
    return transformed_array

def gaussian(x, u, sigma, paradigm):
    if paradigm == 'face':
        return non_circular_g(x, sigma, u)
    elif paradigm == 'grating':
        g = circular_g(2 * x, 2 * u, 1 / sigma)

        # Batched NaN handling
        nan_mask = torch.isnan(g)
        if nan_mask.any():
            # Replace NaN-affected entries with delta-like peaks
            g_clean = g.clone()
            # g and x must be same shape: [..., res]
            # Find closest x to u in last dim
            delta_idx = torch.argmin(torch.abs(x - u), dim=-1, keepdim=True)  # shape [..., 1]

            # Flatten to index cleanly
            flat_idx = torch.arange(g.numel(), device=g.device).reshape(g.shape)
            mask_flat = nan_mask.reshape(-1, g.shape[-1])
            g_flat = g_clean.reshape(-1, g.shape[-1])

            for i in range(g_flat.shape[0]):
                if mask_flat[i].any():
                    g_flat[i].zero_()
                    g_flat[i, delta_idx.reshape(-1)[i]] = 1.0

            g = g_flat.reshape(g.shape)

        return g
    
def non_circular_g(x, sigma, u):
    return torch.exp(-((x-u)**2)/(2*sigma*sigma))

def circular_g(x, u, sigma):
    # x = np.atleast_1d(x)
    c = 1 / (2*torch.pi*torch.special.i0(sigma))
    return c * torch.exp(sigma * torch.cos(x-u))

def smooth_abs(x, eps=1e-6):
    return torch.sqrt(x**2 + eps)
def smooth_sign(x):
    s = 10
    return torch.tanh(x*s)
def smooth_min(a, b, beta=10.0):
    return (a * torch.exp(-beta * a) + b * torch.exp(-beta * b)) / (torch.exp(-beta * a) + torch.exp(-beta * b))
def smooth_max(a, b, beta=10.0):
    return (a * torch.exp(beta * a) + b * torch.exp(beta * b)) / (torch.exp(beta * a) + torch.exp(beta * b))
