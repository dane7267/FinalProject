import numpy as np
import scipy as sp
import torch

def produce_basic_statistics(y, plag):
    AM = calculate_AM(y)
    WC = calculate_WC_nonvectorised(y)
    BC = calculate_BC(y)
    CP = calculate_CP(WC, BC)
    AMS = calculate_AMS(y)
    AMA = calculate_AMA(y)

    return AM, WC, BC, CP, AMS, AMA

def calculate_AMS(y):
    n_subs, n, v = y.shape
    nBins = 6
    # Split into condition blocks
    cond1_p1 = y[:, :n // 4, :]            # (n_subs, n//4, v)
    cond1_p2 = y[:, n // 4:n // 2, :]
    cond2_p1 = y[:, n // 2:3 * n // 4, :]
    cond2_p2 = y[:, 3 * n // 4:, :]
    # Combine for t-tests: shape (n_subs, n//2, v)
    cond1_combined = torch.cat([cond1_p1, cond1_p2], dim=1)
    cond2_combined = torch.cat([cond2_p1, cond2_p2], dim=1)

    # Batched t-test function
    def batched_ttest(c1, c2):
        mean1 = c1.mean(dim=1)
        mean2 = c2.mean(dim=1)
        std1 = c1.std(dim=1, unbiased=False)
        std2 = c2.std(dim=1, unbiased=False)
        n1 = c1.shape[1]
        n2 = c2.shape[1]
        denom = torch.sqrt((std1 ** 2) / n1 + (std2 ** 2) / n2)
        t_stat = (mean1 - mean2) / denom.clamp(min=1e-6)  # avoid div by zero
        return t_stat  # (n_subs, v)
    
    # Get t-values per subject per voxel
    tval1 = batched_ttest(cond1_combined, cond2_combined)
    tval2 = batched_ttest(cond2_combined, cond1_combined)

    # Sort indices based on abs t-values (for each subject)
    tval_sorted_ind1 = torch.argsort(torch.abs(tval1), dim=1)
    tval_sorted_ind2 = torch.argsort(torch.abs(tval2), dim=1)
    
    # Compute means across trials (per condition block)
    c1_init = cond1_p1.mean(dim=1)  # (n_subs, v)
    c1_rep = cond1_p2.mean(dim=1)
    c2_init = cond2_p1.mean(dim=1)
    c2_rep = cond2_p2.mean(dim=1)

    # Reorder based on t-sorted indices
    gather1 = lambda data, idx: torch.gather(data, 1, idx)
    c1_sinit = gather1(c1_init, tval_sorted_ind1)
    c1_srep = gather1(c1_rep, tval_sorted_ind1)
    c2_sinit = gather1(c2_init, tval_sorted_ind2)
    c2_srep = gather1(c2_rep, tval_sorted_ind2)

    # Compute trends
    abs_init_trend = (c1_sinit + c2_sinit) / 2
    abs_rep_trend = (c1_srep + c2_srep) / 2
    AS = abs_init_trend - abs_rep_trend  # (n_subs, v)

    # Bin index logic (same across all subjects)
    ranks = torch.arange(1, v + 1, device=y.device).float()
    perc_inds = round_away_from_zero(((ranks * (nBins - 1)) / v)).long()  # (v,)
    bin_mask = torch.nn.functional.one_hot(perc_inds, num_classes=nBins).T.float()  # (nBins, v)

    # Expand for all subjects
    bin_mask = bin_mask.unsqueeze(0)  # (1, nBins, v)
    AS_exp = AS.unsqueeze(1)          # (n_subs, 1, v)

    # Bin and average
    numerators = (AS_exp * bin_mask).sum(dim=2)       # (n_subs, nBins)
    denominators = bin_mask.sum(dim=2).clamp(min=1)   # (1, nBins)
    AMS = numerators / denominators                   # (n_subs, nBins)

    return AMS

def calculate_AMA(y):
    nBins = 6
    n_subs, n, v = y.shape
    cond1_p1 = y[:, :n // 4, :v]
    cond1_p2 = y[:, n // 4:n // 2, :v]
    cond2_p1 = y[:, n // 2:3 * n // 4, :v]
    cond2_p2 = y[:, 3 * n // 4:, :v]

    cond1r1 = torch.mean(cond1_p1, axis=1)
    cond2r1 = torch.mean(cond2_p1, axis=1)
    cond1r2 = torch.mean(cond1_p2, axis=1)
    cond2r2 = torch.mean(cond2_p2, axis=1)

    mean_vox_cond1 = torch.cat([cond1_p1, cond1_p2], dim=1).mean(dim=1) #Not transposing but also not changing dimensions from original.
    mean_vox_cond2 = torch.cat([cond2_p1, cond2_p2], dim=1).mean(dim=1)
    sAmp1r, ind1 = torch.sort(mean_vox_cond1, dim=1)
    sAmp2r, ind2 = torch.sort(mean_vox_cond2, dim=1)

    # Sort the voxel-averaged vectors using gather
    sAmp1r1 = torch.gather(cond1r1, 1, ind1)
    sAmp1r2 = torch.gather(cond1r2, 1, ind1)
    sAmp2r1 = torch.gather(cond2r1, 1, ind2)
    sAmp2r2 = torch.gather(cond2r2, 1, ind2)

    # Compute slope across conditions, then average
    sAmp = ((sAmp1r1 - sAmp1r2) + (sAmp2r1 - sAmp2r2)) / 2  # (n_subs, v)
    # Compute bin indices (shared across subjects)
    ranks = torch.arange(1, v + 1, device=y.device).float()
    percInds = round_away_from_zero(((ranks * (nBins - 1)) / v)).long()  # (v,)
    # percInds = (torch.round((torch.arange(1, v + 1) * (nBins - 1)) / v) / (nBins - 1)) * (nBins - 1)
    AMA = torch.zeros((n_subs, nBins), dtype=torch.float32, device=y.device)

    one_hot = torch.nn.functional.one_hot(percInds, num_classes=nBins).float() #(v, nBins)
    bin_mask = one_hot.T.unsqueeze(0) #(1, nBins, v)
    sAmp_exp = sAmp.unsqueeze(1) #(n_subs, 1, v)

    #Multiply and sum across voxels
    numerators = (sAmp_exp * bin_mask).sum(dim=2) #(n_subs, nBins)
    denominators = bin_mask.sum(dim=2).clamp(min=1) #(1, nBins)
    AMA = numerators / denominators
    return AMA

def calculate_AM(y):
    v = y.shape[2]
    n = y.shape[1] #y has shape 18 x 32 x 200
    subs = y.shape[0]
    cond1_p1 = y[:,:n // 4, :v]
    cond1_p2 = y[:, n // 4:n // 2, :v]
    cond2_p1 = y[:, n // 2:3 * n // 4, :v]
    cond2_p2 = y[:, 3 * n // 4:, :v]
    cond1r1 = torch.mean(cond1_p1, axis=1)
    cond2r1 = torch.mean(cond2_p1, axis=1)
    cond1r2 = torch.mean(cond1_p2, axis=1)
    cond2r2 = torch.mean(cond2_p2, axis=1)
    # print(cond1r1.shape)
    avgAmp1r1 = torch.mean(cond1r1, axis=1)
    avgAmp2r1 = torch.mean(cond2r1, axis=1)
    avgAmp1r2 = torch.mean(cond1r2, axis=1)
    avgAmp2r2 = torch.mean(cond2r2, axis=1)

    avgR1 = ((avgAmp1r1+avgAmp2r1)/2).unsqueeze(1)
    avgR2 = ((avgAmp1r2+avgAmp2r2)/2).unsqueeze(1)
    return torch.column_stack((avgR1, avgR2))

# def calculate_WC(y):
#     n_subs = y.shape[0]
#     n = y.shape[1]
#     v = y.shape[2]
#     cond1_p1 = y[:,:n // 4, :v]
#     cond2_p1 = y[:, n // 2:3 * n // 4, :v]

#     cond1_p1_centered = cond1_p1 - cond1_p1.mean(dim=1, keepdim=True)
#     cond2_p1_centered = cond2_p1 - cond2_p1.mean(dim=1, keepdim=True)

#     def batched_corr(x):
#         x_t = x.transpose(1,2)
#         cov = torch.matmul(x_t, x_t.transpose(1,2)) / (x.shape[1] -1)
#         std = x_t.std(dim=2)
#         return cov / (std[:,:,None] * std[:, None, :]) + 1e-7
        
#     corr1_p1 = batched_corr(cond1_p1_centered)
#     corr2_p1 = batched_corr(cond2_p1_centered)
#     avgcorr1 = (corr1_p1 + corr2_p1) / 2
#     wtc1 = avgcorr1.mean(dim=(1,2)).unsqueeze(1)

#     cond1_p2 = y[:, n // 4:n // 2, :v]
#     cond2_p2 = y[:, 3 * n // 4:, :v]

#     cond1_p2_centered = cond1_p2 - cond1_p2.mean(dim=1, keepdim=True)
#     cond2_p2_centered = cond2_p2 - cond2_p2.mean(dim=1, keepdim=True)

#     corr1_p2 = batched_corr(cond1_p2_centered)
#     corr2_p2 = batched_corr(cond2_p2_centered)


#     avgcorr2 = (corr1_p2 + corr2_p2) / 2
#     wtc2 = avgcorr2.mean(dim=(1, 2)).unsqueeze(1)

#     return torch.column_stack((wtc1, wtc2))

def calculate_WC_nonvectorised(y):
    n_subs, n, v = y.shape
    wtc1 = torch.zeros((n_subs, 1))
    wtc2 = torch.zeros((n_subs, 1))
    for sub in range(n_subs):
        pattern = y[sub]
        cond1_p1 = pattern[:n//4, :v]
        cond2_p1 = pattern[n//2:3*n//4, :v]
        cond1_p2 = pattern[n // 4:n // 2, :v]
        cond2_p2 = pattern[3 * n // 4:, :v]

        cond1_p1_corr = (pytorch_corr(cond1_p1.T, None, rowvar=False) + pytorch_corr(cond2_p1.T, None, rowvar=False)) / 2
        wtc1[sub] = torch.mean(torch.mean(cond1_p1_corr, axis=0))
        cond1_p2_corr = (pytorch_corr(cond1_p2.T, None, rowvar=False) + pytorch_corr(cond2_p2.T, None, rowvar=False)) / 2
        wtc2[sub] = torch.mean(torch.mean(cond1_p2_corr, axis=0))
    return torch.column_stack((wtc1, wtc2))


def calculate_BC(y):
    n_subs, n, v = y.shape
    btc1 = torch.zeros((n_subs, 1))
    btc2 = torch.zeros((n_subs, 1))
    for sub in range(n_subs):
        pattern = y[sub]
        cond1_p1 = pattern[:n//4, :v]
        cond2_p1 = pattern[n//2:3*n//4, :v]

        pp1 = pytorch_corr(cond1_p1.T, cond2_p1.T, rowvar=False)
        pp11 = (cond1_p1.T).shape[1]
        ppx = pp1[:pp11, pp11:]
        # print(torch.mean(torch.mean(ppx, axis=0)))
        btc1[sub] = torch.mean(torch.mean(ppx, axis=0))
        
        cond1_p2 = pattern[n // 4:n // 2, :v]
        cond2_p2 = pattern[3 * n // 4:, :v]
        pp2 = pytorch_corr(cond1_p2.T, cond2_p2.T, rowvar=False)
        pp22 = (cond1_p2.T).shape[1]
        ppx2 = pp2[:pp22, pp22:]
        # print(torch.mean(torch.mean(ppx2, axis=0)))
        btc2[sub] = torch.mean(torch.mean(ppx2, axis=0))

    return torch.column_stack((btc1, btc2))

def pytorch_corr(x, y, rowvar):
    if rowvar == False:
        if y is not None:
            x = torch.cat((x, y), dim=1)
        x = x.T
        mean = x.mean(dim=1, keepdim=True)
        x_centered = x - mean
        n = x.shape[1]
        cov = (x_centered @ x_centered.T) / (n-1)



        stddev = torch.sqrt(torch.diag(cov))
        outer_stddev = stddev[:, None] * stddev[None, :]
        corr = cov / outer_stddev
        corr = torch.clamp(corr, -1.0, 1.0)
        return corr

def calculate_CP(WC, BC):
    return WC - BC
    
def produce_confidence_interval(y, pflag):
    AM, WC, BC, CP, AMS, AMA = produce_basic_statistics(y, pflag)

    def compute_slope(data):
        data=data.float()
        L = data.shape[1]
        X = torch.vstack((torch.arange(1, L+1), torch.ones(L))).T
        pX = torch.linalg.pinv(X)
        return torch.matmul(pX, data.T)[0]

    slopes = torch.stack([torch.mean(compute_slope(data)) for data in [AM, WC, BC, CP, AMS, AMA]
                          ])
    return slopes

def round_away_from_zero(x):
        return torch.where(x >= 0, (x + 0.5).floor(), (x - 0.5).ceil())

# y = torch.tensor([[[-0.9969, -1.3622],
#          [ 0.7768, -1.1933],
#          [ 0.8446, -0.3450],
#          [ 0.1953, -0.0743],
#          [-0.8318,  1.1681],
#          [-0.3868,  2.1993],
#          [ 2.4177, -0.5848],
#          [ 0.3025,  1.4126]],

#         [[-0.5642, -1.4456],
#          [-0.8783,  1.2357],
#          [ 0.1728, -0.6139],
#          [ 0.6109,  0.1852],
#          [ 0.7538, -1.7449],
#          [ 1.7639,  0.8480],
#          [ 0.6495, -2.4907],
#          [-1.4072, -0.2315]]])

# print(produce_basic_statistics(y, 1))