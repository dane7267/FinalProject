import numpy as np
import scipy as sp
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def produce_basic_statistics(y, plag, Kfold):
    AM = calculate_AM(y)
    WC = calculate_WC_nonvectorised(y)
    BC = calculate_BC(y)
    CP = calculate_CP(WC, BC)
    AMS = calculate_AMS(y)
    AMA = calculate_AMA(y)
    CPPM = calculate_CPPM(y, Kfold)
    CPZP = calculate_CPZP(y, Kfold)
    ZPMINUSCP = calculate_ZPminusCP(CPPM, CPZP)

    return AM, CP, WC, BC, AMA, AMS, CPPM, CPZP, ZPMINUSCP

def calculate_AMS(y):
    n_subs, n, v = y.shape
    nBins = 6
    # Split into condition blocks
    cond1_p1 = y[:, :n // 4, :]            # (n_subs, n//4, v)
    cond1_p2 = y[:, n // 4:n // 2, :]
    cond2_p1 = y[:, n // 2:3 * n // 4, :]
    cond2_p2 = y[:, 3 * n // 4:, :]
    # Combine for t-tests: shape (n_subs, n//2, v)
    cond1_combined = np.concatenate([cond1_p1, cond1_p2], axis=1)
    cond2_combined = np.concatenate([cond2_p1, cond2_p2], axis=1)

    # Batched t-test function
    def batched_ttest(c1, c2):
        mean1 = np.mean(c1, axis=1)
        mean2 = np.mean(c2, axis=1)
        std1 = np.std(c1, axis=1, ddof=0)
        std2 = np.std(c2, axis=1, ddof=0)
        n1 = c1.shape[1]
        n2 = c2.shape[1]
        denom = np.sqrt((std1 ** 2) / n1 + (std2 ** 2) / n2)
        t_stat = (mean1 - mean2) / denom.clip(min=1e-6)  # avoid div by zero
        return t_stat  # (n_subs, v)
    
    # Get t-values per subject per voxel
    tval1 = batched_ttest(cond1_combined, cond2_combined)
    tval2 = batched_ttest(cond2_combined, cond1_combined)

    # Sort indices based on abs t-values (for each subject)
    tval_sorted_ind1 = np.argsort(np.abs(tval1), axis=1)
    tval_sorted_ind2 = np.argsort(np.abs(tval2), axis=1)
    
    # Compute means across trials (per condition block)
    c1_init = np.mean(cond1_p1,axis=1)# (n_subs, v)
    c1_rep = np.mean(cond1_p2, axis=1)
    c2_init = np.mean(cond2_p1,axis=1)# (n_subs, v)
    c2_rep = np.mean(cond2_p2, axis=1)

    # Reorder based on t-sorted indices
    gather1 = lambda data, idx: np.take_along_axis(data, idx, axis=1)
    c1_sinit = gather1(c1_init, tval_sorted_ind1)
    c1_srep = gather1(c1_rep, tval_sorted_ind1)
    c2_sinit = gather1(c2_init, tval_sorted_ind2)
    c2_srep = gather1(c2_rep, tval_sorted_ind2)

    # Compute trends
    abs_init_trend = (c1_sinit + c2_sinit) / 2
    abs_rep_trend = (c1_srep + c2_srep) / 2
    AS = abs_init_trend - abs_rep_trend  # (n_subs, v)

    # Bin index logic (same across all subjects)
    ranks = np.arange(1, v + 1).astype(float)
    perc_inds = round_away_from_zero((ranks * (nBins - 1)) / v).astype(int)
    bin_mask = np.eye(nBins)[perc_inds].T.astype(float)

    # Expand for all subjects
    bin_mask = np.expand_dims(bin_mask, axis=0)  # (1, nBins, v)
    AS_exp = np.expand_dims(AS, axis=1)        # (n_subs, 1, v)

    # Bin and average
    numerators = (AS_exp * bin_mask).sum(axis=2)       # (n_subs, nBins)
    denominators = bin_mask.sum(axis=2).clip(min=1)   # (1, nBins)
    AMS = numerators / denominators                   # (n_subs, nBins)

    return AMS

def calculate_AMA(y):
    nBins = 6
    n_subs, n, v = y.shape
    cond1_p1 = y[:, :n // 4, :v]
    cond1_p2 = y[:, n // 4:n // 2, :v]
    cond2_p1 = y[:, n // 2:3 * n // 4, :v]
    cond2_p2 = y[:, 3 * n // 4:, :v]

    cond1r1 = np.mean(cond1_p1, axis=1)
    cond2r1 = np.mean(cond2_p1, axis=1)
    cond1r2 = np.mean(cond1_p2, axis=1)
    cond2r2 = np.mean(cond2_p2, axis=1)

    mean_vox_cond1 = np.mean(np.concatenate([cond1_p1, cond1_p2], axis=1), axis=1) #Not transposing but also not changing dimensions from original.
    mean_vox_cond2 = np.mean(np.concatenate([cond2_p1, cond2_p2], axis=1), axis=1)
    # sAmp1r, ind1 = np.sort(mean_vox_cond1, axis=1)
    # sAmp2r, ind2 = np.sort(mean_vox_cond2, axis=1)

    ind1 = np.argsort(mean_vox_cond1, axis=1)
    ind2 = np.argsort(mean_vox_cond2, axis=1)
    

    # Sort the voxel-averaged vectors using gather
    sAmp1r1 = np.take_along_axis(cond1r1, ind1, axis=1)
    sAmp1r2 = np.take_along_axis(cond1r2, ind1, axis=1)
    sAmp2r1 = np.take_along_axis(cond2r1, ind2, axis=1)
    sAmp2r2 = np.take_along_axis(cond2r2, ind2, axis=1)

    # Compute slope across conditions, then average
    sAmp = ((sAmp1r1 - sAmp1r2) + (sAmp2r1 - sAmp2r2)) / 2  # (n_subs, v)
    # Compute bin indices (shared across subjects)
    ranks = np.arange(1, v + 1).astype(float)
    percInds = round_away_from_zero((ranks * (nBins - 1)) / v).astype(int)
    AMA = np.zeros((n_subs, nBins), dtype=np.float32)

    # One-hot encode the bin indices
    one_hot = np.eye(nBins)[percInds]  # shape: (v, nBins)
    bin_mask = one_hot.T[np.newaxis, :, :]  # shape: (1, nBins, v)

    # Expand sAmp to shape (n_subs, 1, v)
    sAmp_exp = sAmp[:, np.newaxis, :]  # shape: (n_subs, 1, v)

    #Multiply and sum across voxels
    numerators = (sAmp_exp * bin_mask).sum(axis=2) #(n_subs, nBins)
    denominators = bin_mask.sum(axis=2).clip(min=1) #(1, nBins)
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
    cond1r1 = np.mean(cond1_p1, axis=1)
    cond2r1 = np.mean(cond2_p1, axis=1)
    cond1r2 = np.mean(cond1_p2, axis=1)
    cond2r2 = np.mean(cond2_p2, axis=1)
    # print(cond1r1.shape)
    avgAmp1r1 = np.mean(cond1r1, axis=1)
    avgAmp2r1 = np.mean(cond2r1, axis=1)
    avgAmp1r2 = np.mean(cond1r2, axis=1)
    avgAmp2r2 = np.mean(cond2r2, axis=1)

    # avgR1 = ((avgAmp1r1+avgAmp2r1)/2).unsqueeze(1)
    avgR1 = ((avgAmp1r1 + avgAmp2r1) / 2)[:, np.newaxis]
    avgR2 = ((avgAmp1r2+avgAmp2r2)/2)[:, np.newaxis]
    return np.column_stack((avgR1, avgR2))


def calculate_WC_nonvectorised(y):
    n_subs, n, v = y.shape
    wtc1 = np.zeros((n_subs, 1))
    wtc2 = np.zeros((n_subs, 1))
    for sub in range(n_subs):
        pattern = y[sub]
        cond1_p1 = pattern[:n//4, :v]
        cond2_p1 = pattern[n//2:3*n//4, :v]
        cond1_p2 = pattern[n // 4:n // 2, :v]
        cond2_p2 = pattern[3 * n // 4:, :v]

        cond1_p1_corr = (pytorch_corr(cond1_p1.T, None, rowvar=False) + pytorch_corr(cond2_p1.T, None, rowvar=False)) / 2
        wtc1[sub] = np.mean(np.mean(cond1_p1_corr, axis=0))
        cond1_p2_corr = (pytorch_corr(cond1_p2.T, None, rowvar=False) + pytorch_corr(cond2_p2.T, None, rowvar=False)) / 2
        wtc2[sub] = np.mean(np.mean(cond1_p2_corr, axis=0))
    return np.column_stack((wtc1, wtc2))


def calculate_BC(y):
    n_subs, n, v = y.shape
    btc1 = np.zeros((n_subs, 1))
    btc2 = np.zeros((n_subs, 1))
    for sub in range(n_subs):
        pattern = y[sub]
        cond1_p1 = pattern[:n//4, :v]
        cond2_p1 = pattern[n//2:3*n//4, :v]

        pp1 = pytorch_corr(cond1_p1.T, cond2_p1.T, rowvar=False)
        pp11 = (cond1_p1.T).shape[1]
        ppx = pp1[:pp11, pp11:]
        # print(torch.mean(torch.mean(ppx, axis=0)))
        btc1[sub] = np.mean(np.mean(ppx, axis=0))
        
        cond1_p2 = pattern[n // 4:n // 2, :v]
        cond2_p2 = pattern[3 * n // 4:, :v]
        pp2 = pytorch_corr(cond1_p2.T, cond2_p2.T, rowvar=False)
        pp22 = (cond1_p2.T).shape[1]
        ppx2 = pp2[:pp22, pp22:]
        # print(torch.mean(torch.mean(ppx2, axis=0)))
        btc2[sub] = np.mean(np.mean(ppx2, axis=0))

    return np.column_stack((btc1, btc2))

def calculate_CPPM(y, Kfold):
    #y is 18 x 32 x 200
    n_subs, n, v = y.shape
    cp1_array = np.zeros((n_subs, 1))
    for sub in range(n_subs):
        pattern = y[sub]
        cond1_p1 = pattern[:n//4, :v]
        cond1_p2 = pattern[n // 4:n // 2, :v]
        F = np.vstack([cond1_p1, cond1_p2])
        Y = np.concatenate([
            np.zeros((cond1_p1.shape[0], 1)),
            np.ones((cond1_p2.shape[0], 1))
        ])
        Fm = np.mean(F, axis=1)
        svm_model = SVC(kernel='linear')
        scores = cross_val_score(svm_model, Fm.reshape(-1, 1), Y.ravel(), cv=Kfold)
        mean_score = np.mean(scores)
        
        cp1_array[sub] = mean_score
    # print(cp1_array)
    return cp1_array

def calculate_CPZP(y, Kfold):
    #y is 18 x 32 x 200
    n_subs, n, v = y.shape
    cp1_array = np.zeros((n_subs, 1))
    for sub in range(n_subs):
        pattern = y[sub]
        cond1_p1 = pattern[:n//4, :v]
        cond1_p2 = pattern[n // 4:n // 2, :v]
        F = np.vstack([cond1_p1, cond1_p2])
        Y = np.concatenate([
            np.zeros((cond1_p1.shape[0], 1)),
            np.ones((cond1_p2.shape[0], 1))
        ])
        Fz = sp.stats.zscore(F, axis=1)
        svm_z = SVC(kernel='linear')
        cv_scores_z = cross_val_score(svm_z, Fz, Y.ravel(), cv=Kfold)
        cp1_array[sub] = np.mean(cv_scores_z)
    # print(cp1_array)
    return cp1_array

def calculate_ZPminusCP(CPPM, CPZP):
    return CPZP - CPPM

def pytorch_corr(x, y, rowvar):
    if rowvar == False:
        if y is not None:
            x = np.concatenate((x, y), axis=1)
        x = x.T
        # mean = x.mean(dim=1, keepdim=True)
        mean = np.mean(x, axis=1, keepdims=True)
        x_centered = x - mean
        n = x.shape[1]
        cov = (x_centered @ x_centered.T) / (n-1)



        stddev = np.sqrt(np.diag(cov))
        outer_stddev = stddev[:, None] * stddev[None, :]
        corr = cov / outer_stddev
        corr = np.clip(corr, -1.0, 1.0)
        return corr

def calculate_CP(WC, BC):
    return WC - BC
    
def produce_slopes(y, pflag, Kfold):
    
    AM, CP, WC, BC, AMA, AMS, CPPM, CPZP, ZPMINUSCP = produce_basic_statistics(y, pflag, Kfold)

    def compute_slope(data):
        L = data.shape[1]
        X = np.vstack((np.arange(1, L+1), np.ones(L))).T
        pX = np.linalg.pinv(X)
        return np.matmul(pX, data.T)[0]

    slopes = np.stack([np.mean(compute_slope(data)) for data in [AM, WC, BC, CP, AMS, AMA]
                          ])
    return slopes, np.mean(CPPM), np.mean(CPZP), np.mean(ZPMINUSCP)

def round_away_from_zero(x):
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))
