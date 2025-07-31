import numpy as np
import scipy as sp
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
import torch
import torch.nn as nn


def produce_basic_statistics(y, plag, Kfold):
    AM = calculate_AM(y)
    WC = calculate_WC_nonvectorised(y)
    BC = calculate_BC(y)
    CP = calculate_CP(WC, BC)
    AMS = calculate_AMS(y)
    AMA = calculate_AMA(y)
    CPPM = calculate_CPPM_logistic(y, Kfold = 5)
    CPZP = calculate_CPZP_logistic(y, Kfold = 5)
    ZPMINUSCP = calculate_ZPminusCP(CPPM, CPZP)
    print(ZPMINUSCP)

    return AM, CP, WC, BC, AMA, AMS, CPPM, CPZP, ZPMINUSCP

def calculate_AMS(y):
    nBins = 6
    abs_ad_trend = np.zeros((len(y), nBins))

    for sub in range(len(y)):
        pattern = y[sub]
        v = pattern.shape[1] # voxels
        n = pattern.shape[0] # trials


        if n % 4 != 0:
            raise ValueError("Assumes 4 conditions with equal trials")
        
        cond1_p = [
            pattern[:n // 4, :v],               # Rows 1 to n/4
            pattern[n // 4:n // 2, :v],         # Rows n/4 + 1 to n/2
        ]
        cond2_p = [
            pattern[n // 2:3 * n // 4, :v],     # Rows n/2 + 1 to 3n/4
            pattern[3 * n // 4:, :v],           # Rows 3n/4 + 1 to n
        ]

        # Perform t-tests
        tval1, pval1 = sp.stats.ttest_ind(np.vstack([cond1_p[0], cond1_p[1]]), np.vstack([cond2_p[0], cond2_p[1]]), axis=0)
        tval2, pval2 = sp.stats.ttest_ind(np.vstack([cond2_p[0], cond2_p[1]]), np.vstack([cond1_p[0], cond1_p[1]]), axis=0)

        # Sorting the t-values by their absolute values
        tval_sorted_ind1 = np.argsort(np.abs(tval1))
        tval_sorted_ind2 = np.argsort(np.abs(tval2))

        # Compute means for conditions
        c1_init = np.mean(cond1_p[0], axis=0)
        c1_rep = np.mean(cond1_p[1], axis=0)
        c2_init = np.mean(cond2_p[0], axis=0)
        c2_rep = np.mean(cond2_p[1], axis=0)

        # Reorder based on sorted indices
        c1_sinit = c1_init[tval_sorted_ind1]
        c1_srep = c1_rep[tval_sorted_ind1]
        c2_sinit = c2_init[tval_sorted_ind2]
        c2_srep = c2_rep[tval_sorted_ind2]

        # Compute trends
        abs_init_trend = (c1_sinit + c2_sinit) / 2
        abs_rep_trend = (c1_srep + c2_srep) / 2
        abs_adaptation_trend = abs_init_trend - abs_rep_trend

        AS = abs_adaptation_trend
        AS1=abs_init_trend
        AS2=abs_rep_trend

        # Compute the percentage indices (similar to MATLAB's rounding and indexing)
        #percInds = (np.round((np.arange(1, len(AA) + 1) * (nBins - 1)) / len(AA)) / (nBins - 1)) * (nBins - 1) + 1
        percInds = (np.round((np.arange(1, len(AS) + 1) * (nBins - 1)) / len(AS)) / (nBins - 1)) * (nBins - 1)



        for i in range(nBins):
            abs_ad_trend[sub, i] = np.mean(AS[percInds == i], axis=0)

    AMS = abs_ad_trend
    return AMS

def calculate_AMA(y):
    nBins = 6
    sc_trend = np.zeros((len(y), nBins))

    for sub in range(len(y)):
        pattern = y[sub]
        v = pattern.shape[1] # voxels
        n = pattern.shape[0] # trials


        if n % 4 != 0:
            raise ValueError("Assumes 4 conditions with equal trials")
        
        cond1_p = [
            pattern[:n // 4, :v],               # Rows 1 to n/4
            pattern[n // 4:n // 2, :v],         # Rows n/4 + 1 to n/2
        ]
        cond2_p = [
            pattern[n // 2:3 * n // 4, :v],     # Rows n/2 + 1 to 3n/4
            pattern[3 * n // 4:, :v],           # Rows 3n/4 + 1 to n
        ]

        # Compute means
        cond1r1 = np.mean(cond1_p[0], axis=0)
        cond2r1 = np.mean(cond2_p[0], axis=0)
        cond1r2 = np.mean(cond1_p[1], axis=0)
        cond2r2 = np.mean(cond2_p[1], axis=0)



        sAmp1r = np.sort(np.mean(np.hstack([cond1_p[0].T, cond1_p[1].T]), axis=1))
        ind1 = np.argsort(np.mean(np.hstack([cond1_p[0].T, cond1_p[1].T]), axis=1))

        sAmp2r = np.sort(np.mean(np.hstack([cond2_p[0].T, cond2_p[1].T]), axis=1))
        ind2 = np.argsort(np.mean(np.hstack([cond2_p[0].T, cond2_p[1].T]), axis=1))

        # Reorder based on indices
        sAmp1r1 = cond1r1[ind1]
        sAmp2r1 = cond2r1[ind2]
        sAmp1r2 = cond1r2[ind1]
        sAmp2r2 = cond2r2[ind2]

        # Compute slope
        sAmp = ((sAmp1r1 - sAmp1r2) + (sAmp2r1 - sAmp2r2)) / 2

        AA = sAmp
        # Compute the percentage indices (similar to MATLAB's rounding and indexing)
        #percInds = (np.round((np.arange(1, len(AA) + 1) * (nBins - 1)) / len(AA)) / (nBins - 1)) * (nBins - 1) + 1
        percInds = (np.round((np.arange(1, len(AA) + 1) * (nBins - 1)) / len(AA)) / (nBins - 1)) * (nBins - 1)



        for i in range(nBins):
            sc_trend[sub, i] = np.mean(AA[percInds == i], axis=0)

    AMA = sc_trend

    return AMA

def calculate_AM(y):
    avgAmp1r1 = np.zeros((len(y), 1))
    avgAmp2r1 = np.zeros((len(y), 1))
    avgAmp1r2 = np.zeros((len(y), 1))
    avgAmp2r2 = np.zeros((len(y), 1))

    nBins = 6
    for sub in range(len(y)):
        pattern = y[sub]
        v = pattern.shape[1] # voxels
        n = pattern.shape[0] # trials


        if n % 4 != 0:
            raise ValueError("Assumes 4 conditions with equal trials")
        
        cond1_p = [
            pattern[:n // 4, :v],               # Rows 1 to n/4
            pattern[n // 4:n // 2, :v],         # Rows n/4 + 1 to n/2
        ]
        cond2_p = [
            pattern[n // 2:3 * n // 4, :v],     # Rows n/2 + 1 to 3n/4
            pattern[3 * n // 4:, :v],           # Rows 3n/4 + 1 to n
        ]


        # Compute means
        cond1r1 = np.mean(cond1_p[0], axis=0)
        cond2r1 = np.mean(cond2_p[0], axis=0)
        cond1r2 = np.mean(cond1_p[1], axis=0)
        cond2r2 = np.mean(cond2_p[1], axis=0)


        """Analysing the 6 adaptation criteria in the voxel pattern"""

        """MAM average repetition effect"""
        #print(np.mean(cond1r1))
        avgAmp1r1[sub, :] = np.mean(cond1r1, axis=0)
        #print("avgAmp1r1")
        #print(avgAmp1r1)
        #This is very far off
        avgAmp2r1[sub, :] = np.mean(cond2r1, axis=0)
        avgAmp1r2[sub, :] = np.mean(cond1r2, axis=0)
        avgAmp2r2[sub, :] = np.mean(cond2r2, axis=0)
        avgR1 = (avgAmp1r1+avgAmp2r1)/2
        avgR2 = (avgAmp1r2+avgAmp2r2)/2
    AM = np.column_stack((avgR1, avgR2))
    return AM

def calculate_WC_nonvectorised(y):
    
    wtc1 = np.zeros((len(y), 1))
    wtc2 = np.zeros((len(y), 1))


    for sub in range(len(y)):
       
        pattern = y[sub]
        v = pattern.shape[1] # voxels
        n = pattern.shape[0] # trials
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

    btc1 = np.zeros((len(y), 1))
    btc2 = np.zeros((len(y), 1))
    for sub in range(len(y)):
        pattern = y[sub]
        v = pattern.shape[1] # voxels
        n = pattern.shape[0] # trials

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

# def assign_runs(n_trials, n_runs=9):
#     base_trials = n_trials // n_runs
#     extra = n_trials % n_runs
#     run_lengths = [base_trials + (1 if i < extra else 0) for i in range(n_runs)]
#     run_labels = np.concatenate([[i] * r for i, r in enumerate(run_lengths)])
#     return run_labels

def calculate_CPPM(y, Kfold):
    cp1_array = np.zeros((len(y), 1))
    for sub in range(len(y)):
        pattern = y[sub]
        v = pattern.shape[1] # voxels
        n = pattern.shape[0] # trials
        # run_labels = assign_runs(pattern.shape[0])
        # logo = LeaveOneGroupOut()    

        
        # n_runs = 4
        # trials_per_run = n // n_runs
        # run_labels = np.repeat(np.arange(n_runs), trials_per_run)


        if n % 4 != 0:
            raise ValueError("Assumes 4 conditions with equal trials")
        
        cond1_p = [
            pattern[:n // 4, :v],               # Rows 1 to n/4
            pattern[n // 4:n // 2, :v],         # Rows n/4 + 1 to n/2
        ]
        cond1_p1 = cond1_p[0]
        cond1_p2 = cond1_p[1]
        cond2_p = [
            pattern[n // 2:3 * n // 4, :v],     # Rows n/2 + 1 to 3n/4
            pattern[3 * n // 4:, :v],           # Rows 3n/4 + 1 to n
        ]
        cond2_p1 = cond2_p[0]
        cond2_p2 = cond2_p[1]
        F = np.vstack([cond1_p1, cond1_p2])
        Y = np.concatenate([
            np.zeros((cond1_p1.shape[0], 1)),
            np.ones((cond1_p2.shape[0], 1))
        ])
        # F = np.vstack([cond2_p1, cond2_p2])
        # Y = np.concatenate([
        #     np.zeros((cond2_p1.shape[0], 1)),
        #     np.ones((cond2_p2.shape[0], 1))
        # ])

        # run_labels = assign_runs(F.shape[0], n_runs=6)

        Fm = np.mean(F, axis=1)
        svm_model = SVC(kernel='linear')
        # logo = LeaveOneGroupOut()
        scores = cross_val_score(svm_model, Fm.reshape(-1, 1), Y.ravel(), cv=Kfold)
        # scores = cross_val_score(svm_model, Fm.reshape(-1, 1), Y.ravel(), cv=logo, groups=run_labels)
        mean_score = np.mean(scores)
        
        cp1_array[sub] = mean_score
    return cp1_array

def calculate_CPPM(y, Kfold):
    n_subs, n, v = y.shape
    cp1_array = torch.zeros((n_subs, 1))
    for sub in range(n_subs):
        pattern = y[sub]
        cond1_p1 = pattern[:n//4]
        cond1_p2 = pattern[n//4:n//2]
        F = torch.cat([cond1_p1, cond1_p2], dim=0)
        Y = torch.cat([
            torch.zeros((cond1_p1.shape[0], 1), dtype=torch.float32),
            torch.ones((cond1_p2.shape[0], 1), dtype=torch.float32)
        ])
        Fm = F.mean(dim=1, keepdim=True)

        num_samples = Fm.shape[0]
        fold_size = num_samples // Kfold
        indices = torch.randperm(num_samples)
        accs = []

        for k in range(Kfold):
            val_idx = indices[k * fold_size: (k + 1) * fold_size]
            train_idx = torch.cat([indices[:k * fold_size], indices[(k + 1) * fold_size:]])

            X_train, y_train = Fm[train_idx], Y[train_idx]
            X_val, y_val = Fm[val_idx], Y[val_idx]

            # Closed-form least squares with bias
            X_design = torch.cat([X_train, torch.ones_like(X_train)], dim=1)
            w, *_ = torch.linalg.lstsq(X_design, y_train)

            X_val_design = torch.cat([X_val, torch.ones_like(X_val)], dim=1)
            val_logits = X_val_design @ w
            val_preds = (val_logits > 0.5).float()

            acc = (val_preds == y_val).float().mean()
            accs.append(acc)

        cp1_array[sub, 0] = torch.stack(accs).mean()

    return cp1_array


def calculate_CPPM_logistic(y, Kfold):
    cp1_array = torch.zeros((len(y), 1))
    for sub in range(len(y)):
        pattern = y[sub]
        v = pattern.shape[1] # voxels
        n = pattern.shape[0] # trials
        cond1_p = [
            pattern[:n // 4, :v].tolist(),               # Rows 1 to n/4
            pattern[n // 4:n // 2, :v].tolist(),         # Rows n/4 + 1 to n/2
        ]
        cond1_p1 = torch.tensor(cond1_p[0])
        cond1_p2 = torch.tensor(cond1_p[1])
        cond2_p = [
            pattern[n // 2:3 * n // 4, :v].tolist(),     # Rows n/2 + 1 to 3n/4
            pattern[3 * n // 4:, :v].tolist(),           # Rows 3n/4 + 1 to n
        ]
        cond2_p1 = torch.tensor(cond2_p[0])
        cond2_p2 = torch.tensor(cond2_p[1])







        F = torch.cat([cond2_p1, cond2_p2], dim=0)  # shape: (n//2, v)
        Y = torch.cat([
            torch.zeros((cond2_p1.shape[0],), dtype=torch.float32, device=F.device),
            torch.ones((cond2_p2.shape[0],), dtype=torch.float32, device=F.device)
        ])  # shape: (n//2,)
        Fm = F.mean(dim=1, keepdim=True)

        num_samples = Fm.shape[0]
        fold_size = num_samples // Kfold
        indices = torch.randperm(num_samples)
        accs = []

        for k in range(Kfold):
            val_idx = indices[k * fold_size: (k + 1) * fold_size]
            train_idx = torch.cat([indices[:k * fold_size], indices[(k + 1) * fold_size:]])

            X_train, y_train = Fm[train_idx], Y[train_idx]
            X_val, y_val = Fm[val_idx], Y[val_idx]

            # Closed-form least squares with bias
            X_design = torch.cat([X_train, torch.ones_like(X_train)], dim=1)
            w, *_ = torch.linalg.lstsq(X_design, y_train)

            X_val_design = torch.cat([X_val, torch.ones_like(X_val)], dim=1)
            val_logits = X_val_design @ w
            val_preds = (val_logits > 0.5).float()

            acc = (val_preds == y_val).float().mean()
            accs.append(acc)

        cp1_array[sub, 0] = torch.stack(accs).mean()

    return cp1_array.numpy()  # shape: (n_subs, 1)

def calculate_CPZP_logistic(y, Kfold):
    cp1_array = torch.zeros((len(y), 1))
    for sub in range(len(y)):
        pattern = y[sub]
        v = pattern.shape[1] # voxels
        n = pattern.shape[0] # trials
        cond1_p = [
            pattern[:n // 4, :v].tolist(),               # Rows 1 to n/4
            pattern[n // 4:n // 2, :v].tolist(),         # Rows n/4 + 1 to n/2
        ]
        cond1_p1 = torch.tensor(cond1_p[0])
        cond1_p2 = torch.tensor(cond1_p[1])
        cond2_p = [
            pattern[n // 2:3 * n // 4, :v].tolist(),     # Rows n/2 + 1 to 3n/4
            pattern[3 * n // 4:, :v].tolist(),           # Rows 3n/4 + 1 to n
        ]
        cond2_p1 = torch.tensor(cond2_p[0])
        cond2_p2 = torch.tensor(cond2_p[1])


        F = torch.cat([cond2_p1, cond2_p2], dim=0)  # shape: (n//2, v)
        Y = torch.cat([
            torch.zeros((cond2_p1.shape[0],), dtype=torch.float32, device=F.device),
            torch.ones((cond2_p2.shape[0],), dtype=torch.float32, device=F.device)
        ])  # shape: (n//2,)

        # Fm = F.mean(dim=1, keepdim=True)  # shape: (n//2, 1)
        mean = F.mean(dim=1, keepdim=True)
        std = F.std(dim=1, keepdim=True) + 1e-6
        Fz = (F-mean)/std
        # K-fold cross-validation
        num_samples = Fz.shape[0]
        fold_size = num_samples // Kfold
        indices = torch.randperm(num_samples, device=F.device)

        accs = []
        for k in range(Kfold):
            val_idx = indices[k * fold_size: (k + 1) * fold_size]
            train_idx = torch.cat([indices[:k * fold_size], indices[(k + 1) * fold_size:]])

            X_train, y_train = Fz[train_idx], Y[train_idx]
            X_val, y_val = Fz[val_idx], Y[val_idx]

            model = nn.Linear(v, 1)
            """Why is this above different"""
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

            # Simple training loop (few steps only for cross-validation)
            for _ in range(20):
                optimizer.zero_grad()
                pred = model(X_train).squeeze()
                loss = nn.functional.binary_cross_entropy_with_logits(pred, y_train)
                loss.backward()
                optimizer.step()

            # Validation accuracy
            with torch.no_grad():
                val_logits = model(X_val).squeeze()
                val_preds = (val_logits > 0).float()
                acc = (val_preds == y_val).float().mean()
                accs.append(acc)

        cp1_array[sub, 0] = torch.stack(accs).mean()

    return cp1_array.numpy()  # shape: (n_subs, 1)



def calculate_CPPM_collapsed(y, Kfold):
    cp1_array = np.zeros((len(y), 1))
    for sub in range(len(y)):
        pattern = y[sub]
        v = pattern.shape[1] # voxels
        n = pattern.shape[0] # trials


        if n % 4 != 0:
            raise ValueError("Assumes 4 conditions with equal trials")
        
        cond1_p = [
            pattern[:n // 4, :v],               # Rows 1 to n/4
            pattern[n // 4:n // 2, :v],         # Rows n/4 + 1 to n/2
        ]
        cond1_p1 = cond1_p[0]
        cond1_p2 = cond1_p[1]

        cond2_p = [
            pattern[n // 2:3 * n // 4, :v],     # Rows n/2 + 1 to 3n/4
            pattern[3 * n // 4:, :v],           # Rows 3n/4 + 1 to n
        ]
        cond2_p1 = cond2_p[0]
        cond2_p2 = cond2_p[1]

        F = np.vstack([cond1_p1, cond2_p1, cond1_p2, cond2_p2])
        Y = np.concatenate([
            np.zeros((np.vstack([cond1_p1, cond2_p1]).shape[0], 1)),
            np.ones((np.vstack([cond1_p1, cond2_p1]).shape[0], 1))
        ])
        Fm = np.mean(F, axis=1)
        svm_model = SVC(kernel='linear')
        scores = cross_val_score(svm_model, Fm.reshape(-1, 1), Y.ravel(), cv=Kfold)
        mean_score = np.mean(scores)
        
        cp1_array[sub] = mean_score
    return cp1_array

def calculate_CPZP(y, Kfold):
    cp1_array = np.zeros((len(y), 1))
    for sub in range(len(y)):
        pattern = y[sub]
        v = pattern.shape[1] # voxels
        n = pattern.shape[0] # trials


        if n % 4 != 0:
            raise ValueError("Assumes 4 conditions with equal trials")
        
        cond1_p = [
            pattern[:n // 4, :v],               # Rows 1 to n/4
            pattern[n // 4:n // 2, :v],         # Rows n/4 + 1 to n/2
        ]
        cond1_p1 = cond1_p[0]
        cond1_p2 = cond1_p[1]
        cond2_p = [
            pattern[n // 2:3 * n // 4, :v],     # Rows n/2 + 1 to 3n/4
            pattern[3 * n // 4:, :v],           # Rows 3n/4 + 1 to n
        ]
        cond2_p1 = cond2_p[0]
        cond2_p2 = cond2_p[1]


        F = np.vstack([cond1_p1, cond1_p2])
        Y = np.concatenate([
            np.zeros((cond1_p1.shape[0], 1)),
            np.ones((cond1_p2.shape[0], 1))
        ])
        # F = np.vstack([cond2_p1, cond2_p2])
        # Y = np.concatenate([
        #     np.zeros((cond2_p1.shape[0], 1)),
        #     np.ones((cond2_p2.shape[0], 1))
        # ])
        Fz = sp.stats.zscore(F, axis=1)
        svm_z = SVC(kernel='linear')
        cv_scores_z = cross_val_score(svm_z, Fz, Y.ravel(), cv=Kfold)
        cp1_array[sub] = np.mean(cv_scores_z)
    return cp1_array

def calculate_CPZP_collapsed(y, Kfold):
    cp1_array = np.zeros((len(y), 1))
    for sub in range(len(y)):
        pattern = y[sub]
        v = pattern.shape[1] # voxels
        n = pattern.shape[0] # trials


        if n % 4 != 0:
            raise ValueError("Assumes 4 conditions with equal trials")
        
        cond1_p = [
            pattern[:n // 4, :v],               # Rows 1 to n/4
            pattern[n // 4:n // 2, :v],         # Rows n/4 + 1 to n/2
        ]
        cond1_p1 = cond1_p[0]
        cond1_p2 = cond1_p[1]

        cond2_p = [
            pattern[n // 2:3 * n // 4, :v],     # Rows n/2 + 1 to 3n/4
            pattern[3 * n // 4:, :v],           # Rows 3n/4 + 1 to n
        ]
        cond2_p1 = cond2_p[0]
        cond2_p2 = cond2_p[1]


        F = np.vstack([cond1_p1, cond2_p1, cond1_p2, cond2_p2])
        Y = np.concatenate([
            np.zeros((np.vstack([cond1_p1, cond2_p1]).shape[0], 1)),
            np.ones((np.vstack([cond1_p1, cond2_p1]).shape[0], 1))
        ])
        Fz = sp.stats.zscore(F, axis=1)
        svm_z = SVC(kernel='linear')
        cv_scores_z = cross_val_score(svm_z, Fz, Y.ravel(), cv=Kfold)
        cp1_array[sub] = np.mean(cv_scores_z)
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
