# pylint: disable=E1101
import torch
import numpy as np

def evaluate_model(model, P_tensor, P_static_tensor, P_avg_interval_tensor,
                      P_length_tensor, P_time_tensor, P_var_prior_emb_tensor,
                      batch_size=100, n_classes=2, static=None):
    model.eval()
    P_tensor = P_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    P_length_tensor = P_length_tensor.cuda()
    P_avg_interval_tensor = P_avg_interval_tensor.cuda()
    if P_static_tensor is None:
        Pstatic = None
    else:
        P_static_tensor = P_static_tensor.cuda()
        N, Fs = P_static_tensor.shape

    N, F, Ff = P_tensor.shape
    n_batches, rem = N // batch_size, N % batch_size
    out = torch.zeros(N, n_classes)
    start = 0
    for i in range(n_batches):
        P = P_tensor[start:start + batch_size]
        P_time = P_time_tensor[start:start + batch_size]
        P_length = P_length_tensor[start:start + batch_size]
        P_avg_interval = P_avg_interval_tensor[start:start + batch_size]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        middleoutput = model.forward(P, Pstatic, P_avg_interval, P_length, P_time,
                                     P_var_prior_emb_tensor)
        out[start:start + batch_size] = middleoutput.detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[start:start + rem]
        P_time = P_time_tensor[start:start + rem]
        P_length = P_length_tensor[start:start + rem]
        P_avg_interval = P_avg_interval_tensor[start:start + rem]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + rem]
        whatever = model.forward(P, Pstatic, P_avg_interval, P_length, P_time,
                                 P_var_prior_emb_tensor)
        out[start:start + rem] = whatever.detach().cpu()
    return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def getStats(P_tensor):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        if vals_f.size == 0:
            mf[f] = 0.0
        else:
            mf[f] = np.mean(vals_f)
        stdf[f] = np.std(vals_f)
        # stdf[f] = np.max([stdf[f], eps])
    return mf, stdf

def getStats_static(P_tensor, dataset='P12'):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))
    ms = np.zeros((S, 1))
    ss = np.ones((S, 1))

    if dataset == 'P12' or dataset == 'physionet':
        # ['Age' 'Gender=0' 'Gender=1' 'Height' 'ICUType=1' 'ICUType=2' 'ICUType=3' 'ICUType=4' 'Weight']
        bool_categorical = [0, 1, 1, 0, 1, 1, 1, 1, 0]
    elif dataset == 'P19':
        # ['Age' 'Gender' 'Unit1' 'Unit2' 'HospAdmTime' 'ICULOS']
        bool_categorical = [0, 1, 0, 0, 0, 0]

    for s in range(S):
        if bool_categorical[s] == 0:  # if not categorical
            vals_s = Ps[s, :]
            vals_s = vals_s[vals_s > 0]
            ms[s] = np.mean(vals_s)
            ss[s] = np.std(vals_s)
    return ms, ss

def tensorize_normalize_extract_feature(P, y, mf, stdf, ms, ss):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_time_tensor = np.zeros((len(P), T, F))
    P_mask_tensor = np.zeros((len(P), T, F))
    P_avg_interval_tensor = np.zeros((len(P), T, F))
    P_length_tensor = np.zeros([len(P), 1])
    P_static_tensor = np.zeros((len(P), D))
    max = 0
    for i in range(len(P)):
        P_tensor[i] = P[i]['arr']
        P[i]['time'] = P[i]['time'] / 60 if T == 60 else P[i]['time'] / 2880
        P_time_tensor[i] = P[i]['time']
        if np.max(P[i]['time'][1:] - P[i]['time'][:-1]) > max:
            max = np.max(P[i]['time'][1:] - P[i]['time'][:-1])
        P_mask_tensor[i] = P[i]['arr'] > 0
        P_static_tensor[i] = P[i]['extended_static']
        P_length_tensor[i] = P[i]['length']
        for j in range(F):
            idx_not_zero = np.where(P_mask_tensor[i][:, j])[0]

            if len(idx_not_zero) > 0:
                t = P[i]['time'][idx_not_zero]
                if len(idx_not_zero) == 1:
                    P_avg_interval_tensor[i][idx_not_zero, j] = P[i]['length'] / 2
                else:
                    right_interval = np.append(idx_not_zero[1:] - idx_not_zero[:-1], P[i]['length'] - idx_not_zero[-1])
                    left_interval = np.insert(idx_not_zero[1:] - idx_not_zero[:-1], 0, idx_not_zero[0])
                    P_avg_interval_tensor[i][idx_not_zero, j] = (left_interval + right_interval) / 2

    P_tensor = mask_normalize(P_tensor, mf, stdf)

    P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)

    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)

    return torch.FloatTensor(P_tensor), torch.FloatTensor(P_static_tensor), \
        torch.FloatTensor(P_avg_interval_tensor), torch.FloatTensor(P_length_tensor), torch.Tensor(P_time_tensor), y_tensor

def tensorize_normalize_exact_feature_mimic3(P, y, mf, stdf):
    T, F = 292, 16

    P_tensor = np.zeros((len(P), T, F))
    P_time_tensor = np.zeros((len(P), T, F))
    P_mask_tensor = np.zeros((len(P), T, F))
    P_avg_interval_tensor = np.zeros((len(P), T, F))
    P_length_tensor = np.zeros([len(P), 1])

    for i in range(len(P)):
        P_tensor[i][:P[i][4]] = P[i][2]
        if T == 292:
            P[i][1] = P[i][1] / 48
        elif T == 60:
            P[i][1] = P[i][1] / 60
        else:
            P[i][1] = P[i][1] / 2880

        P_time_tensor[i][:P[i][4]] = np.tile(P[i][1], (F, 1)).T
        P_mask_tensor[i][: P[i][4]] = P[i][3]
        P_length_tensor[i] = P[i][4]
        for j in range(F):
            idx_not_zero = np.where(P_mask_tensor[i][:, j])
            if len(idx_not_zero[0]) > 0:
                t = P[i][1][idx_not_zero]
                if len(idx_not_zero[0]) == 1:
                    P_avg_interval_tensor[i][idx_not_zero[0], j] = P[i][4] / 2
                else:
                    right_interval = np.insert(t[1:] - t[:-1], -1, (t[1:] - t[:-1])[-1])
                    left_interval = np.insert(t[1:] - t[:-1], 0, (t[1:] - t[:-1])[0])
                    P_avg_interval_tensor[i][idx_not_zero[0], j] = (left_interval + right_interval) / 2

    P_tensor = mask_normalize(P_tensor, mf, stdf)
    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)

    return torch.FloatTensor(P_tensor), None, \
        torch.FloatTensor(P_avg_interval_tensor), torch.FloatTensor(P_length_tensor), torch.Tensor(P_time_tensor), y_tensor

def mask_normalize_delta(P_delta_tensor):
    # input normalization
    # set missing values to zero after normalization
    idx_missing = np.where(P_delta_tensor == 0)
    idx_existing = np.where(P_delta_tensor != 0)
    max = np.max(P_delta_tensor[idx_existing])
    min = np.min(P_delta_tensor[idx_existing])
    if min == max:
        return P_delta_tensor
    P_delta_tensor = (P_delta_tensor - min) / ((max - min) + 1e-18)
    P_delta_tensor[idx_missing] = 0
    return P_delta_tensor

def get_data_split(base_path='./data/P12', split_path='', dataset='P12'):
    # load data
    if dataset == 'mimic3':
        Ptrain = np.load(base_path + '/mimic3_train_x.npy', allow_pickle=True)
        Pval = np.load(base_path + '/mimic3_val_x.npy', allow_pickle=True)
        Ptest = np.load(base_path + '/mimic3_test_x.npy', allow_pickle=True)
        ytrain = np.load(base_path + '/mimic3_train_y.npy', allow_pickle=True).reshape(-1, 1)
        yval = np.load(base_path + '/mimic3_val_y.npy', allow_pickle=True).reshape(-1, 1)
        ytest = np.load(base_path + '/mimic3_test_y.npy', allow_pickle=True).reshape(-1, 1)
        return Ptrain, Pval, Ptest, ytrain, yval, ytest

    if dataset == 'P12' or dataset == 'physionet':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
    elif dataset == 'P19':
        Pdict_list = np.load(base_path + '/processed_data/PT_dict_list_6.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes_6.npy', allow_pickle=True)

    idx_train, idx_val, idx_test = np.load(base_path + split_path, allow_pickle=True)

    # extract train/val/test examples
    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]
    y = arr_outcomes[:, -1].reshape((-1, 1))
    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]
    return Ptrain, Pval, Ptest, ytrain, yval, ytest

def mask_normalize(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    M = 1 * (P_tensor > 0) + 0 * (P_tensor <= 0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
    return Pfinal_tensor

def mask_normalize_static(P_tensor, ms, ss):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))

    # input normalization
    for s in range(S):
        Ps[s] = (Ps[s] - ms[s]) / (ss[s] + 1e-18)

    # set missing values to zero after normalization
    for s in range(S):
        idx_missing = np.where(Ps[s, :] <= 0)
        Ps[s, idx_missing] = 0

    # reshape back
    Pnorm_tensor = Ps.reshape((S, N)).transpose((1, 0))
    return Pnorm_tensor