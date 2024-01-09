# pylint: disable=E1101
import torch
import torch.nn as nn
import numpy as np
import math
import pickle
from physionet import PhysioNet
from sklearn import model_selection

def evaluate_10(model, P_tensor, P_static_tensor, P_avg_interval_tensor,
                            P_length_tensor, P_time_pre_tensor, P_time_tensor, P_var_prior_emb_tensor,
                      batch_size=100, n_classes=2, static=None):
    model.eval()
    P_tensor = P_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    P_length_tensor = P_length_tensor.cuda()
    P_avg_interval_tensor = P_avg_interval_tensor.cuda()
    P_time_pre_tensor = P_time_pre_tensor.cuda()
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
        P_time_pre = P_time_pre_tensor[start:start + batch_size]
        P_avg_interval = P_avg_interval_tensor[start:start + batch_size]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        middleoutput = model.forward(P, Pstatic, P_avg_interval, P_length, P_time_pre, P_time, P_var_prior_emb_tensor)
        out[start:start + batch_size] = middleoutput.detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[start:start + rem]
        P_time = P_time_tensor[start:start + rem]
        P_length = P_length_tensor[start:start + rem]
        P_time_pre = P_time_pre_tensor[start:start + rem]
        P_avg_interval = P_avg_interval_tensor[start:start + rem]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + rem]
        whatever = model.forward(P, Pstatic, P_avg_interval, P_length, P_time_pre, P_time, P_var_prior_emb_tensor)
        out[start:start + rem] = whatever.detach().cpu()
    return out

def tensorize_normalize_mimic3(P, y, mf, stdf):
    T, F = 292, 16
    # D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_time_pre_tensor = np.zeros((len(P), T, F))
    P_time_tensor = np.zeros((len(P), T, F))
    P_mask_tensor = np.zeros((len(P), T, F))
    P_avg_interval_tensor = np.zeros((len(P), T, F))
    P_length_tensor = np.zeros([len(P), 1])
    # P_static_tensor = np.zeros((len(P), D))

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
        # P_static_tensor[i] = P[i]['extended_static']
        P_length_tensor[i] = P[i][4]
        for j in range(F):
            idx_not_zero = np.where(P_mask_tensor[i][:, j])

            if len(idx_not_zero[0]) > 0:
                x = P_tensor[i][:, j][idx_not_zero]
                t = P[i][1][idx_not_zero]
                P_time_pre_tensor[i][:len(idx_not_zero[0]), j] = t
                if len(idx_not_zero[0]) == 1:
                    P_avg_interval_tensor[i][idx_not_zero[0], j] = P[i][4] / 2
                else:
                    right_interval = np.insert(t[1:] - t[:-1], -1, (t[1:] - t[:-1])[-1])
                    left_interval = np.insert(t[1:] - t[:-1], 0, (t[1:] - t[:-1])[0])
                    P_avg_interval_tensor[i][idx_not_zero[0], j] = (left_interval + right_interval) / 2

    P_tensor = mask_normalize(P_tensor, mf, stdf)

    # P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)

    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)

    return torch.FloatTensor(P_tensor), None, \
        torch.FloatTensor(P_avg_interval_tensor), torch.FloatTensor(P_length_tensor), torch.Tensor(P_time_pre_tensor), torch.Tensor(P_time_tensor), y_tensor

def tensorize_normalize(P, y, mf, stdf, ms, ss):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_time_pre_tensor = np.zeros((len(P), T, F))
    P_time_tensor = np.zeros((len(P), T, F))
    P_fft_tensor = np.zeros((len(P), 3, T, F))
    P_mask_tensor = np.zeros((len(P), T, F))
    P_avg_interval_tensor = np.zeros((len(P), T, F))
    P_length_tensor = np.zeros([len(P), 1])
    P_static_tensor = np.zeros((len(P), D))
    max = 0
    for i in range(len(P)):
        P_tensor[i] = P[i]['arr']
        P[i]['time'] = P[i]['time'] / 60 if T==60 else P[i]['time'] / 2880
        P_time_tensor[i] = P[i]['time']
        if np.max(P[i]['time'][1:] - P[i]['time'][:-1]) > max:
            max = np.max(P[i]['time'][1:] - P[i]['time'][:-1])
        P_mask_tensor[i] = P[i]['arr'] > 0
        P_static_tensor[i] = P[i]['extended_static']
        P_length_tensor[i] = P[i]['length']
        for j in range(F):
            idx_not_zero = np.where(P_mask_tensor[i][:, j])[0]

            if len(idx_not_zero) > 0:
                x = P_tensor[i][:, j][idx_not_zero]
                t = P[i]['time'][idx_not_zero]
                P_time_pre_tensor[i][:len(idx_not_zero), j] = t[:, -1]
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
        torch.FloatTensor(P_avg_interval_tensor), torch.FloatTensor(P_length_tensor), torch.Tensor(P_time_pre_tensor), torch.Tensor(P_time_tensor), y_tensor

def evaluate_nufft_10(model, P_tensor, P_fft_tensor, P_static_tensor, P_avg_interval_tensor,
                            P_length_tensor, P_time_pre_tensor, P_time_tensor, P_var_prior_emb_tensor,
                      batch_size=100, n_classes=2, static=None):
    model.eval()
    P_tensor = P_tensor.cuda()
    P_fft_tensor = P_fft_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    P_length_tensor = P_length_tensor.cuda()
    P_avg_interval_tensor = P_avg_interval_tensor.cuda()
    P_time_pre_tensor = P_time_pre_tensor.cuda()
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
        P_fft = P_fft_tensor[start: start + batch_size]
        P_time = P_time_tensor[start:start + batch_size]
        P_length = P_length_tensor[start:start + batch_size]
        P_time_pre = P_time_pre_tensor[start:start + batch_size]
        P_avg_interval = P_avg_interval_tensor[start:start + batch_size]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        middleoutput = model.forward(P, P_fft, Pstatic, P_avg_interval, P_length, P_time_pre, P_time, P_var_prior_emb_tensor)
        out[start:start + batch_size] = middleoutput.detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[start:start + rem]
        P_fft = P_fft_tensor[start: start + rem]
        P_time = P_time_tensor[start:start + rem]
        P_length = P_length_tensor[start:start + rem]
        P_time_pre = P_time_pre_tensor[start:start + rem]
        P_avg_interval = P_avg_interval_tensor[start:start + rem]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + rem]
        whatever = model.forward(P, P_fft, Pstatic, P_avg_interval, P_length, P_time_pre, P_time, P_var_prior_emb_tensor)
        out[start:start + rem] = whatever.detach().cpu()
    return out
def evaluate_nufft_lamb(model, P_tensor, P_fft_tensor, P_static_tensor, P_avg_interval_tensor,
                            P_length_tensor, P_time_pre_tensor, P_time_tensor, batch_size=100, n_classes=2, static=None):
    model.eval()
    P_tensor = P_tensor.cuda()
    P_fft_tensor = P_fft_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    P_length_tensor = P_length_tensor.cuda()
    P_avg_interval_tensor = P_avg_interval_tensor.cuda()
    P_time_pre_tensor = P_time_pre_tensor.cuda()
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
        P_fft = P_fft_tensor[start: start + batch_size]
        P_time = P_time_tensor[start:start + batch_size]
        P_length = P_length_tensor[start:start + batch_size]
        P_time_pre = P_time_pre_tensor[start:start + batch_size]
        P_avg_interval = P_avg_interval_tensor[start:start + batch_size]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        middleoutput = model.forward(P, P_fft, Pstatic, P_avg_interval, P_length, P_time_pre, P_time)
        out[start:start + batch_size] = middleoutput[0].detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[start:start + rem]
        P_fft = P_fft_tensor[start: start + rem]
        P_time = P_time_tensor[start:start + rem]
        P_length = P_length_tensor[start:start + rem]
        P_time_pre = P_time_pre_tensor[start:start + rem]
        P_avg_interval = P_avg_interval_tensor[start:start + rem]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + rem]
        whatever = model.forward(P, P_fft, Pstatic, P_avg_interval, P_length, P_time_pre, P_time)
        out[start:start + rem] = whatever[0].detach().cpu()
    return out
def nufft(x, t):
    N = len(x)
    f = np.array([i / N for i in range(N)])
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        X[k] = np.sum(x * np.exp(-2j * np.pi * t * f[k]))
    return X, f

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max

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

#def tensorize_normalize(P, y, mf, stdf, ms, ss, interp):
#    T, F = P[0]['arr'].shape
#    D = len(P[0]['extended_static'])
#
#    P_tensor = np.zeros((len(P), T, F))
#    P_time = np.zeros((len(P), T, 1))
#    P_delta_t = np.zeros((len(P), T, 1))
#    P_length = np.zeros([len(P), 1])
#    P_static_tensor = np.zeros((len(P), D))
#    for i in range(len(P)):
#        P_tensor[i] = P[i]['arr']
#        P_time[i] = P[i]['time']
#        P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
#        P_static_tensor[i] = P[i]['extended_static']
#        P_length[i] = P[i]['length']
#        P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
#        if P[i]['length'] < T:
#            P_delta_t[i][P[i]['length']] = 0
#    P_tensor = mask_normalize(P_tensor, mf, stdf)
#    P_tensor = torch.Tensor(P_tensor)
#
#    P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
#    P_tensor = torch.cat((P_tensor, P_time), dim=2)
#
#    P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
#    P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
#    P_static_tensor = torch.Tensor(P_static_tensor)
#    y_tensor = y
#    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
#    return P_tensor, P_static_tensor, torch.FloatTensor(P_delta_t_tensor), torch.tensor(P_length), P_time, y_tensor

def tensorize_normalize_other(P, y, mf, stdf):
    T, F = P[0].shape
    P_time = np.zeros((len(P), T, 1))
    P_delta_t = np.zeros((len(P), T, 1))
    P_length = np.zeros([len(P), 1])

    for i in range(len(P)):
        tim = torch.linspace(0, T, T).reshape(-1, 1)
        P_time[i] = tim
        P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
        P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
        P_length[i] = 600
    P_tensor = mask_normalize(P, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0
    P_tensor = torch.cat((P_tensor, P_time), dim=2)
    P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))

    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, None, torch.FloatTensor(P_delta_t_tensor), torch.tensor(P_length), P_time, y_tensor

# def tensorize_normalize_with_nufft(P, y, mf, stdf, ms, ss):
#     T, F = P[0]['arr'].shape
#     D = len(P[0]['extended_static'])
#
#     P_tensor = np.zeros((len(P), T, F))
#     P_fft_tensor = np.zeros((len(P), 3, T, F))
#     P_time = np.zeros((len(P), T, 1))
#     P_delta_t = np.zeros((len(P), T, 1))
#     P_length = np.zeros([len(P), 1])
#     P_static_tensor = np.zeros((len(P), D))
#
#     P_gt_mask = np.zeros((len(P), T, F))
#     P_obs_mask = np.zeros((len(P), T, F))
#     for i in range(len(P)):
#         P_tensor[i] = P[i]['arr']
#         P_time[i] = P[i]['time']  / 60.0
#
#         observed_masks = P[i]['arr'] > 0
#         masks = observed_masks.reshape(-1).copy()
#         obs_indices = np.where(masks)[0].tolist()
#         miss_indices = np.random.choice(
#             obs_indices, (int)(len(obs_indices) * 0.2), replace=False
#         )
#         masks[miss_indices] = False
#         gt_masks = masks.reshape(observed_masks.shape)
#         P_gt_mask[i] = gt_masks
#         P_obs_mask[i] = observed_masks
#
#         # P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
#         # P_static_tensor[i] = P[i]['extended_static']
#         # P_length[i] = P[i]['length']
#         # P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
#         # if P[i]['length'] < T:
#         #     P_delta_t[i][P[i]['length']] = 0
#     P_tensor = mask_normalize(P_tensor, mf, stdf)
#
#     with open('physionet_benchmark.pk', "wb") as f:
#         pickle.dump(
#             [P_tensor[:, :, :36], P_obs_mask, P_gt_mask, P_time.squeeze(-1), y.squeeze(-1)], f
#         )
#
#     for i in range(len(P)):
#         for j in range(F):
#             idx_not_zero = np.where(P_tensor[i][:, j])
#             if len(idx_not_zero[0]) > 1:
#                 x = P_tensor[i][:, j][idx_not_zero]
#                 t = P_time[i][idx_not_zero]
#                 nufft_complex, f = nufft(x, t)
#                 interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
#                 P_fft_tensor[i][0][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = f
#                 P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
#                     np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
#                 P_fft_tensor[i][2][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
#                     np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))
#
#     P_tensor = torch.Tensor(P_tensor)
#
#     P_time = torch.Tensor(P_time)  # convert mins to hours
#     P_tensor = torch.cat((P_tensor, P_time), dim=2)
#
#     P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
#     P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
#     P_static_tensor = torch.Tensor(P_static_tensor)
#     y_tensor = y
#     y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
#     return P_tensor, torch.FloatTensor(P_fft_tensor), P_static_tensor, torch.FloatTensor(P_delta_t_tensor), \
#         torch.FloatTensor(P_length), P_time, y_tensor

def tensorize_normalize_with_nufft(P, y, mf, stdf, ms, ss):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_time_pre_tensor = np.zeros((len(P), T, F))
    P_time_tensor = np.zeros((len(P), T, F))
    P_fft_tensor = np.zeros((len(P), 3, T, F))
    P_mask_tensor = np.zeros((len(P), T, F))
    P_avg_interval_tensor = np.zeros((len(P), T, F))
    P_length_tensor = np.zeros([len(P), 1])
    P_static_tensor = np.zeros((len(P), D))
    max = 0
    for i in range(len(P)):
        P_tensor[i] = P[i]['arr']
        P[i]['time'] = P[i]['time'] / 60 if T==60 else P[i]['time'] / 2880
        P_time_tensor[i] = P[i]['time']
        if np.max(P[i]['time'][1:] - P[i]['time'][:-1]) > max:
            max = np.max(P[i]['time'][1:] - P[i]['time'][:-1])
        P_mask_tensor[i] = P[i]['arr'] > 0
        P_static_tensor[i] = P[i]['extended_static']
        P_length_tensor[i] = P[i]['length']
        for j in range(F):
            idx_not_zero = np.where(P_mask_tensor[i][:, j])[0]

            if len(idx_not_zero) > 0:
                x = P_tensor[i][:, j][idx_not_zero]
                t = P[i]['time'][idx_not_zero]
#                nufft_complex, f = nufft(x, t)
                P_time_pre_tensor[i][:len(idx_not_zero), j] = t[:, -1]
#                P_fft_tensor[i][0][:len(idx_not_zero), j] = f
#                P_fft_tensor[i][1][:len(idx_not_zero), j] = \
#                    np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
#                P_fft_tensor[i][2][:len(idx_not_zero), j] = \
#                    np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))
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

    return torch.FloatTensor(P_tensor), torch.FloatTensor(P_fft_tensor), torch.FloatTensor(P_static_tensor), \
        torch.FloatTensor(P_avg_interval_tensor), torch.FloatTensor(P_length_tensor), torch.Tensor(P_time_pre_tensor), torch.Tensor(P_time_tensor), y_tensor

def tensorize_normalize_with_nufft_phy(P, y, mf, stdf):
    T = np.max([ex[1].shape[0] for ex in P])
    F = P[0][2].shape[1]

    P_tensor = np.zeros((len(P), T, F))
    P_time_pre_tensor = np.zeros((len(P), T, F))
    P_time_tensor = np.zeros((len(P), T, F))
    P_fft_tensor = np.zeros((len(P), 3, T, F))
    P_mask_tensor = np.zeros((len(P), T, F))
    P_avg_interval_tensor = np.zeros((len(P), T, F))
    P_length_tensor = np.zeros([len(P), 1])
    # max = 0
    for i in range(len(P)):
        length = P[i][1].shape[0]
        P_tensor[i][:length] = P[i][2]
        # P[i][1] = P[i][1] / 48
        P_time_tensor[i][:length] = torch.repeat_interleave(P[i][1].unsqueeze(-1) / 48, F, dim=-1)
        P_mask_tensor[i][:length] = P[i][3]
        P_length_tensor[i] = length
        for j in range(F):
            idx_not_zero = np.where(P_mask_tensor[i][:, j])[0]

            if len(idx_not_zero) > 0:
                x = P_tensor[i][:, j][idx_not_zero]
                t = P[i][1][idx_not_zero].numpy()
                nufft_complex, f = nufft(x, t)
                P_time_pre_tensor[i][:len(idx_not_zero), j] = t
                P_fft_tensor[i][0][:len(idx_not_zero), j] = f
                P_fft_tensor[i][1][:len(idx_not_zero), j] = \
                    np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
                P_fft_tensor[i][2][:len(idx_not_zero), j] = \
                    np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))
                if len(idx_not_zero) == 1:
                    P_avg_interval_tensor[i][idx_not_zero, j] = length / 2
                else:
                    right_interval = np.append(idx_not_zero[1:] - idx_not_zero[:-1], length - idx_not_zero[-1])
                    left_interval = np.insert(idx_not_zero[1:] - idx_not_zero[:-1], 0, idx_not_zero[0])
                    P_avg_interval_tensor[i][idx_not_zero, j] = (left_interval + right_interval) / 2

    P_tensor = mask_normalize(P_tensor, mf, stdf)

    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)

    return torch.FloatTensor(P_tensor), torch.FloatTensor(P_fft_tensor), None, \
        torch.FloatTensor(P_avg_interval_tensor), torch.FloatTensor(P_length_tensor), torch.Tensor(P_time_pre_tensor), torch.Tensor(P_time_tensor), y_tensor


def tensorize_normalize_with_nufft_mimic3(P, y, mf, stdf):
    T, F = 292, 16
    # D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_time_pre_tensor = np.zeros((len(P), T, F))
    P_time_tensor = np.zeros((len(P), T, F))
    P_fft_tensor = np.zeros((len(P), 3, T, F))
    P_mask_tensor = np.zeros((len(P), T, F))
    P_avg_interval_tensor = np.zeros((len(P), T, F))
    P_length_tensor = np.zeros([len(P), 1])
    # P_static_tensor = np.zeros((len(P), D))

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
        # P_static_tensor[i] = P[i]['extended_static']
        P_length_tensor[i] = P[i][4]
        for j in range(F):
            idx_not_zero = np.where(P_mask_tensor[i][:, j])

            if len(idx_not_zero[0]) > 0:
                x = P_tensor[i][:, j][idx_not_zero]
                t = P[i][1][idx_not_zero]
#                nufft_complex, f = nufft(x, t)
                P_time_pre_tensor[i][:len(idx_not_zero[0]), j] = t
#                P_fft_tensor[i][0][:len(idx_not_zero[0]), j] = f
#                P_fft_tensor[i][1][:len(idx_not_zero[0]), j] = \
#                    np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
#                P_fft_tensor[i][2][:len(idx_not_zero[0]), j] = \
#                    np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))
                if len(idx_not_zero[0]) == 1:
                    P_avg_interval_tensor[i][idx_not_zero[0], j] = P[i][4] / 2
                else:
                    right_interval = np.insert(t[1:] - t[:-1], -1, (t[1:] - t[:-1])[-1])
                    left_interval = np.insert(t[1:] - t[:-1], 0, (t[1:] - t[:-1])[0])
                    P_avg_interval_tensor[i][idx_not_zero[0], j] = (left_interval + right_interval) / 2

    P_tensor = mask_normalize(P_tensor, mf, stdf)

    # P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)

    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)

    return torch.FloatTensor(P_tensor), torch.FloatTensor(P_fft_tensor), None, \
        torch.FloatTensor(P_avg_interval_tensor), torch.FloatTensor(P_length_tensor), torch.Tensor(P_time_pre_tensor), torch.Tensor(P_time_tensor), y_tensor


# def tensorize_normalize_with_nufft_mimic3(P, y, mf, stdf):
#     T, F = 292, 16
#
#     P_tensor = np.zeros((len(P), T, F))
#     P_fft_tensor = np.zeros((len(P), 3, T, F))
#     P_time = np.zeros((len(P), T, 1))
#     P_delta_t = np.zeros((len(P), T, 1))
#     P_length = np.zeros([len(P), 1])
#     for i in range(len(P)):
#         P_length[i] = P[i][4]
#         P_tensor[i][:P[i][4]] = P[i][2]
#         P_time[i][:P[i][4]] = P[i][1].reshape(-1, 1) / 60.0
#         P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
#         P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
#         if P_length[i] < T:
#             P_delta_t[i][P[i][4]] = 0
#     P_tensor = mask_normalize(P_tensor, mf, stdf)
#     for i in range(len(P)):
#         for j in range(F):
#             idx_not_zero = np.where(P_tensor[i][:, j])
#             if len(idx_not_zero[0]) > 1:
#                 x = P_tensor[i][:, j][idx_not_zero]
#                 t = P_time[i][idx_not_zero]
#                 nufft_complex, f = nufft(x, t)
#                 interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
#                 P_fft_tensor[i][0][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = f
#                 P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
#                     np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
#                 P_fft_tensor[i][2][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
#                     np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))
#
#     P_tensor = torch.Tensor(P_tensor)
#
#     P_time = torch.Tensor(P_time)  # convert mins to hours
#     P_tensor = torch.cat((P_tensor, P_time), dim=2)
#
#     P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
#     y_tensor = y
#     y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
#     return P_tensor, torch.FloatTensor(P_fft_tensor), None, torch.FloatTensor(P_delta_t_tensor), \
#         torch.FloatTensor(P_length), P_time, y_tensor

def tensorize_normalize_with_nufft_mimic3_missing(P, y, mf, stdf, missingtype, missingratio, idx = None):
    origin_T, F = 292, 16
    if missingratio <= 0:
        return tensorize_normalize_with_nufft_mimic3(P, y, mf, stdf)
    else:
        if missingtype == 'time':
            T = int((1 - missingratio) * origin_T)
            P_tensor = np.zeros((len(P), T, F))
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])
            P_fft_tensor = np.zeros((len(P), 3, T, F))
            for i in range(len(P)):
                selected_num = math.ceil(P[i][4] * (1 - missingratio))
                if selected_num > T:
                    selected_num = T
                idx = np.sort(
                    np.random.choice(P[i][4], selected_num, replace=False))
                length = idx.size
                P_tensor[i][:length] = P[i][2][idx, :]
                P_time[i][:length] = P[i][1][idx].reshape(-1, 1) / 60
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_length[i] = length
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60
                if length < T:
                    P_delta_t[i][length] = 0

            P_tensor = mask_normalize(P_tensor, mf, stdf)
            for i in range(len(P)):
                for j in range(F):
                    idx_not_zero = np.where(P_tensor[i][:, j])
                    if len(idx_not_zero[0]) > 1:
                        x = P_tensor[i][:, j][idx_not_zero]
                        t = P_time[i][idx_not_zero]
                        nufft_complex, f = nufft(x, t)
                        interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
                        # P_fft_tensor[i][0][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = f
                        P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
                        # P_fft_tensor[i][2][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                        #     np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))

            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time)  # convert mins to hours
            P_tensor = torch.cat((P_tensor, P_time), dim=2)

            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, torch.FloatTensor(P_fft_tensor), None, torch.FloatTensor(P_delta_t_tensor), \
                torch.FloatTensor(P_length), P_time, y_tensor
        else:
            T = origin_T
            P_tensor = np.zeros((len(P), T, F))
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])
            P_static_tensor = np.zeros((len(P), D))
            P_fft_tensor = np.zeros((len(P), 3, T, F))
            for i in range(len(P)):
                P_tensor[i][:, idx] = P[i]['arr'][:, idx]
                P_time[i] = P[i]['time']
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_static_tensor[i] = P[i]['extended_static']
                P_length[i] = P[i]['length']
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                if P[i]['length'] < T:
                    P_delta_t[i][P[i]['length']] = 0
            P_tensor = mask_normalize(P_tensor, mf, stdf)
            for i in range(len(P)):
                for j in range(F):
                    idx_not_zero = np.where(P_tensor[i][:, j])
                    if len(idx_not_zero[0]) > 1:
                        x = P_tensor[i][:, j][idx_not_zero]
                        t = P_time[i][idx_not_zero]
                        nufft_complex, f = nufft(x, t)
                        interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
                        P_fft_tensor[i][0][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = f
                        P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
                        P_fft_tensor[i][2][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))

            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
            P_tensor = torch.cat((P_tensor, P_time), dim=2)

            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
            P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
            P_static_tensor = torch.Tensor(P_static_tensor)
            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, torch.FloatTensor(P_fft_tensor), P_static_tensor, torch.FloatTensor(P_delta_t_tensor), \
                torch.FloatTensor(P_length), P_time, y_tensor



def tensorize_normalize_misssing(P, y, mf, stdf, ms, ss, missingtype, missingratio, idx = None):
    origin_T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])
    if missingratio > 0:
        if missingtype == 'time':
            T = int((1 - missingratio) * origin_T)
            P_tensor = np.zeros((len(P), T, F))
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])
            P_static_tensor = np.zeros((len(P), D))
            for i in range(len(P)):
                idx = np.sort(
                    np.random.choice(P[i]['length'], math.ceil(P[i]['length'] * (1 - missingratio)), replace=False))
                length = idx.size
                P_tensor[i][:length] = P[i]['arr'][idx, :]
                P_time[i][:length] = P[i]['time'][idx, :]
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_static_tensor[i] = P[i]['extended_static']
                P_length[i] = length
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                if length < T:
                    P_delta_t[i][length] = 0
            P_tensor = mask_normalize(P_tensor, mf, stdf)
            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
            P_tensor = torch.cat((P_tensor, P_time), dim=2)

            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
            P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
            P_static_tensor = torch.Tensor(P_static_tensor)
            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, P_static_tensor, torch.FloatTensor(P_delta_t_tensor), torch.tensor(
                P_length), P_time, y_tensor
        elif missingtype == 'variable':
                T = origin_T
                F = round((1 - missingratio) * F)
                P_tensor = np.zeros((len(P), T, F))
                P_time = np.zeros((len(P), T, 1))
                P_delta_t = np.zeros((len(P), T, 1))
                P_length = np.zeros([len(P), 1])
                P_static_tensor = np.zeros((len(P), D))
                for i in range(len(P)):
                    P_tensor[i] = P[i]['arr'][:, idx]
                    P_time[i] = P[i]['time']
                    P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                    P_static_tensor[i] = P[i]['extended_static']
                    P_length[i] = P[i]['length']
                    P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                    if P[i]['length'] < T:
                        P_delta_t[i][P[i]['length']] = 0
                P_tensor = mask_normalize(P_tensor, mf[idx], stdf[idx])
                P_tensor = torch.Tensor(P_tensor)

                P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
                P_tensor = torch.cat((P_tensor, P_time), dim=2)

                P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
                P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
                P_static_tensor = torch.Tensor(P_static_tensor)
                y_tensor = y
                y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
                return P_tensor, P_static_tensor, torch.FloatTensor(P_delta_t_tensor), torch.tensor(
                    P_length), P_time, y_tensor
    else:
        return tensorize_normalize(P, y, mf, stdf, ms, ss, None)

def tensorize_normalize_other_with_nufft(P, y, mf, stdf):
    T, F = P[0].shape
    P_time = np.zeros((len(P), T, 1))
    P_delta_t = np.zeros((len(P), T, 1))
    P_length = np.zeros([len(P), 1])
    P_fft_tensor = np.zeros((len(P), 3, T, F))

    for i in range(len(P)):
        tim = torch.linspace(0, T, T).reshape(-1, 1)
        P_time[i] = tim
        P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
        P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
        P_length[i] = 600
    P_tensor = mask_normalize(P, mf, stdf)
    for i in range(len(P)):
        for j in range(F):
            idx_not_zero = np.where(P_tensor[i][:, j])
            if len(idx_not_zero[0]) > 1:
                x = P_tensor[i][:, j][idx_not_zero]
                t = P_time[i][idx_not_zero]
                nufft_complex= np.fft.fft(x)
                interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
                P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                    np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)

    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0
    P_tensor = torch.cat((P_tensor, P_time), dim=2)
    P_delta_t_tensor = P_delta_t.squeeze(-1)
    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, torch.FloatTensor(P_fft_tensor), None, torch.FloatTensor(P_delta_t_tensor), torch.FloatTensor(P_length), P_time, y_tensor

def tensorize_normalize_with_nufft_misssing(P, y, mf, stdf, ms, ss, missingtype, missingratio, idx = None):
    origin_T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])
    if missingratio <= 0:
        return tensorize_normalize_with_nufft(P, y, mf, stdf, ms, ss, None)
    else:
        if missingtype == 'time':
            T = int((1 - missingratio) * origin_T)
            P_tensor = np.zeros((len(P), T, F))
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])
            P_static_tensor = np.zeros((len(P), D))
            P_fft_tensor = np.zeros((len(P), 3, T, F))
            for i in range(len(P)):
                idx = np.sort(
                    np.random.choice(P[i]['length'], math.ceil(P[i]['length'] * (1 - missingratio)), replace=False))
                length = idx.size
                P_tensor[i][:length] = P[i]['arr'][idx, :]
                P_time[i][:length] = P[i]['time'][idx, :] / 60
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_static_tensor[i] = P[i]['extended_static']
                P_length[i] = length
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                if length < T:
                    P_delta_t[i][length] = 0

            P_tensor = mask_normalize(P_tensor, mf, stdf)
            for i in range(len(P)):
                for j in range(F):
                    idx_not_zero = np.where(P_tensor[i][:, j])
                    if len(idx_not_zero[0]) > 1:
                        x = P_tensor[i][:, j][idx_not_zero]
                        t = P_time[i][idx_not_zero]
                        nufft_complex, f = nufft(x, t)
                        interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
                        P_fft_tensor[i][0][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = f
                        P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
                        P_fft_tensor[i][2][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))

            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time)  # convert mins to hours
            P_tensor = torch.cat((P_tensor, P_time), dim=2)

            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
            P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
            P_static_tensor = torch.Tensor(P_static_tensor)
            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, torch.FloatTensor(P_fft_tensor), P_static_tensor, torch.FloatTensor(P_delta_t_tensor), \
                torch.FloatTensor(P_length), P_time, y_tensor
        else:
            T = origin_T
            P_tensor = np.zeros((len(P), T, F))
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])
            P_static_tensor = np.zeros((len(P), D))
            P_fft_tensor = np.zeros((len(P), 3, T, F))
            for i in range(len(P)):
                P_tensor[i][:, idx] = P[i]['arr'][:, idx]
                P_time[i] = P[i]['time']
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_static_tensor[i] = P[i]['extended_static']
                P_length[i] = P[i]['length']
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                if P[i]['length'] < T:
                    P_delta_t[i][P[i]['length']] = 0
            P_tensor = mask_normalize(P_tensor, mf, stdf)
            for i in range(len(P)):
                for j in range(F):
                    idx_not_zero = np.where(P_tensor[i][:, j])
                    if len(idx_not_zero[0]) > 1:
                        x = P_tensor[i][:, j][idx_not_zero]
                        t = P_time[i][idx_not_zero]
                        nufft_complex, f = nufft(x, t)
                        interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
                        P_fft_tensor[i][0][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = f
                        P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
                        P_fft_tensor[i][2][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))

            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
            P_tensor = torch.cat((P_tensor, P_time), dim=2)

            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
            P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
            P_static_tensor = torch.Tensor(P_static_tensor)
            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, torch.FloatTensor(P_fft_tensor), P_static_tensor, torch.FloatTensor(P_delta_t_tensor), \
                torch.FloatTensor(P_length), P_time, y_tensor

def tensorize_normalize_other_missing(P, y, mf, stdf, missingtype, missingratio, idx=None):
    origin_T, F = P[0].shape
    if missingratio > 0:
        if missingtype == 'time':
            T = int((1 - missingratio) * origin_T)
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])
            P_new = np.zeros((len(P), T, F))
            for i in range(len(P)):
                idx = np.sort(np.random.choice(origin_T, round(T), replace=False))
                P_new[i] = P[i, idx, :]
                tim = torch.linspace(0, origin_T, origin_T).reshape(-1, 1)[idx]
                P_time[i] = tim
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1))
                P_length[i] = T
            P_tensor = mask_normalize(P_new, mf, stdf)
            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time)
            P_tensor = torch.cat((P_tensor, P_time), dim=2)
            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))

            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, None, torch.FloatTensor(P_delta_t_tensor), torch.tensor(P_length), P_time, y_tensor
        elif missingtype == 'variable':
            F = round((1 - missingratio) * F)
            T = origin_T
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])

            for i in range(len(P)):
                tim = torch.linspace(0, T, T).reshape(-1, 1)
                P_time[i] = tim
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                P_length[i] = 600
            P_tensor = mask_normalize(P[:, :, idx], mf[idx], stdf[idx])
            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time) / 60.0
            P_tensor = torch.cat((P_tensor, P_time), dim=2)
            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))

            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, None, torch.FloatTensor(P_delta_t_tensor), torch.tensor(P_length), P_time, y_tensor
    else:
        return tensorize_normalize_other(P, y, mf, stdf)

def tensorize_normalize_misssing_ode(P, y, mf, stdf, ms, ss, missingtype, missingratio, idx = None, combined_tt = None):
    origin_T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    if missingratio > 0:
        if missingtype == 'time':
            T = int((1 - missingratio) * origin_T)
            D = P[0]['arr'].shape[1]
            left_time_list = []
            idx_list = []
            left_time = np.zeros((len(P), T))
            for i in range(len(P)):
                idx = np.sort(
                    np.random.choice(P[i]['length'], math.ceil(P[i]['length'] * (1 - missingratio)), replace=False))
                left_time[i][:len(idx)] = P[i]['time'][idx][:, 0]
                left_time_list.append(P[i]['time'][idx][:, 0])
                idx_list.append(idx)
            combined_tt, inverse_indices = torch.unique(torch.cat([torch.FloatTensor(ex) for ex in left_time_list]), sorted=True,
                                                    return_inverse=True)
            offset = 0
            combined_vals = torch.zeros([len(P), len(combined_tt), D])
            combined_mask = torch.zeros([len(P), len(combined_tt), D])
            combined_labels = torch.squeeze(torch.FloatTensor(y))

            for i in range(len(P)):
                tt = left_time_list[i]
                vals = P[i]['arr'][idx_list[i]]
                mask = np.zeros_like(vals)
                mask[np.where(vals != 0)] = 1
                indices = inverse_indices[offset:offset + len(tt)]
                offset += len(tt)
                combined_vals[i, indices] = torch.FloatTensor(vals)
                combined_mask[i, indices] = torch.FloatTensor(mask)
            Ptensor = mask_normalize(combined_vals.numpy(), mf, stdf)
            return torch.FloatTensor(Ptensor), combined_tt / torch.max(combined_tt), combined_labels
        elif missingtype == 'variable':
                T = origin_T
                F = round((1 - missingratio) * F)
                P_tensor = np.zeros((len(P), T, F))
                P_time = np.zeros((len(P), T, 1))
                P_delta_t = np.zeros((len(P), T, 1))
                P_length = np.zeros([len(P), 1])
                P_static_tensor = np.zeros((len(P), D))
                for i in range(len(P)):
                    P_tensor[i] = P[i]['arr'][:, idx]
                    P_time[i] = P[i]['time']
                    P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                    P_static_tensor[i] = P[i]['extended_static']
                    P_length[i] = P[i]['length']
                    P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                    if P[i]['length'] < T:
                        P_delta_t[i][P[i]['length']] = 0
                P_tensor = mask_normalize(P_tensor, mf[idx], stdf[idx])
                P_tensor = torch.Tensor(P_tensor)

                P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
                P_tensor = torch.cat((P_tensor, P_time), dim=2)

                P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
                P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
                P_static_tensor = torch.Tensor(P_static_tensor)
                y_tensor = y
                y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
                return P_tensor, P_static_tensor, torch.FloatTensor(P_delta_t_tensor), torch.tensor(
                    P_length), P_time, y_tensor
    else:
        D = P[0]['arr'].shape[1]
        combined_tt, inverse_indices = torch.unique(torch.cat([torch.FloatTensor(ex['time']) for ex in P]), sorted=True,
                                                    return_inverse=True)
        offset = 0
        combined_vals = torch.zeros([len(P), len(combined_tt), D])
        combined_mask = torch.zeros([len(P), len(combined_tt), D])

        combined_labels = torch.squeeze(torch.FloatTensor(y))

        for i in range(len(P)):
            tt = P[i]['time']
            vals = P[i]['arr']
            mask = np.zeros_like(vals)
            mask[np.where(vals != 0)] = 1
            indices = inverse_indices[offset:offset + len(tt)]
            offset += len(tt)

            combined_vals[i, indices] = torch.FloatTensor(vals).unsqueeze(1)
            combined_mask[i, indices] = torch.FloatTensor(mask).unsqueeze(1)
        Ptensor = mask_normalize(combined_vals.numpy(), mf, stdf)
        return torch.FloatTensor(Ptensor), combined_tt / torch.max(combined_tt), combined_labels

def tensorize_normalize_other_missing_ode(P, y, mf, stdf, missingtype, missingratio, idx=None):
    origin_T, F = P[0].shape
    if missingratio > 0:
        if missingtype == 'time':
            T = int((1 - missingratio) * origin_T)
            D = F
            left_time_list = []
            idx_list = []
            left_time = np.zeros((len(P), T))
            time = np.arange(0, origin_T)
            for i in range(len(P)):
                idx = np.sort(np.random.choice(origin_T, round(T), replace=False))
                left_time[i][:len(idx)] = time[idx][:]
                left_time_list.append(time[idx][:])
                idx_list.append(idx)
            combined_tt, inverse_indices = torch.unique(torch.cat([torch.FloatTensor(ex) for ex in left_time_list]),
                                                        sorted=True,
                                                        return_inverse=True)
            offset = 0
            combined_vals = torch.zeros([len(P), len(combined_tt), D])
            combined_mask = torch.zeros([len(P), len(combined_tt), D])
            combined_labels = torch.squeeze(torch.FloatTensor(y))

            for i in range(len(P)):
                tt = left_time_list[i]
                vals = P[i][idx_list[i]]
                mask = np.zeros_like(vals)
                mask[np.where(vals != 0)] = 1
                indices = inverse_indices[offset:offset + len(tt)]
                offset += len(tt)
                combined_vals[i, indices] = torch.FloatTensor(vals)
                combined_mask[i, indices] = torch.FloatTensor(mask)
            Ptensor = mask_normalize(combined_vals.numpy(), mf, stdf)
            return torch.FloatTensor(Ptensor), combined_tt / torch.max(combined_tt), combined_labels
        elif missingtype == 'variable':
            F = round((1 - missingratio) * F)
            T = origin_T
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])

            for i in range(len(P)):
                tim = torch.linspace(0, T, T).reshape(-1, 1)
                P_time[i] = tim
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                P_length[i] = 600
            P_tensor = mask_normalize(P[:, :, idx], mf[idx], stdf[idx])
            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time) / 60.0
            P_tensor = torch.cat((P_tensor, P_time), dim=2)
            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))

            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, None, torch.FloatTensor(P_delta_t_tensor), torch.tensor(P_length), P_time, y_tensor
    else:
        Ptensor = mask_normalize(P, mf, stdf)
        combined_tt = torch.arange(0, origin_T)
        combined_labels = torch.squeeze(torch.FloatTensor(y))
        return torch.FloatTensor(Ptensor), combined_tt / origin_T, combined_labels

def tensorize_normalize_with_nufft_mimic3_missing_ode(P, y, mf, stdf, missingtype, missingratio, idx = None, combined_tt = None):
    origin_T, F = 292, 16
    if missingratio > 0:
        if missingtype == 'time':
            T = int((1 - missingratio) * origin_T)
            D = F
            left_time_list = []
            idx_list = []
            left_time = np.zeros((len(P), T))
            for i in range(len(P)):
                selected_num = math.ceil(P[i][4] * (1 - missingratio))
                if selected_num > T:
                    selected_num = T
                idx = np.sort(
                    np.random.choice(P[i][4], selected_num, replace=False))
                left_time[i][:len(idx)] = P[i][1][idx]
                left_time_list.append(P[i][1][idx])
                idx_list.append(idx)
            combined_tt, inverse_indices = torch.unique(torch.cat([torch.FloatTensor(ex * 100 // 1 / 100) for ex in left_time_list]), sorted=True,
                                                    return_inverse=True)
            offset = 0
            combined_vals = torch.zeros([len(P), len(combined_tt), D])
            combined_mask = torch.zeros([len(P), len(combined_tt), D])
            combined_labels = torch.squeeze(torch.FloatTensor(y))

            for i in range(len(P)):
                tt = left_time_list[i]
                vals = P[i][2][idx_list[i]]
                mask = P[i][3][idx_list[i]]
                indices = inverse_indices[offset:offset + len(tt)]
                offset += len(tt)
                combined_vals[i, indices] = torch.FloatTensor(vals)
                combined_mask[i, indices] = torch.FloatTensor(mask)
            Ptensor = mask_normalize(combined_vals.numpy(), mf, stdf)
            return torch.FloatTensor(Ptensor), combined_tt / torch.max(combined_tt), combined_labels
        elif missingtype == 'variable':
                T = origin_T
                F = round((1 - missingratio) * F)
                P_tensor = np.zeros((len(P), T, F))
                P_time = np.zeros((len(P), T, 1))
                P_delta_t = np.zeros((len(P), T, 1))
                P_length = np.zeros([len(P), 1])
                P_static_tensor = np.zeros((len(P), D))
                for i in range(len(P)):
                    P_tensor[i] = P[i]['arr'][:, idx]
                    P_time[i] = P[i]['time']
                    P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                    P_static_tensor[i] = P[i]['extended_static']
                    P_length[i] = P[i]['length']
                    P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                    if P[i]['length'] < T:
                        P_delta_t[i][P[i]['length']] = 0
                P_tensor = mask_normalize(P_tensor, mf[idx], stdf[idx])
                P_tensor = torch.Tensor(P_tensor)

                P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
                P_tensor = torch.cat((P_tensor, P_time), dim=2)

                P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
                P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
                P_static_tensor = torch.Tensor(P_static_tensor)
                y_tensor = y
                y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
                return P_tensor, P_static_tensor, torch.FloatTensor(P_delta_t_tensor), torch.tensor(
                    P_length), P_time, y_tensor
    else:
        D = F
        combined_tt, inverse_indices = torch.unique(torch.cat([torch.FloatTensor(ex[1] * 100 // 1 / 100) for ex in P]), sorted=True,
                                                    return_inverse=True)
        offset = 0
        combined_vals = torch.zeros([len(P), len(combined_tt), D])
        combined_mask = torch.zeros([len(P), len(combined_tt), D])

        combined_labels = torch.squeeze(torch.FloatTensor(y))

        for i in range(len(P)):
            tt = (P[i][1] * 100) // 1 / 100
            vals = P[i][2]
            mask = P[i][3]
            indices = inverse_indices[offset:offset + len(tt)]
            offset += len(tt)

            combined_vals[i, indices] = torch.FloatTensor(vals)
            combined_mask[i, indices] = torch.FloatTensor(mask)
        Ptensor = mask_normalize(combined_vals.numpy(), mf, stdf)
        return torch.FloatTensor(Ptensor), combined_tt / torch.max(combined_tt), combined_labels

def tensorize_normalize_other_missing_with_nufft(P, y, mf, stdf, missingtype, missingratio, idx=None):
    origin_T, F = P[0].shape
    if missingratio > 0:
        if missingtype == 'time':
            T = int((1 - missingratio) * origin_T)
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])
            P_new = np.zeros((len(P), T, F))
            P_fft_tensor = np.zeros((len(P), 3, T, F))
            for i in range(len(P)):
                idx = np.sort(np.random.choice(origin_T, round(T), replace=False))
                P_new[i] = P[i, idx, :]
                tim = torch.linspace(0, origin_T, origin_T).reshape(-1, 1)[idx]
                P_time[i] = tim
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1))
                P_length[i] = T
            P_tensor = mask_normalize(P_new, mf, stdf)
            for i in range(len(P)):
                for j in range(F):
                    idx_not_zero = np.where(P_tensor[i][:, j])
                    if len(idx_not_zero[0]) > 1:
                        x = P_tensor[i][:, j][idx_not_zero]
                        t = P_time[i][idx_not_zero]
                        nufft_complex= np.fft.fft(x)
                        interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
                        P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)

            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time)
            P_tensor = torch.cat((P_tensor, P_time), dim=2)
            P_delta_t_tensor = P_delta_t.squeeze(-1)
            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, torch.FloatTensor(P_fft_tensor), None, torch.FloatTensor(P_delta_t_tensor), torch.FloatTensor(P_length), P_time, y_tensor
        elif missingtype == 'variable':
            T = origin_T
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])
            P_fft_tensor = np.zeros((len(P), 3, T, F))
            for i in range(len(P)):
                tim = torch.linspace(0, T, T).reshape(-1, 1)
                P_time[i] = tim
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                P_length[i] = 600
            del_idx = [j for j in np.arange(F) if j not in idx]
            P[:, :, del_idx] = 0
            P_tensor= mask_normalize(P, mf, stdf)
            for i in range(len(P)):
                for j in range(F):
                    idx_not_zero = np.where(P_tensor[i][:, j])
                    if len(idx_not_zero[0]) > 1:
                        x = P_tensor[i][:, j][idx_not_zero]
                        t = P_time[i][idx_not_zero]
                        nufft_complex, f = nufft(x, t)
                        interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
                        P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)


            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time) / 60.0
            P_tensor = torch.cat((P_tensor, P_time), dim=2)
            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))

            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, torch.FloatTensor(P_fft_tensor), None, torch.FloatTensor(P_delta_t_tensor), torch.FloatTensor(P_length), P_time, y_tensor
    else:
        return tensorize_normalize_other_with_nufft(P, y, mf, stdf)

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

def one_hot(y_):
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def create_net(n_inputs, n_outputs, n_layers=0, n_units=10, nonlinear=nn.Tanh, add_softmax=False, dropout=0.0):
    if n_layers >= 0:
        layers = [nn.Linear(n_inputs, n_units)]
        for i in range(n_layers):
            layers.append(nonlinear())
            layers.append(nn.Linear(n_units, n_units))
            layers.append(nn.Dropout(p=dropout))

        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_outputs))
        if add_softmax:
            layers.append(nn.Softmax(dim=-1))

    else:
        layers = [nn.Linear(n_inputs, n_outputs)]

        if add_softmax:
            layers.append(nn.Softmax(dim=-1))

    return nn.Sequential(*layers)
import matplotlib.pyplot as plt

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def get_data_split(base_path='./data/P12data', split_path='', split_type='random', reverse=False, baseline=True, dataset='P12', predictive_label='mortality'):
    # load data
    if dataset == 'mimic3':
        Ptrain = np.load(base_path + '/mimic3_train_x.npy', allow_pickle=True)
        Pval = np.load(base_path + '/mimic3_val_x.npy', allow_pickle=True)
        Ptest = np.load(base_path + '/mimic3_test_x.npy', allow_pickle=True)
        ytrain = np.load(base_path + '/mimic3_train_y.npy', allow_pickle=True).reshape(-1, 1)
        yval = np.load(base_path + '/mimic3_val_y.npy', allow_pickle=True).reshape(-1, 1)
        ytest = np.load(base_path + '/mimic3_test_y.npy', allow_pickle=True).reshape(-1, 1)
        Pdict_list = np.concatenate([Ptrain, Pval, Ptest], axis=0)
        return Ptrain, Pval, Ptest, ytrain, yval, ytest
    # if dataset == 'physionet':
    #     dataset_obj = PhysioNet('data/physionet', train=True,
    #                                   quantization=0.016,
    #                                   download=True, n_samples=4000)[:]
    #     train_data, test_data = model_selection.train_test_split(dataset_obj, train_size=0.8,
    #                                      random_state=42, shuffle=True)
    #     train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
    #                                      random_state=11, shuffle=True)
    #     y_train = np.array([train_data[i][-1].item() for i in range(len(train_data))]).reshape(-1, 1)
    #     y_val = np.array([train_data[i][-1].item() for i in range(len(val_data))]).reshape(-1, 1)
    #     y_test = np.array([train_data[i][-1].item() for i in range(len(test_data))]).reshape(-1, 1)
    #     return train_data, val_data, test_data, y_train, y_val, y_test



    if dataset == 'P12' or dataset == 'physionet':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''
    elif dataset == 'P19':
        Pdict_list = np.load(base_path + '/processed_data/PT_dict_list_6.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes_6.npy', allow_pickle=True)
        dataset_prefix = 'P19_'
    elif dataset == 'PAM':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''  # not applicable

    # plt.rcParams['font.family'] = 'Times New Roman'
    # # row_descriptions = ['ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol', 'Creatinine',
    # #  'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP',
    # #  'MechVent', 'Mg', 'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
    # #  'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'pH']
    # row_descriptions = ['Bilirubin', 'Creatinine',
    #  'PaO2 / FiO2', 'GCS', 'HR', 'Lactate', 'MAP',
    #  'MechVent', 'PaCO2', 'Platelets',
    #  'RespRate', 'SysABP', 'Temp', 'WBC']
    # ind = [5, 7,
    #        9, 10, 14, 16, 17,
    #        18, 24, 26,
    #        27, 29, 30, 34]
    # True_sum = []
    # False_sum = []
    # for i in range(1709, len(Pdict_list)):
    #     if arr_outcomes[i][-1] == 0:
    #         continue
    #     # 获取矩阵的行数和列数
    #     mask_matrix = Pdict_list[i]['arr'][:Pdict_list[i]['length']].T[ind]
    #
    #     num_rows, num_cols = mask_matrix.shape
    #
    #     # 创建一个新的图像
    #     plt.figure()
    #     # 绘制分界线和方格
    #     for row in range(num_rows):
    #         plt.axhline(y=row, color='black', linewidth=1)
    #
    #     for col in range(num_cols + 1):
    #         plt.axvline(x=col, color='black', linewidth=1)
    #     # 遍历矩阵中的每个元素，并绘制相应的方格
    #     for row in range(num_rows):
    #         # if row == 9:
    #         #     continue
    #         for col in range(num_cols):
    #             # 根据mask矩阵中的值设置方格颜色
    #             if mask_matrix[row, col] == 0:
    #                 color = 'white'
    #             elif (row == 0 and mask_matrix[row, col] > 6.0) or\
    #                 (row == 1 and mask_matrix[row, col] > 3.5) or\
    #                 (row == 2 and Pdict_list[i]['arr'][:Pdict_list[i]['length']].T[25, col] / mask_matrix[row, col] < 200) or\
    #                 (row == 3 and mask_matrix[row, col] < 9) or\
    #                 (row == 4 and mask_matrix[row, col] > 100) or\
    #                 (row == 5 and (mask_matrix[row, col] > 2)) or\
    #                 (row == 6 and mask_matrix[row, col] < 70) or\
    #                 (row == 7 and mask_matrix[row, col] == 1) or\
    #                 (row == 8 and mask_matrix[row, col] < 32) or\
    #                 (row == 9 and mask_matrix[row, col] < 50) or\
    #                 (row == 10 and mask_matrix[row, col] > 20) or\
    #                 (row == 11 and mask_matrix[row, col] < 90) or\
    #                     (row == 12 and (mask_matrix[row, col] < 36 or mask_matrix[row, col]) > 38) or \
    #                     (row == 13 and (mask_matrix[row, col] < 4 or mask_matrix[row, col] > 12)):
    #                 color = 'red'
    #             else:
    #                 color = 'black'
    #
    #             # 绘制方格
    #             plt.fill_between([col, col + 1], num_rows - row - 1, num_rows - row, color=color)
    #             # plt.text(-0.5, num_rows - row - 0.5, row_descriptions[row], ha='right', va='center')
    #             plt.text(-0.5, num_rows - row - 0.5, row_descriptions[row], ha='right', va='center',
    #                      fontsize=8)
    #     # 设置坐标轴范围
    #     plt.xlim(0, num_cols)
    #     plt.ylim(0, num_rows)
    #
    #     # 隐藏y轴坐标轴
    #     plt.gca().axes.get_yaxis().set_visible(False)
    #     # plt.gca().axes.get_xaxis().set_axisline_style('->', size=3)
    #     plt.xticks([])
    #
    #     # 在右下角添加标签 "time" 并添加x轴箭头
    #     plt.xlabel("time", fontsize=12, ha='right')
    #     # plt.axis('equal')  # 使x轴和y轴的刻度一致，确保正方形
    #
    #     plt.savefig('./picture/anomaly/' + str(i) + '.png', dpi=300, bbox_inches='tight')
    #     # 显示图像
    #     # plt.show()


    idx_train, idx_val, idx_test = np.load(base_path + split_path, allow_pickle=True)

    # extract train/val/test examples
    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]
    if dataset == 'P12' or dataset == 'P19' or dataset == 'PAM' or dataset == 'physionet':
        if predictive_label == 'mortality':
            y = arr_outcomes[:, -1].reshape((-1, 1))
        elif predictive_label == 'LoS':  # for P12 only
            y = arr_outcomes[:, 3].reshape((-1, 1))
            y = np.array(list(map(lambda los: 0 if los <= 3 else 1, y)))[..., np.newaxis]
    for i in range(len(Pdict_list)):
        y_i = y[i]
        if y_i == 0:
            continue
        arr = Pdict_list[i]['arr']
    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]

    return Ptrain, Pval, Ptest, ytrain, yval, ytest

def evaluate_nufft(model, P_tensor, P_fft_tensor, P_static_tensor, P_avg_interval_tensor,
                            P_length_tensor, P_time_pre_tensor, P_time_tensor, batch_size=100, n_classes=2, static=None):
    model.eval()
    P_tensor = P_tensor.cuda()
    P_fft_tensor = P_fft_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    P_length_tensor = P_length_tensor.cuda()
    P_avg_interval_tensor = P_avg_interval_tensor.cuda()
    P_time_pre_tensor = P_time_pre_tensor.cuda()
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
        P_fft = P_fft_tensor[start: start + batch_size]
        P_time = P_time_tensor[start:start + batch_size]
        P_length = P_length_tensor[start:start + batch_size]
        P_time_pre = P_time_pre_tensor[start:start + batch_size]
        P_avg_interval = P_avg_interval_tensor[start:start + batch_size]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        middleoutput = model.forward(P, P_fft, Pstatic, P_avg_interval, P_length, P_time_pre, P_time)
        out[start:start + batch_size] = middleoutput.detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[start:start + rem]
        P_fft = P_fft_tensor[start: start + rem]
        P_time = P_time_tensor[start:start + rem]
        P_length = P_length_tensor[start:start + rem]
        P_time_pre = P_time_pre_tensor[start:start + rem]
        P_avg_interval = P_avg_interval_tensor[start:start + rem]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + rem]
        whatever = model.forward(P, P_fft, Pstatic, P_avg_interval, P_length, P_time_pre, P_time)
        out[start:start + rem] = whatever.detach().cpu()
    return out

def evaluate_nufft_varemb(model, P_tensor, P_fft_tensor, P_static_tensor, P_avg_interval_tensor,
                            P_length_tensor, P_time_pre_tensor, P_time_tensor, P_var_prior_emb_tensor, batch_size=100, n_classes=2, static=None):
    model.eval()
    P_tensor = P_tensor.cuda()
    P_fft_tensor = P_fft_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    P_length_tensor = P_length_tensor.cuda()
    P_avg_interval_tensor = P_avg_interval_tensor.cuda()
    P_time_pre_tensor = P_time_pre_tensor.cuda()
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
        P_fft = P_fft_tensor[start: start + batch_size]
        P_time = P_time_tensor[start:start + batch_size]
        P_length = P_length_tensor[start:start + batch_size]
        P_time_pre = P_time_pre_tensor[start:start + batch_size]
        P_avg_interval = P_avg_interval_tensor[start:start + batch_size]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        middleoutput = model.forward(P, P_fft, Pstatic, P_avg_interval, P_length, P_time_pre, P_time, P_var_prior_emb_tensor)
        out[start:start + batch_size] = middleoutput.detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[start:start + rem]
        P_fft = P_fft_tensor[start: start + rem]
        P_time = P_time_tensor[start:start + rem]
        P_length = P_length_tensor[start:start + rem]
        P_time_pre = P_time_pre_tensor[start:start + rem]
        P_avg_interval = P_avg_interval_tensor[start:start + rem]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + rem]
        whatever = model.forward(P, P_fft, Pstatic, P_avg_interval, P_length, P_time_pre, P_time, P_var_prior_emb_tensor)
        out[start:start + rem] = whatever.detach().cpu()
    return out


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