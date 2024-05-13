# -*- coding:utf-8 -*-
import os
import argparse
import warnings
import time
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='physionet', choices=['P12', 'P19', 'physionet', 'mimic3'])
parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--epochs', type=int, default=10)  #
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=4)
parser.add_argument('--rarity_alpha', type=float, default=1)
parser.add_argument('--query_vector_dim', type=int, default=5)
parser.add_argument('--node_emb_dim', type=int, default=8)
parser.add_argument('--plm', type=str, default='bert')
parser.add_argument('--plm_rep_dim', type=int, default=768)
parser.add_argument('--source', type=str, default='gpt')

args, unknown = parser.parse_known_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from model import *
from utils import *

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

# Create model save path
model_path = './models/'
if not os.path.exists(model_path):
    os.mkdir(model_path)

# Load command line hyperparameters
dataset = args.dataset
batch_size = args.batch_size
learning_rate = args.lr
num_epochs = args.epochs
hidden_dim = args.hidden_dim
rarity_alpha = args.rarity_alpha
query_vector_dim = args.query_vector_dim
node_emb_dim = args.node_emb_dim
plm_rep_dim = args.plm_rep_dim
source = args.source

print('Dataset used: ', dataset)

# Set dataset parameters
if dataset == 'P12':
    base_path = 'data/P12'
    start = 0
    variables_num = 36
    d_static = 9
    timestamp_num = 215
    n_class = 2
    split_idx = 1
elif dataset == 'physionet':
    base_path = 'data/physionet'
    start = 4
    variables_num = 36
    d_static = 9
    timestamp_num = 215
    n_class = 2
    split_idx = 5
elif dataset == 'P19':
    base_path = 'data/P19'
    d_static = 6
    variables_num = 34
    timestamp_num = 60
    n_class = 2
    split_idx = 1
elif dataset == 'mimic3':
    base_path = 'data/mimic3'
    start = 0
    d_static = 0
    variables_num = 16
    timestamp_num = 292
    n_class = 2
    split_idx = 0

# Evaluation metrics
acc_arr = []
auprc_arr = []
auroc_arr = []

if source == 'gpt':
    suffix = '_var_rep_gpt_source.pt'
elif source == 'name':
    suffix = '_var_rep_name_source.pt'
elif source == 'wiki':
    suffix = '_var_rep_wiki_source.pt'

# Run five experiments
for k in range(5):
    # Set different random seed
    torch.manual_seed(k)
    torch.cuda.manual_seed(k)
    np.random.seed(k)

    # Load semantic representations of variables obtained through PLM
    if dataset == 'P12':
        split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
        P_var_plm_rep_tensor = torch.load(base_path + '/P12_' + args.plm + suffix).to(device)
    elif dataset == 'physionet':
        split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
        P_var_plm_rep_tensor = torch.load(base_path + '/physionet_' + args.plm + suffix).to(device)
    elif dataset == 'P19':
        split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
        P_var_plm_rep_tensor = torch.load(base_path + '/P19_' + args.plm + suffix).to(device)
    elif dataset == 'mimic3':
        split_path = ''
        P_var_plm_rep_tensor = torch.load(base_path + '/mimic3_' + args.plm + suffix).to(device)

    # Prepare data and split the dataset
    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, dataset=dataset)
    print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

    # Normalize data and extract required model inputs
    if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet':
        T, F = Ptrain[0]['arr'].shape
        D = len(Ptrain[0]['extended_static'])
        Ptrain_tensor = np.zeros((len(Ptrain), T, F))
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))

        for i in range(len(Ptrain)):
            Ptrain_tensor[i] = Ptrain[i]['arr']
            Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

        # Calculate mean and standard deviation of variables in the training set
        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset=dataset)

        Ptrain_tensor, Ptrain_static_tensor, Ptrain_avg_interval_tensor, \
            Ptrain_length_tensor, Ptrain_time_tensor, ytrain_tensor \
            = tensorize_normalize_extract_feature(Ptrain, ytrain, mf, stdf, ms, ss)
        Pval_tensor, Pval_static_tensor, Pval_avg_interval_tensor, \
            Pval_length_tensor, Pval_time_tensor, yval_tensor \
            = tensorize_normalize_extract_feature(Pval, yval, mf, stdf, ms, ss)
        Ptest_tensor, Ptest_static_tensor, Ptest_avg_interval_tensor, \
            Ptest_length_tensor, Ptest_time_tensor, ytest_tensor \
            = tensorize_normalize_extract_feature(Ptest, ytest, mf, stdf, ms, ss)

    elif dataset == 'mimic3':
        T, F = timestamp_num, variables_num
        Ptrain_tensor = np.zeros((len(Ptrain), T, F))
        for i in range(len(Ptrain)):
            Ptrain_tensor[i][:Ptrain[i][4]] = Ptrain[i][2]

        # Calculate mean and standard deviation of variables in the training set
        mf, stdf = getStats(Ptrain_tensor)

        Ptrain_tensor, Ptrain_static_tensor, Ptrain_avg_interval_tensor, \
            Ptrain_length_tensor, Ptrain_time_tensor, ytrain_tensor \
            = tensorize_normalize_exact_feature_mimic3(Ptrain, ytrain, mf, stdf)
        Pval_tensor, Pval_static_tensor, Pval_avg_interval_tensor, \
            Pval_length_tensor, Pval_time_tensor, yval_tensor \
            = tensorize_normalize_exact_feature_mimic3(Pval, yval, mf, stdf)
        Ptest_tensor, Ptest_static_tensor, Ptest_avg_interval_tensor, \
            Ptest_length_tensor, Ptest_time_tensor, ytest_tensor \
            = tensorize_normalize_exact_feature_mimic3(Ptest, ytest, mf, stdf)

    # Load the model
    model = TEDGN(DEVICE=device,
                  hidden_dim=hidden_dim,
                  num_of_variables=variables_num,
                  num_of_timestamps=timestamp_num,
                  d_static=d_static,
                  n_class=2,
                  rarity_alpha=rarity_alpha,
                  query_vector_dim=query_vector_dim,
                  node_emb_dim=node_emb_dim,
                  plm_rep_dim=plm_rep_dim)


    params = (list(model.parameters()))
    print('model', model)
    print('parameters:', count_parameters(model))

    # Cross-entropy loss, Adam optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Upsample minority class
    idx_0 = np.where(ytrain == 0)[0]
    idx_1 = np.where(ytrain == 1)[0]
    n0, n1 = len(idx_0), len(idx_1)
    expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
    expanded_n1 = len(expanded_idx_1)
    K0 = n0 // int(batch_size / 2)
    K1 = expanded_n1 // int(batch_size / 2)
    n_batches = np.min([K0, K1])

    best_val_epoch = 0
    best_aupr_val = best_auc_val = 0.0
    best_loss_val = 100.0

    print('Stop epochs: %d, Batches/epoch: %d, Total batches: %d' % (
        num_epochs, n_batches, num_epochs * n_batches))

    start = time.time()

    for epoch in range(num_epochs):
        """Training"""
        model.train()

        # Shuffle data
        np.random.shuffle(expanded_idx_1)
        I1 = expanded_idx_1
        np.random.shuffle(idx_0)
        I0 = idx_0
        for n in range(n_batches):
            # Get current batch data
            idx0_batch = I0[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
            idx1_batch = I1[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
            idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
            P, P_static, P_avg_interval, P_length, P_time, y = \
                Ptrain_tensor[idx].cuda(), Ptrain_static_tensor[idx].cuda() if d_static != 0 else None, \
                    Ptrain_avg_interval_tensor[idx].cuda(), Ptrain_length_tensor[idx].cuda(), \
                    Ptrain_time_tensor[idx].cuda(), ytrain_tensor[idx].cuda()

            # Backward pass
            outputs = model.forward(P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Calculate training set evaluation metrics
        train_probs = torch.squeeze(torch.sigmoid(outputs))
        train_probs = train_probs.cpu().detach().numpy()
        train_y = y.cpu().detach().numpy()
        train_auroc = roc_auc_score(train_y, train_probs[:, 1])
        train_auprc = average_precision_score(train_y, train_probs[:, 1])

        """Validation"""
        model.eval()
        with torch.no_grad():
            out_val = evaluate_model(model, Pval_tensor, Pval_static_tensor, Pval_avg_interval_tensor,
                                        Pval_length_tensor, Pval_time_tensor, P_var_plm_rep_tensor,
                                        n_classes=n_class, batch_size=batch_size)
            out_val = torch.squeeze(torch.sigmoid(out_val))
            out_val = out_val.detach().cpu().numpy()
            y_val_pred = np.argmax(out_val, axis=1)
            acc_val = np.sum(yval.ravel() == y_val_pred.ravel()) / yval.shape[0]
            val_loss = torch.nn.CrossEntropyLoss().cuda()(torch.from_numpy(out_val), torch.from_numpy(yval.squeeze(1)).long())
            auc_val = roc_auc_score(yval, out_val[:, 1])
            aupr_val = average_precision_score(yval, out_val[:, 1])
            print(
                "Validation: Epoch %d, train_loss:%.4f, train_auprc:%.2f, train_auroc:%.2f, val_loss:%.4f, acc_val: %.2f, aupr_val: %.2f, auc_val: %.2f" %
                (epoch, loss.item(), train_auprc * 100, train_auroc * 100,
                 val_loss.item(), acc_val * 100, aupr_val * 100, auc_val * 100))

            # Save the model weights with the best AUPRC on the validation set
            if aupr_val > best_aupr_val:
                best_auc_val = auc_val
                best_aupr_val = aupr_val
                best_val_epoch = epoch
                save_time = str(int(time.time()))
                torch.save(model.state_dict(),
                           model_path + '_' + dataset + '_' + save_time + '_' + str(k) + '.pt')

    end = time.time()
    time_elapsed = end - start
    print('Total Time elapsed: %.3f mins' % (time_elapsed / 60.0))

    """testing"""
    model.eval()
    model.load_state_dict(
        torch.load(model_path + '_' + dataset + '_' + save_time + '_' + str(k) + '.pt'))
    with torch.no_grad():
        out_test = evaluate_model(model, Ptest_tensor, Ptest_static_tensor, Ptest_avg_interval_tensor,
                                  Ptest_length_tensor, Ptest_time_tensor, P_var_plm_rep_tensor,
                                  n_classes=n_class,batch_size=batch_size).numpy()
        denoms = np.sum(np.exp(out_test.astype(np.float64)), axis=1).reshape((-1, 1))
        y_test = ytest.copy()
        probs = np.exp(out_test.astype(np.float64)) / denoms
        ypred = np.argmax(out_test, axis=1)
        acc = np.sum(y_test.ravel() == ypred.ravel()) / y_test.shape[0]
        auc = roc_auc_score(y_test, probs[:, 1])
        aupr = average_precision_score(y_test, probs[:, 1])

        print('Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f' % (auc * 100, aupr * 100, acc * 100))
        print('classification report', classification_report(y_test, ypred))
        print(confusion_matrix(y_test, ypred, labels=list(range(n_class))))

    acc_arr.append(acc * 100)
    auprc_arr.append(aupr * 100)
    auroc_arr.append(auc * 100)

print('args.dataset', args.dataset)
# Display the mean and standard deviation of five runs
mean_acc, std_acc = np.mean(acc_arr), np.std(acc_arr)
mean_auprc, std_auprc = np.mean(auprc_arr), np.std(auprc_arr)
mean_auroc, std_auroc = np.mean(auroc_arr), np.std(auroc_arr)
print('------------------------------------------')
print('Accuracy = %.1f±%.1f' % (mean_acc, std_acc))
print('AUPRC    = %.1f±%.1f' % (mean_auprc, std_auprc))
print('AUROC    = %.1f±%.1f' % (mean_auroc, std_auroc))