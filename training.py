import argparse
from dataset_loader import DataLoader
from utils import random_planetoid_splits
from models import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
import numpy as np
import time
from relabel import postel, postel_with_pseudo_label, postel_nodewise, postel_nodewise_with_pseudo_label
from torch_geometric.utils import one_hot
from copy import deepcopy

def relabel_data(args, data, dataset):
    with torch.no_grad():
        if args.labeling_method == 'postel':
            print('Applying PosteL label smoothing...')
            new_label = postel(data, dataset.num_classes, args)
            data.y = one_hot(data.y, dataset.num_classes)
            data.y[data.train_mask] = new_label
            pass
        elif args.labeling_method == 'postel_nodewise':
            print('Applying PosteL (nodewise) label smoothing...')
            new_label = postel_nodewise(data, dataset.num_classes, args)
            data.y = one_hot(data.y, dataset.num_classes)
            data.y[data.train_mask] = new_label
        else:
            raise ValueError

def get_optimizer(args, model):
    if args.net=='GPRGNN':
        optimizer = torch.optim.Adam([{ 'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': args.lr}])

    elif args.net =='BernNet':
        optimizer = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.Bern_lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    return optimizer

def get_pseudo_label(model, data, args):
    with torch.no_grad():
        val_test_mask = torch.logical_or(data.val_mask, data.test_mask)
        logits = model(data)
        pseudo_label = logits[val_test_mask].max(1)[1]

    return pseudo_label

def RunExp(args, dataset, data, Net, percls_trn, val_lb, num_run):
    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]

        if args.labeling_method != 'vanilla':
            loss = F.cross_entropy(out, data.y[data.train_mask])
        else:
            out = F.log_softmax(out, dim=1)
            nll = F.nll_loss(out, data.y[data.train_mask])
            loss = nll

        reg_loss=None
        loss.backward()
        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for split_type, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]

            if args.labeling_method == 'vanilla':
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            else:
                acc = pred.eq(data.y[mask].max(1)[1]).sum().item() / mask.sum().item()

            out = model(data)[mask]

            if args.labeling_method != 'vanilla':
                loss = F.cross_entropy(out, data.y[mask])
            else:
                out = F.log_softmax(out, dim=1)
                loss = F.nll_loss(out, data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    tmp_net = Net(dataset, args)

    #randomly split dataset
    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb,args.seed)
    model, data = tmp_net.to(device), data.to(device)

    backup_y = data.y
    relabel_data(args, data, dataset)
    optimizer = get_optimizer(args, model)

    backbone_best_val_acc = backbone_test_acc = 0
    backbone_best_val_loss = float('inf')
    backbone_val_loss_history = []
    backbone_val_acc_history = []

    time_run=[]
    for epoch in range(args.epochs):
        t_st=time.time()
        train(model, optimizer, data, args.dprate)
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < backbone_best_val_loss:
            backbone_best_val_acc = val_acc
            backbone_best_val_loss = val_loss
            backbone_test_acc = tmp_test_acc
            best_model_state_dict = deepcopy(model.state_dict())
            if args.net =='BernNet':
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu()
                theta = torch.relu(theta).numpy()
            else:
                theta = args.alpha

        if epoch >= 0:
            backbone_val_loss_history.append(val_loss)
            backbone_val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    backbone_val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    #print('The sum of epochs:',epoch)
                    break

    print(f"Backbone test: {backbone_test_acc:.4f}, backbone val: {backbone_best_val_acc:.4f}, backbone_val_loss: {backbone_best_val_loss:.4f}")

    if args.labeling_method != 'vanilla':
        data.y = backup_y

    global_best_val_loss = float('inf')
    last_best_val_loss = backbone_best_val_loss
    test_acc = backbone_test_acc
    best_val_acc = backbone_best_val_acc
    iter_num = 1
    global_best_model_state_dict = best_model_state_dict
    while global_best_val_loss > last_best_val_loss:
        global_best_val_loss = last_best_val_loss
        global_best_model_state_dict = best_model_state_dict
        last_best_test_acc = test_acc
        last_best_val_acc = best_val_acc

        model.load_state_dict(best_model_state_dict)
        pseudo_label = get_pseudo_label(model, data, args)
        del model
        
        if args.labeling_method == 'postel':
            new_label = postel_with_pseudo_label(data, dataset.num_classes, args, pseudo_label)
        elif args.labeling_method == 'postel_nodewise':
            new_label = postel_nodewise_with_pseudo_label(data, dataset.num_classes, args, pseudo_label)
        else:
            raise ValueError
        
        data.y = one_hot(data.y, dataset.num_classes)
        data.y[data.train_mask] = new_label

        model = Net(dataset, args).to(device)
        optimizer = get_optimizer(args, model)

        best_val_acc = test_acc = 0
        best_val_loss = float('inf')
        val_loss_history = []
        val_acc_history = []
        for epoch in range(args.epochs):
            t_st=time.time()
            train(model, optimizer, data, args.dprate)
            time_epoch=time.time()-t_st  # each epoch train times
            time_run.append(time_epoch)

            [train_acc, val_acc, tmp_test_acc], preds, [
                train_loss, val_loss, tmp_test_loss] = test(model, data)
            

            if val_loss < best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                test_acc = tmp_test_acc
                best_model_state_dict = deepcopy(model.state_dict())
                if args.net =='BernNet':
                    TEST = tmp_net.prop1.temp.clone()
                    theta = TEST.detach().cpu()
                    theta = torch.relu(theta).numpy()
                else:
                    theta = args.alpha

            if epoch >= 0:
                val_loss_history.append(val_loss)
                val_acc_history.append(val_acc)
                if args.early_stopping > 0 and epoch > args.early_stopping:
                    tmp = torch.tensor(
                        val_loss_history[-(args.early_stopping + 1):-1])
                    if val_loss > tmp.mean().item():
                        #print('The sum of epochs:',epoch)
                        break

        if args.labeling_method != 'vanilla':
            data.y = backup_y
        last_best_val_loss = best_val_loss
        print(f"Iteration {iter_num}, test: {test_acc:.4f}, val: {best_val_acc:.4f}, val_loss: {best_val_loss:.4f}")
        iter_num += 1

    return last_best_test_acc, last_best_val_acc, theta, time_run, backbone_test_acc, backbone_best_val_acc, iter_num-1, global_best_model_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2108550661, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate.')       
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')  
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')

    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps for APPNP/ChebNet/GPRGNN.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPNP/GPRGNN.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    parser.add_argument('--Init', type=str,choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR', help='initialization for GPRGNN.')
    parser.add_argument('--heads', default=8, type=int, help='attention heads for GAT.')
    parser.add_argument('--output_heads', default=1, type=int, help='output_heads for GAT.')

    parser.add_argument('--dataset', type=str, choices=['Cora','Citeseer','Pubmed','Computers','Photo','Chameleon','Squirrel','Actor','Texas','Cornell'],
                        default='Cornell')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'GPRGNN','BernNet','MLP'], default='GCN')
    parser.add_argument('--Bern_lr', type=float, default=0.01, help='learning rate for BernNet propagation layer.')

    # Arguments for relabeling
    parser.add_argument('--labeling_method', type=str, default='postel', choices=['postel', 'postel_nodewise'])
    parser.add_argument('--soft_label_ratio', type=float, default=0.8, help='interpolation ratio for soft label')
    parser.add_argument('--smoothing_ratio', type=float, default=0.4, help='interpolation ratio for uniform soft label')
    parser.add_argument('--degree_cutoff', type=int, default=1, help='nodes with a degree lower than the cutoff will be disregarded')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for probability calculation')

    args = parser.parse_args()

    #10 fixed seeds for splits
    SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]

    print(args)
    print("---------------------------------------------")

    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
    elif gnn_name == 'BernNet':
        Net = BernNet
    elif gnn_name =='MLP':
        Net = MLP

    dataset = DataLoader(args.dataset)
    data = dataset[0]

    percls_trn = int(round(args.train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(args.val_rate*len(data.y)))

    results = []
    backbone_results = []
    time_results=[]
    iter_num_results=[]
    best_model_state_dicts = []
    for RP in tqdm(range(args.runs)):
        args.seed=SEEDS[RP]
        test_acc, best_val_acc, theta_0,time_run, backbone_test_acc, backbone_best_val_acc, iter_num, best_model_state_dict = RunExp(args, dataset, data, Net, percls_trn, val_lb, RP)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc, 0])
        backbone_results.append([backbone_test_acc, backbone_best_val_acc, 0])
        iter_num_results.append(iter_num)
        best_model_state_dicts.append(best_model_state_dict)
        print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')
        if args.net == 'BernNet':
            print('Theta:', [float('{:.4f}'.format(i)) for i in theta_0])

    run_sum=0
    epochsss=0
    for i in time_results:
        run_sum+=sum(i)
        epochsss+=len(i)

    print("each run avg_time:",run_sum/(args.runs),"s")
    print("each epoch avg_time:",1000*run_sum/epochsss,"ms")

    test_acc_mean, val_acc_mean, _ = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
    backbone_test_acc_mean, backbone_val_acc_mean, _ = np.mean(backbone_results, axis=0) * 100
    backbone_test_acc_std = np.sqrt(np.var(backbone_results, axis=0)[0]) * 100

    values=np.asarray(results)[:,0]
    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))
    backbone_values=np.asarray(backbone_results)[:,0]
    backbone_uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(backbone_values,func=np.mean,n_boot=1000),95)-backbone_values.mean()))
    mean_iter_num = torch.tensor(iter_num_results).to(torch.float).mean()

    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.2f} ± {uncertainty*100:.2f}  \t val acc mean = {val_acc_mean:.2f}')
    
