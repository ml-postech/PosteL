import torch
from torch_geometric.utils import one_hot, to_torch_sparse_tensor, spmm, remove_self_loops, to_dense_adj, degree, dense_to_sparse, k_hop_subgraph

def neighborhood_label_distribution(data, train_mask, val_mask, test_mask, num_classes, num_hop=1, for_train=True):
    # Get neighborhood label distribution from train data
    assert data.y.dim() == 1
    y = data.y.clone().detach()
    if for_train:
        y[val_mask] = num_classes
        y[test_mask] = num_classes
    one_hot_y = one_hot(y, num_classes+1)
    one_hot_y = one_hot_y[:,:-1]
    sparse_edge_index = to_torch_sparse_tensor(data.edge_index)

    label_dist = spmm(sparse_edge_index, one_hot_y)

    label_distribution = []
    for i in range(num_classes):
        if for_train:
            label_histogram = label_dist[train_mask][data.y[train_mask]==i].sum(dim=0)
        else:
            label_histogram = label_dist[data.y==i].sum(dim=0)
        label_histogram = label_histogram / label_histogram.sum()
        if label_histogram.isnan().max():
            label_distribution.append(torch.ones_like(label_histogram).view(1,-1)/label_histogram.shape[0])
        else:
            label_distribution.append(label_histogram.view(1,-1))
    label_distribution = torch.cat(label_distribution, dim=0)

    return label_distribution.to(torch.float64), label_dist.to(torch.int)

def nodewise_neighborhood_label_distribution(data, train_mask, val_mask, test_mask, num_classes, for_train=True):
    # Get neighborhood label distribution from train data
    assert data.y.dim() == 1
    y = data.y.clone().detach()
    if for_train:
        y[val_mask] = num_classes
        y[test_mask] = num_classes
    one_hot_y = one_hot(y, num_classes+1)
    one_hot_y = one_hot_y[:,:-1]
    dense_adj = to_dense_adj(data.edge_index).squeeze()
    normalized_dense_adj = (1/degree(data.edge_index[0], data.x.shape[0])).unsqueeze(1) * dense_adj
    normalized_sparse_edge_index = dense_to_sparse(normalized_dense_adj)
    normalized_sparse_edge_index = to_torch_sparse_tensor(normalized_sparse_edge_index[0], normalized_sparse_edge_index[1])
    label_dist = spmm(normalized_sparse_edge_index, one_hot_y)

    # re-normazlie to exclude valid, test nodes
    label_dist = label_dist / label_dist.sum(dim=1, keepdim=True)
    label_dist[label_dist.isnan().max(dim=1)[0]] = 0

    label_distribution = []
    for i in range(num_classes):
        if for_train:
            label_histogram = label_dist[train_mask][data.y[train_mask]==i].sum(dim=0)
        else:
            label_histogram = label_dist[data.y==i].sum(dim=0)
        label_histogram = label_histogram / label_histogram.sum()
        label_distribution.append(label_histogram.view(1,-1))
    label_distribution = torch.cat(label_distribution, dim=0)

    sparse_edge_index = to_torch_sparse_tensor(data.edge_index)
    unnormalized_label_dist = spmm(sparse_edge_index, one_hot_y)

    return label_distribution.to(torch.float64), unnormalized_label_dist.to(torch.int)

def nodewise_neighborhood_label_distribution_with_pseudo_label(data, train_mask, val_mask, test_mask, num_classes, pseudo_label):
    # Get neighborhood label distribution from train data
    assert data.y.dim() == 1
    y = data.y.clone().detach()
    test_val_mask = torch.logical_or(val_mask, test_mask)
    y[test_val_mask] = pseudo_label
    one_hot_y = one_hot(y, num_classes)

    dense_adj = to_dense_adj(data.edge_index).squeeze()
    normalized_dense_adj = (1/degree(data.edge_index[0], data.x.shape[0])).unsqueeze(1) * dense_adj
    normalized_sparse_edge_index = dense_to_sparse(normalized_dense_adj)
    normalized_sparse_edge_index = to_torch_sparse_tensor(normalized_sparse_edge_index[0], normalized_sparse_edge_index[1])
    label_dist = spmm(normalized_sparse_edge_index, one_hot_y)

    # re-normazlie to exclude valid, test nodes
    label_dist = label_dist / label_dist.sum(dim=1, keepdim=True)
    label_dist[label_dist.isnan().max(dim=1)[0]] = 0

    label_distribution = []
    for i in range(num_classes):
        label_histogram = label_dist[train_mask][data.y[train_mask]==i].sum(dim=0)
        label_histogram = label_histogram / label_histogram.sum()
        label_distribution.append(label_histogram.view(1,-1))
    label_distribution = torch.cat(label_distribution, dim=0)

    sparse_edge_index = to_torch_sparse_tensor(data.edge_index)
    unnormalized_label_dist = spmm(sparse_edge_index, one_hot_y)

    return label_distribution.to(torch.float64), unnormalized_label_dist.to(torch.int)

def neighborhood_label_distribution_with_pseudo_label(data, train_mask, val_mask, test_mask, num_classes, pseudo_label):
    # Get neighborhood label distribution from train data and valid, test data with pseudo label
    assert data.y.dim() == 1
    y = data.y.clone().detach()
    test_val_mask = torch.logical_or(val_mask, test_mask)
    y[test_val_mask] = pseudo_label

    one_hot_y = one_hot(y, num_classes+1)
    one_hot_y = one_hot_y[:,:-1]
    sparse_edge_index = to_torch_sparse_tensor(data.edge_index)
    label_dist = spmm(sparse_edge_index, one_hot_y)

    label_distribution = []
    for i in range(num_classes):
        label_histogram = label_dist[train_mask][data.y[train_mask]==i].sum(dim=0)
        label_histogram = label_histogram / label_histogram.sum()
        label_distribution.append(label_histogram.view(1,-1))
    label_distribution = torch.cat(label_distribution, dim=0)

    return label_distribution.to(torch.float64), label_dist.to(torch.int)

def neighborhood_histogram_likelihood(label_histogram, label_distribution):
    likelihood = []
    for label_hist in label_histogram:
        likeli = (label_distribution**label_hist).prod(dim=1)
        likelihood.append(likeli)
    likelihood = torch.stack(likelihood)
    
    return likelihood

def label_smoothing(data, num_classes, smoothing_ratio):
    one_hot_y = one_hot(data.y[data.train_mask])
    new_y = (1-smoothing_ratio) * one_hot_y + smoothing_ratio/(num_classes-1) * torch.logical_not(one_hot_y)

    return new_y

def postel(data, num_classes, args):
    label_marginal_prob = one_hot(data.y[data.train_mask], num_classes).mean(dim=0)
    label_distribution, label_histogram = neighborhood_label_distribution(data, data.train_mask, data.val_mask, data.test_mask, num_classes, 1)
    hist_log_likelihood, degree_mask = neighborhood_histogram_log_likelihood(label_histogram[data.train_mask], label_distribution, args.degree_cutoff)
    
    new_label, mask = prob_label_from_log_likelihood(hist_log_likelihood, label_marginal_prob, args.temperature)

    mask = torch.logical_or(mask, degree_mask)
    new_label[mask] = one_hot(data.y[data.train_mask], num_classes)[mask]

    one_hot_y = one_hot(data.y[data.train_mask], num_classes)
    new_label = (1-args.smoothing_ratio) * new_label + args.smoothing_ratio/(num_classes-1) * torch.logical_not(one_hot_y)
    new_label = args.soft_label_ratio * new_label + (1-args.soft_label_ratio) * one_hot(data.y[data.train_mask], num_classes)

    return new_label

def postel_with_pseudo_label(data, num_classes, args, pseudo_label):
    label_marginal_prob = one_hot(torch.cat((data.y[data.train_mask], pseudo_label[(pseudo_label!=num_classes).nonzero().squeeze()])),num_classes=num_classes).mean(dim=0)
    label_distribution, label_histogram = neighborhood_label_distribution_with_pseudo_label(data, data.train_mask, data.val_mask, data.test_mask, num_classes, pseudo_label)
    hist_log_likelihood, degree_mask = neighborhood_histogram_log_likelihood(label_histogram[data.train_mask], label_distribution, args.degree_cutoff)
    
    new_label, mask = prob_label_from_log_likelihood(hist_log_likelihood, label_marginal_prob, args.temperature)

    mask = torch.logical_or(mask, degree_mask)
    new_label[mask] = one_hot(data.y[data.train_mask], num_classes)[mask]

    one_hot_y = one_hot(data.y[data.train_mask])
    new_label = (1-args.smoothing_ratio) * new_label + args.smoothing_ratio/(num_classes-1) * torch.logical_not(one_hot_y)
    new_label = args.soft_label_ratio * new_label + (1-args.soft_label_ratio) * one_hot(data.y[data.train_mask])

    return new_label

def neighborhood_histogram_log_likelihood(label_histogram, label_distribution, degree_cutoff):
    likelihood = []
    mask = []
    for label_hist in label_histogram:
        likeli = (label_distribution**label_hist).prod(dim=1)
        likelihood.append(torch.log(likeli))
        mask.append(label_hist.sum()<degree_cutoff)
    likelihood = torch.stack(likelihood)
    mask = torch.tensor(mask).to(label_histogram.device)
    
    return likelihood, mask

def prob_label_from_log_likelihood(log_likelihood, prior, temperature=1):
    log_prob = log_likelihood + torch.log(prior)
    log_prob = log_prob - log_prob.max()
    prob = torch.exp(log_prob)
    prob = torch.pow(prob, temperature)
    prob = prob / prob.sum(dim=-1, keepdim=True)
    return prob.to(torch.float32), prob.isnan().max(dim=1).values

def postel_nodewise(data, num_classes, args):
    one_hot_y = one_hot(data.y[data.train_mask])
    label_marginal_prob = one_hot_y.mean(dim=0)
    label_distribution, label_histogram = nodewise_neighborhood_label_distribution(data, data.train_mask, data.val_mask, data.test_mask, num_classes)
    hist_log_likelihood, degree_mask = neighborhood_histogram_log_likelihood(label_histogram[data.train_mask], label_distribution, args.degree_cutoff)
    new_label, mask = prob_label_from_log_likelihood(hist_log_likelihood, label_marginal_prob, args.temperature)

    mask = torch.logical_or(mask, degree_mask)

    new_label[mask] = one_hot(data.y[data.train_mask], num_classes)[mask]

    one_hot_y = one_hot(data.y[data.train_mask], num_classes)
    new_label = (1-args.smoothing_ratio) * new_label + args.smoothing_ratio/(num_classes-1) * torch.logical_not(one_hot_y)
    new_label = args.soft_label_ratio * new_label + (1-args.soft_label_ratio) * one_hot(data.y[data.train_mask], num_classes)

    return new_label

def postel_nodewise_with_pseudo_label(data, num_classes, args, pseudo_label):
    label_marginal_prob = one_hot(torch.cat((data.y[data.train_mask], pseudo_label))).mean(dim=0)
    label_distribution, label_histogram = nodewise_neighborhood_label_distribution_with_pseudo_label(data, data.train_mask, data.val_mask, data.test_mask, num_classes, pseudo_label)
    hist_log_likelihood, degree_mask = neighborhood_histogram_log_likelihood(label_histogram[data.train_mask], label_distribution, args.degree_cutoff)
    new_label, mask = prob_label_from_log_likelihood(hist_log_likelihood, label_marginal_prob, args.temperature)

    mask = torch.logical_or(mask, degree_mask)

    new_label[mask] = one_hot(data.y[data.train_mask], num_classes)[mask]

    one_hot_y = one_hot(data.y[data.train_mask], num_classes)
    new_label = (1-args.smoothing_ratio) * new_label + args.smoothing_ratio/(num_classes-1) * torch.logical_not(one_hot_y)
    new_label = args.soft_label_ratio * new_label + (1-args.soft_label_ratio) * one_hot(data.y[data.train_mask], num_classes)

    return new_label