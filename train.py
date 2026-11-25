import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import utils
import os,random,csv
import dgl
from models import GATModel
import time
from sklearn.metrics import roc_auc_score
import argparse
import evaluates
from sklearn.model_selection import StratifiedKFold
import copy
import pickle as cPickle
import glob
import networkx as nx
import matplotlib.pyplot as plt
from utils import *

def collate(samples):
    ids, graphs, labels,smiles = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return torch.tensor(ids,dtype=torch.int32),batched_graph, torch.tensor(labels, dtype=torch.float32).view(-1,1)


def train(train_dataset,model,optimizer,val_dataset,lamb1,lamb2):
    kl_c = -1.0
    train_results = pd.DataFrame(columns=['epoch', 'train_loss', 'train_auc'])
    val_results = pd.DataFrame(columns=['epoch', 'val_loss', 'val_auc'])
    train_data_loader = DataLoader(train_dataset, batch_size=args.batchsz, shuffle=True, collate_fn=collate)
    val_data_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=collate)
    start_t = time.time()

    val_auc_best = 0.5
    val_loss_best = np.inf
    val_epoch_best = 0
    num = 0
    for epoch in range(0, args.epochs):
        train_preds = []
        train_targets = []
        train_loss = 0.0
        train_n_total = 0
        model.train()
        for index, item in enumerate(train_data_loader):
            id, bg, target = item
            id, bg, target = id.to(device), bg.to(device), target.to(device)
            alphas,logits,evidence = model(bg,bg.ndata['x'],return_output='alpha')
            labels_1hot = F.one_hot(target.long().squeeze(1), num_classes=2).float()
            loss_mse = compute_mse(labels_1hot, evidence,lamb1,lamb2)
            kl_alpha = evidence * (1 - labels_1hot) + lamb2
            loss_kl = compute_kl_loss(kl_alpha, lamb2)
            if kl_c == -1:
                regr = np.minimum(1.0, epoch / 10.)
                grad_loss = loss_mse + regr * loss_kl
            else:
                grad_loss = loss_mse + kl_c * loss_kl
            optimizer.zero_grad()
            grad_loss.backward()
            optimizer.step()

            probs = alphas / torch.sum(alphas, dim=1, keepdim=True)
            train_pred = probs[:, 1].detach().cpu().numpy()
            train_target = target.detach().cpu().numpy().squeeze(1)
            train_preds.append(train_pred)
            train_targets.append(train_target)

            train_loss += grad_loss.detach().item() * target.size(0)
            train_n_total += target.size(0)
        train_loss_final = train_loss / train_n_total
        train_preds = np.concatenate(train_preds,axis=0)
        train_targets = np.concatenate(train_targets,axis=0)
        train_auc = roc_auc_score(train_targets,train_preds)
        train_results.loc[len(train_results)] = {'epoch': epoch,'train_loss': train_loss_final,'train_auc': train_auc}
        val_preds = []
        val_targets = []
        val_loss = 0
        val_n_total = 0
        model.eval()
        with torch.no_grad():
            for index,item in enumerate(val_data_loader):
                id, bg, target = item
                id, bg, target = id.to(device), bg.to(device), target.to(device)
                alphas,logits,evidence = model(bg,bg.ndata['x'],return_output='alpha')
                labels_1hot = F.one_hot(target.long().squeeze(1), num_classes=2).float()
                loss_mse = compute_mse(labels_1hot, evidence,lamb1,lamb2)
                kl_alpha = evidence * (1 - labels_1hot) + lamb2
                loss_kl = compute_kl_loss(kl_alpha, lamb2)
                if kl_c == -1:
                    regr = np.minimum(1.0, epoch / 10.)
                    grad_loss = loss_mse + regr * loss_kl
                else:
                    grad_loss = loss_mse + kl_c * loss_kl
                val_loss += grad_loss.detach().item() * target.size(0)
                val_n_total += target.size(0)

                probs = alphas / torch.sum(alphas, dim=1, keepdim=True)
                val_pred = probs[:, 1].detach().cpu().numpy()
                val_target = target.detach().cpu().numpy().squeeze(1)
                val_preds.append(val_pred)
                val_targets.append(val_target)
        val_loss_final = val_loss/val_n_total
        val_preds = np.concatenate(val_preds,axis=0)
        val_targets = np.concatenate(val_targets,axis=0)
        val_auc = roc_auc_score(val_targets,val_preds)
        val_results.loc[len(val_results)] = {'epoch': epoch + 1,'val_loss': val_loss_final,'val_auc': val_auc}

        if val_auc_best < val_auc:
            val_loss_best = val_loss_final
            val_auc_best = val_auc
            val_model_best = copy.deepcopy(model)
            val_epoch_best = epoch+1
            num = 0
            torch.save(val_model_best.state_dict(),os.path.join(modelPath, 'bestModel.pt'))
        else:
            num += 1
            print('the performance metric not improved in {} epochs on the valid set'.format(num))
        if num >= 15:
            break

    train_loss_best = train_results.loc[val_epoch_best-1,'train_loss']  
    train_auc_best = train_results.loc[val_epoch_best-1,'train_auc']
    # print('bestEpoch:{}\ttime:{:.4f}\ntrain_best_loss:{:.4f}\ttrain_best_auc:{:.4f}\n'
    #       'val_best_loss:{:.4f}\tval_best_auc:{:.4f}'
    #       .format(val_epoch_best,time.time()-start_t,train_loss_best,train_auc_best,val_loss_best,val_auc_best))
    train_results.loc[len(train_results)] = {'epoch': val_epoch_best,'train_loss': train_loss_best,'train_auc': train_auc_best}
    train_results.to_csv(os.path.join(trainResultsPath,'train_results.csv'),index=False)
    val_results.loc[len(val_results)] = {'epoch': val_epoch_best,'val_loss': val_loss_best,'val_auc': val_auc_best}
    val_results.to_csv(os.path.join(valResultsPath,'val_results.csv'),index=False)
    return val_loss_best, val_auc_best, train_loss_best, train_auc_best


def main(lamb1,lamb2):
    model_results = pd.DataFrame(columns=['train_loss_model', 'train_auc_model', 'val_loss_model', 'val_auc_model'])
    train_dataset = cPickle.load(open('./data/graph_data/train.pkl', 'rb'))
    val_dataset = cPickle.load(open('./data/graph_data/val.pkl', 'rb'))
    val_loss_best,val_auc_best,train_loss_best,train_auc_best = \
        train(train_dataset,model,optimizer,val_dataset,lamb1=lamb1,lamb2=lamb2)
    model_results.loc[len(model_results)] = {'train_loss_model': train_loss_best,'train_auc_model': train_auc_best,'val_loss_model': val_loss_best,'val_auc_model': val_auc_best}
    model_results.to_csv(os.path.join(trainResultsPath,'results.csv'),index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", help="datafile to train", default='./data/train.csv')
    parser.add_argument("--val_file", help="datafile to validation", default='./data/val.csv')
    parser.add_argument("--test_file", help="datafile to test", default='./data/test.csv')
    parser.add_argument("--train_dataset", help="dataset to train",default='./data/graph_data/train.pkl')
    parser.add_argument("--val_dataset", help="dataset to validation",default='./data/graph_data/val.pkl')
    parser.add_argument("--test_dataset", help="dataset to test", default="./data/graph_data/test.pkl")
    parser.add_argument('--seed', type=int, default=72, help='Random seed')
    parser.add_argument('--batchsz', type=int, default=32, help='Size of a batch')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters)')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability)')
    parser.add_argument('--results', help="results of train and val",
                        default='./results/')
    parser.add_argument('--models', help="models to save",
                        default="./models/")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    modelPath = args.models
    trainResultsPath = args.results
    valResultsPath = args.results
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    if not os.path.exists(trainResultsPath):
        os.makedirs(trainResultsPath)
    if not os.path.exists(valResultsPath):
        os.makedirs(valResultsPath)

    # load train, validation, and test datasets
    if not os.path.exists(args.train_dataset):
        train_dataset = utils.read_dataset(args.train_file)  
        cPickle.dump(train_dataset, open(args.train_dataset, 'wb'))
        val_dataset = utils.read_dataset(args.val_file)  
        cPickle.dump(val_dataset, open(args.val_dataset, 'wb'))
        test_dataset = utils.read_dataset(args.test_file) 
        cPickle.dump(test_dataset, open(args.test_dataset, 'wb'))
    else:
        trainVal_dataset = cPickle.load(open(args.train_dataset, 'rb'))

    model = GATModel(utils.atom_features_dim, hidden_feats=[8, 8, 8], num_heads=[8, 8, 8], feat_drops=[0., 0., 0.],
                        attn_drops=[0., 0., 0.], alphas=[0.2, 0.2, 0.2], residuals=[True, True, True],
                        activations=[F.elu, F.elu, F.elu], n_tasks=1, agg_modes=['flatten', 'flatten', 'mean'],
                        predictor_dropout=args.dropout,lamb2=0.4).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay,lr=args.lr)
    main(lamb1=1,lamb2=0.4)

