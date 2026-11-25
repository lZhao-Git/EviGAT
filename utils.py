import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

atom_features_dim = 35
bond_features_dim = 10


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom,explicit_H=True,use_chirality=True,degree_dim=6):
    atom_list = ['B','C','N','O','F','Si','P','S','Cl','Br','I','other']
    results = one_of_k_encoding_unk(atom.GetSymbol(),atom_list) + \
              one_of_k_encoding(atom.GetDegree(),list(range(degree_dim))) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                                    Chem.rdchem.HybridizationType.SP3D2,'other']) + \
              [atom.GetIsAromatic()]

    if explicit_H:
        results = results + one_of_k_encoding(atom.GetTotalNumHs(),[0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results,dtype=np.float32)


def bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),bond.IsInRing()]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
        
    return np.array(bond_feats,dtype=np.float32)


def construct_mol_graph(atom_features_mat, edge_index,bond_features_mat):
    src = edge_index[0]
    dst = edge_index[1]
    num_nodes = atom_features_mat.shape[0]
    g = dgl.graph((torch.from_numpy(src).long(), torch.from_numpy(dst).long()), num_nodes=num_nodes)
    g.ndata['x'] = torch.from_numpy(atom_features_mat).float()
    g.edata['edge_attr'] = torch.from_numpy(bond_features_mat).float()
    return g


def smiles_to_mol_graph(smiles, use_edges=True):
    try:
        mol = Chem.MolFromSmiles(smiles)  # Mol object
        # construct node (atom) feature
        atom_features_mat = np.asarray([
            atom_features(atom, explicit_H=True, use_chirality=True, degree_dim=6) for atom in mol.GetAtoms()],
            dtype=np.float32)

        # # construct edge (bond) index
        src, dest = [], []
        for bond in mol.GetBonds():
            # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]  
            dest += [end, start]  

        edge_index = np.asarray([src, dest], dtype=np.int64)  

        # construct edge (bond) feature
        bond_features_mat = None  # deafult None
        if use_edges:
            bond_features_mat = []
            for bond in mol.GetBonds():
                bond_features_mat += 2 * [bond_features(bond)]
            bond_features_mat = np.asarray(bond_features_mat, dtype=np.float32)
        return mol, construct_mol_graph(atom_features_mat,edge_index,bond_features_mat)
    except:
        print(smiles + 'could not be converted to molecular graph due to the internal errors of RDKit')
        return None, None


def read_dataset(filename):
    """Load  dataset
     atoms['Cl', 'F', 'P', 'Br', 'I', 'N', 'O', 'S', 'B', 'Si', 'C']"""
    samples = []
    data_mat = pd.read_csv(filename) 
    smiles = np.array(data_mat.loc[:, 'smiles'])
    target = np.array(data_mat.loc[:, 'class'], dtype=np.float32)

    for i in range(0, data_mat.shape[0]):
        mol, mol_graph = smiles_to_mol_graph(smiles[i])
        samples.append((i, mol_graph, target[i], smiles[i]))
    return samples

def predict_from_alpha(alpha, eps=1e-8):
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / (S)
    evidence = alpha - 1.0
    uncertainty = alpha.shape[1] / (S.squeeze(1))
    belief = evidence / (S)

    prob_pos = prob[:, 1]
    pred = (prob_pos > 0.5).long()
    return prob_pos, pred, uncertainty, evidence, belief

def predict_from_redl(alpha,lamb2=None, eps=1e-8):
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / (S)
    evidence = alpha - lamb2
    uncertainty = alpha.shape[1] / (S.squeeze(1))
    belief = evidence / (S)

    prob_pos = prob[:, 1]
    pred = (prob_pos > 0.5).long()
    uncertainty_sl = (alpha.shape[1] * lamb2) / S.squeeze(1)
    return prob_pos, pred, uncertainty, evidence, belief,uncertainty_sl

def compute_mse(labels_1hot, evidence,lamb1,lamb2):
    num_classes = evidence.shape[-1]

    gap = labels_1hot - (evidence + lamb2) / \
            (evidence + lamb1 * (torch.sum(evidence, dim=-1, keepdim=True) - evidence) + lamb2 * num_classes)

    loss_mse = gap.pow(2).sum(-1)

    return loss_mse.mean()

def compute_kl_loss(alphas, target_concentration, epsilon=1e-8):
    target_alphas = torch.ones_like(alphas) * target_concentration

    alp0 = torch.sum(alphas, dim=-1, keepdim=True)
    target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

    alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
    alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
    assert torch.all(torch.isfinite(alp0_term)).item()

    alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                            + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                            torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
    alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
    assert torch.all(torch.isfinite(alphas_term)).item()

    loss = torch.squeeze(alp0_term + alphas_term).mean()

    return loss








