# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Apply weighted sum and max pooling to the node representations and concatenate the results."""
# pylint: disable= no-member, arguments-differ, invalid-name

import dgl
import torch
import torch.nn as nn
from dgl.readout import sum_nodes
import numpy as np

__all__ = ['WeightedSumAndMax','WeightAndSum']


class WeightAndSum(nn.Module):
    """Compute importance weights for atoms and perform a weighted sum.

    Parameters
    ----------
    in_feats : int
        Input atom feature size

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import WeightAndSum
    >>>
    >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
    >>> g1_node_feats = th.rand(3, 5)  # feature size is 5
    >>> g1_node_feats
    tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
    >>>
    >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
    >>> g2_node_feats = th.rand(4, 5)  # feature size is 5
    >>> g2_node_feats
    tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
    >>>
    >>> weight_and_sum = WeightAndSum(5)  # create a weight and sum layer(in_feats=16)

    Case 1: Input a single graph

    >>> weight_and_sum(g1, g1_node_feats)
    tensor([[1.2194, 0.9490, 1.3235, 0.9609, 0.7710]],
           grad_fn=<SegmentReduceBackward>)

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = th.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> weight_and_sum(batch_g, batch_f)
    tensor([[1.2194, 0.9490, 1.3235, 0.9609, 0.7710],
            [0.5322, 0.5840, 1.0729, 1.3665, 1.2360]],
           grad_fn=<SegmentReduceBackward>)

    Notes
    -----
    WeightAndSum module was commonly used in molecular property prediction networks,
    see the GCN predictor in `dgl-lifesci <https://github.com/awslabs/dgl-lifesci/blob/
    ae0491431804611ba466ff413f69d435789dbfd5/python/dgllife/model/model_zoo/
    gcn_predictor.py>`__
    to understand how to use WeightAndSum layer to get the graph readout output.
    """
    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, g, feats):
        """Compute molecule representations out of atom representations

        Parameters
        ----------
        g : DGLGraph
            DGLGraph with batch size B for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, self.in_feats)
            Representations for all atoms in the molecules
            * N is the total number of atoms in all molecules

        Returns
        -------
        FloatTensor of shape (B, self.in_feats)
            Representations for B molecules
        """
        with g.local_scope():
            g.ndata['h'] = feats
            g.ndata['w'] = self.atom_weighting(g.ndata['h'])
            h_g_sum = sum_nodes(g, 'h', 'w')

        return h_g_sum


# pylint: disable=W0221
class WeightedSumAndMax(nn.Module):
    r"""Apply weighted sum and max pooling to the node
    representations and concatenate the results.

    Parameters
    ----------
    in_feats : int
        Input node feature size
    """
    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()

        self.weight_and_sum = WeightAndSum(in_feats)

    def forward(self, bg, feats):
        """Readout

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization

        Returns
        -------
        h_g : FloatTensor of shape (B, 2 * M1)
            * B is the number of graphs in the batch
            * M1 is the input node feature size, which must match
              in_feats in initialization
        """
        h_g_sum = self.weight_and_sum(bg, feats)
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_max = dgl.max_nodes(bg, 'h')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g


