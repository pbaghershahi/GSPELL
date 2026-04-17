import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv, GCN, global_mean_pool
from typing import List
from utils import *
import ipdb



class PretrainedModel(nn.Module):
    def __init__(
            self, 
            gnn_type,
            in_channels, 
            hidden_channels, 
            out_channels, 
            gnn_num_hid_layers = 2,
            dropout = 0.0, 
            with_bn = False,
            decoder_type = "linear",
            with_last_dropout = True,
            mode = "node",
            device = "cuda",
            *args, 
            **kwargs
        ) -> None:

        super(PretrainedModel, self).__init__(*args, **kwargs)
        self.gnn_type = gnn_type
        self.num_gnn_layers = gnn_num_hid_layers
        self.dropout = dropout
        self.with_bn = with_bn
        self.decoder_type = decoder_type
        self.device = device

        self.gnn_layers = nn.ModuleList([self.get_gnn_layer(gnn_type, in_channels, hidden_channels)])
        gnn_num_hid_layers -= 1
        if with_bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])

        while gnn_num_hid_layers > 0:
            self.gnn_layers.append(self.get_gnn_layer(gnn_type, hidden_channels, hidden_channels))
            gnn_num_hid_layers -= 1
            if with_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
    
        match decoder_type:
            case "linear":
                self.decoder = nn.Linear(hidden_channels, out_channels)
            case "gnn":
                self.decoder = self.get_gnn_layer(gnn_type, hidden_channels, out_channels)
            case _:
                raise Exception("Choose a decoder from ['linear', 'gnn']!")

        self.num_dropouts = len(self.gnn_layers) if with_last_dropout else len(self.gnn_layers) - 1
        match mode:
            case "node":
                self.forward = self.node_forward
            case "graph":
                self.forward - self.graph_forward
            case _:
                raise NotImplementedError
    

    def get_gnn_layer(self, gnn_type, in_channels, out_channels):
        match gnn_type:
            case "gcn":
                return GCNConv(in_channels, out_channels)
            case "gat":
                return GATConv(in_channels, out_channels)
            case "gin":
                return GINConv(nn.Sequential(nn.Linear(in_channels, out_channels)))
            case "sage":
                return SAGEConv(in_channels, out_channels)
            case _:
                raise Exception("The model is not implemented!")
    

    def graph_forward(
        self, 
        x = None, 
        edge_index =None, 
        batch = None, 
        decoder = True, 
        device = None,
        output_embeds = True
    ):

        device = self.device if device is None else device

        if batch:
            if isinstance(batch, list):
                batch = Batch.from_data_list(batch)
            
            batch = batch.to(device)

            x = batch.x
            edge_index = batch.edge_index

        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)
            if self.with_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            if i < len(self.gnn_layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if not decoder:
            return "scores", x
            
        match self.decoder_type:
            case "gnn":
                scores = F.dropout(x, p=self.dropout, training=self.training)
                scores = self.decoder(scores, edge_index)
                scores = global_mean_pool(scores, batch.batch)
            case "linear":
                x = global_mean_pool(x, batch.batch)
                scores = F.dropout(x, p=self.dropout, training=self.training)
                scores = self.decoder(scores)
        
        if output_embeds:
            return scores, x
        else:
            return scores


    def node_forward(
        self, 
        x = None, 
        edge_index = None, 
        batch = None, 
        decoder = True, 
        device = None,
        output_embeds = True
    ):

        device = self.device if device is None else device

        if batch:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)

        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)
            if self.with_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            if i < self.num_dropouts:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if not decoder:
            return "scores", x

        scores = self.decoder(x)
        
        if output_embeds:
            return scores, x
        else:
            return scores



class GNNToSoftPrompt(nn.Module):
    def __init__(self, gnn_h_dim, num_tokens, llm_h_dim, gnn_out_dim):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(gnn_h_dim, 4 * gnn_h_dim),
            nn.ReLU(),
            nn.Linear(4 * gnn_h_dim, num_tokens * llm_h_dim)
        )

        self.num_tokens = num_tokens
        self.llm_h_dim = llm_h_dim

        if gnn_out_dim is None:
            self.has_backward = False
        else:
            self.has_backward = True
            self.backward_proj = nn.Sequential(
                nn.Linear(num_tokens * llm_h_dim, 2 * gnn_h_dim),
                nn.ReLU(),
                nn.Linear(2 * gnn_h_dim, gnn_out_dim)
            )


    def forward(self, gnn_embedding):
        forward_x = self.projection(gnn_embedding)
        soft_prompt = forward_x.view(-1, self.num_tokens, self.llm_h_dim)

        gnn_logits = None
        if self.has_backward:
            gnn_logits = self.backward_proj(F.relu(forward_x))

        return soft_prompt, gnn_logits