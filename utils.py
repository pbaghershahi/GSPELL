import random, math
import pandas as pd, numpy as np
import re, torch, os
from tqdm import tqdm
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
import logging
import time



def fix_seed(seed_value, random_lib=True, numpy_lib=True, torch_lib=True):
    if random_lib:
        random.seed(seed_value)
    if numpy_lib:
        np.random.seed(seed_value)
    if torch_lib:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)



def empty_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)


        
def setup_logger(
    name="global_logger",
    level=logging.DEBUG,
    stream_handler=True,
    file_handler=True,
    formatter=None,
    log_file='default.log'
    ):
    open(log_file, 'w').close()
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger
        
    logger.setLevel(level)
    if formatter is None:
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d,%H:%M:%S'
            )

    if stream_handler:
        sth = logging.StreamHandler()
        sth.setLevel(level)
        sth.setFormatter(formatter)
        logger.addHandler(sth)

    if file_handler:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



class DummyLogger():
    def __init__(self) -> None:
        pass

    def info(self, log_content):
        print(log_content)



def normalize_(input_tensor, dim=0, mode="max", normal_params=None):
    if normal_params is not None:
        if mode == "max":
            input_tensor.div_(normal_params)
        else:
            scalar = normal_params
            input_tensor = scalar.transform(input_tensor.numpy())
    else:
        if mode == "max":
            max_value = input_tensor.max(dim=0).values
            max_value[max_value==0] = 1.
            input_tensor.div_(max_value)
            normal_params = max_value
        else:
            scalar = StandardScaler()
            input_tensor = scalar.fit_transform(input_tensor.numpy())
            normal_params = scalar
    return torch.as_tensor(input_tensor), normal_params



def load_model(cmodel, pmodel=None, read_checkpoint=True, pretrained_path=None):
    if read_checkpoint and pretrained_path is not None:
        pretrained_dict = torch.load(pretrained_path, weights_only=True)["model_state_dict"]
    else:
        pretrained_dict = pmodel.state_dict()
    model_dict = cmodel.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    cmodel.load_state_dict(pretrained_dict)



def custom_k_hop_subgraph(node_idx, num_hops, edge_index):
    if isinstance(node_idx, int) or (isinstance(node_idx, torch.Tensor) and node_idx.dim() == 0):
        node_idx = torch.tensor([node_idx]).to(node_idx.device)

    search_idx = node_idx
    while num_hops > 0:
        matched_edge_idxs = torch.isin(edge_index, search_idx).nonzero()
        masked_edges = edge_index[:, matched_edge_idxs[:, 1].unique()]
        subset, new_edge_index = masked_edges.unique(return_inverse=True)
        search_idx = subset
        num_hops -= 1
        
    edge_mask = torch.zeros(edge_index.shape[1]).bool()
    edge_mask[matched_edge_idxs[:, 1].unique()] = True
    
    if node_idx.shape[0] == 1:
        mapping = (subset.unsqueeze(-1) == node_idx).T.nonzero()[0, 1].unsqueeze(0)
    else:
        mapping = (subset.unsqueeze(-1) == node_idx).T.nonzero()[1, :]

    return subset, new_edge_index, mapping, edge_mask