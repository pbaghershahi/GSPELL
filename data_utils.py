import torch, random, os, logging, ipdb, gc
import numpy as np
from torch_geometric.datasets import QM9, TUDataset, CitationFull, Planetoid, Flickr
from torch_geometric.utils import k_hop_subgraph
from utils import *
from model import *
import pandas as pd
from torch_geometric.loader import DataLoader as PyG_Dataloader
from torch_geometric.data import Data, Batch, Dataset as PyG_Dataset
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.mixture import GaussianMixture
from collections import OrderedDict
from TAGLAS import get_dataset
from copy import deepcopy
from collections import defaultdict, Counter
from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger("global_logger")



class GDataset(nn.Module):
    def __init__(self,):
        pass

    def init_loaders_(self,):
        "Implement this is children classes"
        pass

    def normalize_feats_(self,):
        "Implement this is children classes"
        return "x"

    def init_ds_idxs_(self, train_idxs, val_idxs, test_idxs, train_test_split, shuffle, seed):
        if (train_idxs is not None) and (val_idxs is not None) and (test_idxs is not None):
            self.n_train = train_idxs.size(0)
            self.n_val = val_idxs.size(0)
            self.n_test = test_idxs.size(0)
            self.train_idxs = train_idxs
            self.val_idxs = val_idxs
            self.test_idxs = test_idxs
        else:
            all_idxs = torch.arange(self.num_samples)
            if shuffle: 
                perm = torch.randperm(self.num_samples)
                all_idxs = all_idxs[perm]
            if train_test_split[0] + train_test_split[1] != 1.0:
                val_per = 1 - (train_test_split[0] + train_test_split[1])
            else:
                val_per = 0.0
            self.all_idxs = all_idxs
            self.n_train = int(self.num_samples * train_test_split[0])
            self.n_test = int(self.num_samples * train_test_split[1])
            self.n_val = self.num_samples - (self.n_train + self.n_test)
            self.train_idxs = all_idxs[:int(self.n_train)]
            self.val_idxs = all_idxs[self.n_train:-self.n_test]
            self.test_idxs = all_idxs[-self.n_test:]

    def initialize(self,):
        "Implement this is children classes"
        return "x"



class NodeToGraphDataset(GDataset):

    def __init__(self, **kwargs) -> None:
        super(NodeToGraphDataset, self).__init__()


    def get_data(self, dataset, sample_size=-1, random_sampling=False):
        sampled_idxs = None
        if sample_size > 0:
            if random_sampling:
                sampled_idxs = torch.randperm(dataset.x.size(0))[:sample_size]
            else:
                sampled_idxs = torch.arange(sample_size)
            data = dataset._data.subgraph(sampled_idxs)
        else:
            data = dataset._data

        return data, sampled_idxs


    @property
    def x(self,):
        return self._data.x


    def init_ds_from_indices(self, node_indices):
        "Implement this function in subclasses!"
        pass


    def normalize_feats_(
        self, 
        train_ds,
        test_ds = None,
        val_ds = None,
        normalize_mode = "normal", 
        **kwargs
    ):
        train_ds.x, train_normal_params = normalize_(train_ds.x, dim=0, mode=normalize_mode)
        ds_out = [train_ds, None, None]
        if test_ds is not None:
            test_ds.x, _ = normalize_(
                test_ds.x, 
                dim=0, 
                mode=normalize_mode, 
                normal_params = train_normal_params
            )
            ds_out[1] = test_ds
        if val_ds is not None:
            val_ds.x, _ = normalize_(
                val_ds.x, 
                dim=0, 
                mode=normalize_mode, 
                normal_params = train_normal_params
            )
            ds_out[2] = val_ds
        return ds_out


    def get_bow_embedding(self, data, max_num_frequent_words=None):

        vectorizer = CountVectorizer(
            max_features = max_num_frequent_words, 
            binary = True
        )

        sentences = list(data.raw_text)
        bow_embeddings = vectorizer.fit_transform(sentences).toarray()

        frequent_words = vectorizer.get_feature_names_out()

        sentence_frequent_words = []
        for sentence_vector in bow_embeddings:
            words_in_sentence = [
                frequent_words[i] for i, present in enumerate(sentence_vector) if present == 1
            ]
            sentence_frequent_words.append(words_in_sentence)

        return torch.as_tensor(bow_embeddings, dtype=torch.float), sentence_frequent_words


    def initialize(
        self,
        train_idxs: torch.Tensor = None,
        val_idxs: torch.Tensor = None,
        test_idxs: torch.Tensor = None,
        train_test_split = [0.85, 0.15],
        loader_collate = None,
        batch_size = 32, 
        normalize_mode = None,
        shuffle = False, 
        num_workers = 0,
        **kwargs
    ) -> None:

        self.init_ds_idxs_(
            train_idxs = train_idxs, val_idxs = val_idxs, test_idxs = test_idxs,
            train_test_split = train_test_split,
            shuffle = shuffle, seed = kwargs["seed"] if "seed" in kwargs else 2411
        )

        train_ds = self.init_ds_from_indices(self.train_idxs)
        test_ds, val_ds = None, None
        if self.n_test > 0:
            test_ds = self.init_ds_from_indices(self.test_idxs)
        if self.n_val > 0:
            val_ds = self.init_ds_from_indices(self.val_idxs)

        if normalize_mode in ["normal", "max"]:
            self.normalize_feats_(
                train_ds = train_ds,
                test_ds = test_ds,
                val_ds = val_ds,
                normalize_mode = normalize_mode
            )

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=loader_collate, num_workers=num_workers)
        if self.n_test > 0:
            self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate, num_workers=num_workers)
        if self.n_val > 0:
            self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate, num_workers=num_workers)



class DatasetFromSubgraph(Dataset):
    def __init__(
        self,
        data: Data,
        **kwargs
    ) -> None:
        super(DatasetFromSubgraph, self).__init__()
        self._data = data

    @property
    def x(self):
        return self._data.x

    @x.setter
    def x(self, x_value):
        self._data.x = x_value

    @property
    def y(self):
        return self._data.y

    @property
    def edge_index(self):
        return self._data.edge_index

    @edge_index.setter
    def edge_index(self, edge_index_value):
        self._data.edge_index = edge_index_value

    @property
    def raw_text(self):
        return self._data.raw_text

    def __len__(self):
        return len(self._data.ego_indices)

    def __getitem__(self, idx):
        eidx = self._data.ego_indices[idx]
        return eidx



class AmazonProductDataset(NodeToGraphDataset):
    def __init__(
        self,
        dataset,
        sample_size=-1,
        random_sampling=False,
        num_hops=2,
        use_bow=False,
        max_num_frequent_words=None,
        **kwargs
    ) -> None:
        super(NodeToGraphDataset, self).__init__(**kwargs)
        data, _ = self.get_data(dataset, sample_size, random_sampling)
        data.raw_text = data.x
        if use_bow:
            data.x, data.words = self.get_bow_embedding(data, max_num_frequent_words)
        else:
            data.x = data.x_original.float()
            data.words = None
        data.y = data.label_map.long()
        data.label_info = dict(zip([str(i) for i in range(len(data.label))], data.label))
        data.label_names = data.label
        self._data = data
        self.n_feats = data.x.size(1)
        self.num_samples = data.x.size(0)
        self.num_classes = len(data.label)
        self.num_hops = num_hops
        self.name_ = "amazon_products"


    def __len__(self):
        return len(self._data.x)


    def init_ds_from_indices(self, node_indices):

        subset, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx = node_indices,
            num_hops = self.num_hops,
            edge_index = self._data.edge_index,
            relabel_nodes = True,
            num_nodes = self._data.num_nodes,
            flow = 'target_to_source'
        )

        data = Data(
            x = self._data.x[subset],
            y = self._data.y[subset],
            raw_text = [self._data.raw_text[i] for i in subset] if hasattr(self._data, "raw_text") else [],
            edge_index = subgraph_edge_index,
            original_indices = subset,
            ego_indices = mapping
        )

        return DatasetFromSubgraph(data)



class CoraDataset(NodeToGraphDataset):
    def __init__(
        self,
        path="data/data_cora.pt",
        sample_size=-1,
        random_sampling=False,
        num_hops=2,
        use_bow=False,
        max_num_frequent_words=None,
        **kwargs
    ):
        super(NodeToGraphDataset, self).__init__(**kwargs)
        data = torch.load(path)
        # data, indices = self.get_data(data, sample_size, random_sampling) 
        indices = None
        
        if indices:
            for attr in ['words', 'train_mask', 'val_mask', 'test_mask']:
                if hasattr(data, attr):
                    setattr(data, attr, getattr(data, attr)[indices])

        data.label_info = dict(zip([str(i) for i in range(len(data.label_names))], data.label_names))
        data.label_names = data.label_names

        self._data = data
        self.n_feats = data.x.size(1)
        self.num_samples = data.x.size(0)
        self.num_classes = len(data.label_names)
        self.num_hops = num_hops
        self.name_ = "cora"


    def __len__(self):
        return len(self._data.x)


    def init_ds_from_indices(self, node_indices):

        subset, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx = node_indices,
            num_hops = self.num_hops,
            edge_index = self._data.edge_index,
            relabel_nodes = True,
            num_nodes = self._data.num_nodes,
            flow = 'target_to_source'
        )

        data = Data(
            x = self._data.x[subset],
            y = self._data.y[subset],
            raw_text = [self._data.raw_text[i] for i in subset] if hasattr(self._data, "raw_text") else [],
            edge_index = subgraph_edge_index,
            original_indices = subset,
            ego_indices = mapping
        )

        return DatasetFromSubgraph(data)


class WikicsDataset(NodeToGraphDataset):
    def __init__(
        self,
        path="data/data_wikics.pt",
        sample_size=-1,
        random_sampling=False,
        num_hops=2,
        use_bow=False,
        max_num_frequent_words=None,
        **kwargs
    ):
        super(NodeToGraphDataset, self).__init__(**kwargs)
        data = torch.load(path)
        # data, indices = self.get_data(data, sample_size, random_sampling) 
        indices = None

        if indices:
            for attr in ['words']:
                if hasattr(data, attr):
                    setattr(data, attr, getattr(data, attr)[indices])

    
        data.label_info = {'0': 'Computational linguistics',
                             '1': 'Databases',
                             '2': 'Operating systems',
                             '3': 'Computer architecture',
                             '4': 'Computer security',
                             '5': 'Internet protocols',
                             '6': 'Computer file systems',
                             '7': 'Distributed computing architecture',
                             '8': 'Web technology',
                             '9': 'Programming language topics'}
        data.label_names = list(data.label_info)
        data.raw_text = data.words
        self._data = data
        self.n_feats = data.x.size(1)
        self.num_samples = data.x.size(0)
        self.num_classes = len(data.label_names)
        self.num_hops = num_hops
        self.name_ = "wikics"


    def __len__(self):
        return len(self._data.x)


    def init_ds_from_indices(self, node_indices):

        subset, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx = node_indices,
            num_hops = self.num_hops,
            edge_index = self._data.edge_index,
            relabel_nodes = True,
            num_nodes = self._data.num_nodes,
            flow = 'target_to_source'
        )

        data = Data(
            x = self._data.x[subset],
            y = self._data.y[subset],
            raw_text = [self._data.raw_text[i] for i in subset] if hasattr(self._data, "raw_text") else [],
            edge_index = subgraph_edge_index,
            original_indices = subset,
            ego_indices = mapping
        )

        return DatasetFromSubgraph(data)




class LiarDataset(NodeToGraphDataset):
    def __init__(
        self,
        path="data/data_liar.pt",
        sample_size=-1,
        random_sampling=False,
        num_hops=2,
        use_bow=False,
        max_num_frequent_words=None,
        **kwargs
    ):
        super(NodeToGraphDataset, self).__init__(**kwargs)
        data = torch.load(path)
        data.label_names = ["persons_and_places"] + data.label_names
        # data, indices = self.get_data(data, sample_size, random_sampling) 
        indices = None

        if indices:
            for attr in ['words', 'train_mask', 'val_mask', 'test_mask']:
                if hasattr(data, attr):
                    setattr(data, attr, getattr(data, attr)[indices])

    
        data.label_info = dict(zip([str(i) for i in range(len(data.label_names))], data.label_names))
    
        data.raw_text = data.words
        self._data = data
        self.n_feats = data.x.size(1)
        self.num_samples = data.x.size(0)
        self.num_classes = len(data.label_names)
        self.num_hops = num_hops
        self.name_ = "liar"


    def __len__(self):
        return len(self._data.x)


    def init_ds_from_indices(self, node_indices):

        subset, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx = node_indices,
            num_hops = self.num_hops,
            edge_index = self._data.edge_index,
            relabel_nodes = True,
            num_nodes = self._data.num_nodes,
            flow = 'target_to_source'
        )

        data = Data(
            x = self._data.x[subset],
            y = self._data.y[subset],
            raw_text = [self._data.raw_text[i] for i in subset] if hasattr(self._data, "raw_text") else [],
            edge_index = subgraph_edge_index,
            original_indices = subset,
            ego_indices = mapping
        )

        return DatasetFromSubgraph(data)
    
    

def get_node_dataset(
    dataset_name,
    train_test_split = [0.8, 0.2],
    batch_size = 32,
    normal_mode = "normal",
    seed = 2411,
    sample_size = -1,
    random_sampling = True,
    num_workers = 0,
    use_bow = False,
    num_hops = 2,
    max_num_frequent_words = None,
):

    fix_seed(seed)

    if dataset_name == "Flickr":
        dataset = Flickr(
            root = f'./data/{dataset_name}', 
        )
        data = dataset._data
    elif dataset_name == "products":
        dataset = get_dataset(dataset_name)
        dataset = AmazonProductDataset(
            dataset = dataset,
            sample_size = sample_size,
            random_sampling = random_sampling,
            num_hops = num_hops,
            use_bow = use_bow,
            max_num_frequent_words = max_num_frequent_words
        )
    elif dataset_name == "cora":
        dataset = CoraDataset(
            sample_size=sample_size,
            random_sampling=random_sampling,
            num_hops=num_hops,
            use_bow=use_bow,
            max_num_frequent_words=max_num_frequent_words
        )
    elif dataset_name == "wikics":
        dataset = WikicsDataset(
            sample_size=sample_size,
            random_sampling=random_sampling,
            num_hops=num_hops,
            use_bow=use_bow,
            max_num_frequent_words=max_num_frequent_words
        )
    elif dataset_name == "liar":
        dataset = LiarDataset(
            sample_size=sample_size,
            random_sampling=random_sampling,
            num_hops=num_hops,
            use_bow=use_bow,
            max_num_frequent_words=max_num_frequent_words
        )
    else:
        dataset = Planetoid(
            root = f'./data/{dataset_name}',
            name = dataset_name
            )
        data = dataset._data
    
    dataset.initialize(
        train_test_split = train_test_split,
        batch_size = batch_size,
        normalize_mode = normal_mode,
        shuffle = True,
        seed = seed,
        num_workers = num_workers,
    )

    return dataset
