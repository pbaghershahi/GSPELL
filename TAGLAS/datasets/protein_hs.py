import json
from typing import (
    Optional,
    Callable,
    Any
)

import numpy as np
import torch
from torch import Tensor

from TAGLAS.constants import HF_REPO_ID
from TAGLAS.data import TAGData, TAGDataset, BaseDict
from TAGLAS.utils.dataset import generate_link_split
from TAGLAS.utils.graph import safe_to_undirected
from TAGLAS.utils.io import download_hf_file


class ProteinHS(TAGDataset):
    """Cora co-citation network dataset.
    """
    graph_description = "This is a protein-protein interaction network. Nodes represents the proteins and their descriptions, edge represents positive interactions."

    def __init__(self,
                 name: str = "protein_hs",
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 **kwargs,
                 ) -> None:
        super().__init__(name, root, transform, pre_transform, pre_filter, **kwargs)
        # Generate random split for link prediction.
        self.side_data.link_split, self.side_data.keep_edges = generate_link_split(self._data.edge_index, train_ratio= 0.85, test_ratio = 0.1)

    def raw_file_names(self) -> list:
        return ["protein_names.txt", "protein_desc.json"]

    def download(self) -> None:
        download_hf_file(HF_REPO_ID, subfolder="protein_hs", filename="protein_names.txt", local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="protein_hs", filename="protein_desc.json", local_dir=self.raw_dir)

    def gen_data(self) -> tuple[list[TAGData], Any]:
        pnames2id = {}
        edges = []
        count = 0
        with open(self.raw_paths[0], "r") as f:
            lines = f.readlines()
            for line in lines:
                names = line.split(",")
                names = [n.strip() for n in names]
                for n in names:
                    if n not in pnames2id:
                        pnames2id[n] = count
                        count += 1
                edges.append([pnames2id[n] for n in names])
        with open(self.raw_paths[1], "r") as f:
            protein_desc = json.load(f)
        full_desc = [None] * count
        for k in protein_desc:
            if k not in pnames2id:
                new_k = k.split(".")[0]
            else:
                new_k = k
            full_desc[pnames2id[new_k]] = f"Protein description: {protein_desc[k]['desc']}. Protein Comment: {protein_desc[k]['comment']}"
        edge_index = torch.tensor(edges).T
        edge_index, _ = safe_to_undirected(edge_index)

        # add edge text:
        edge_text_lst = ["Connected proteins have positive interaction."]
        edge_map = torch.zeros(edge_index.size(-1), dtype=torch.long)

        data = TAGData(full_desc, edge_index=edge_index, edge_attr=edge_text_lst, edge_map=edge_map)

        data.node_map = torch.arange(len(full_desc), dtype=torch.long)

        # add link prediction label:
        label_names = ["No", "Yes"]

        data.label = label_names

        side_data = BaseDict()

        return [data], side_data

    def get_NP_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the node-level tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """
        mask = self.side_data.node_split[split][0]
        indexs = torch.where(mask)[0]
        labels = self.label_map[indexs]
        label_map = labels
        return indexs, labels, label_map.tolist()

    def get_LP_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the link-level tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """
        indexs, labels = self.side_data.link_split[split]
        label_map = labels
        return indexs, labels, label_map.tolist()

    def get_NQA_list(self, label_map: list, **kwargs) -> tuple[list[list], np.ndarray, np.ndarray]:
        r"""Return question and answer list for node question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        q_list = ["What is the most likely paper category for the paper?"]
        answer_list = []
        label_features = self.label[:7]
        for l in label_map:
            answer_list.append(label_features[l] + ".")
        a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        a_list = a_list.tolist()
        label_map = [[0, l_idx, a_idx] for l_idx, a_idx in zip(label_map, a_idxs)]
        return label_map, q_list, a_list

    def get_LQA_list(self, label_map: list, **kwargs) -> tuple[list[list], np.ndarray, np.ndarray]:
        r"""Return question and answer list for link question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        q_list = ["Does the two proteins have positive interactions? Please answer yes if two proteins have positive interactions and no if two proteins do not have positive interactions."]
        answer_list = []
        label_features = self.label
        for l in label_map:
            answer_list.append(label_features[l] + ".")
        a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        a_list = a_list.tolist()
        label_map = [[0, l_idx, a_idx] for l_idx, a_idx in zip(label_map, a_idxs)]

        return label_map, q_list, a_list
