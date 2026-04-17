import os
import re
import pickle
import json
import torch
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx
import ipdb


def draw_graph_on_axis(ax, data_obj, num_classes, title):
    if not hasattr(data_obj, 'org_nid'):
        data_obj.org_nid = torch.arange(data_obj.num_nodes)

    g = to_networkx(data_obj, to_undirected=True)
    
    node_labels_dict = {i: str(original_id.item()) for i, original_id in enumerate(data_obj.org_nid)}
    node_class_colors = [data.item() for data in data_obj.y]
    cmap = plt.get_cmap('viridis', num_classes)
    
    if hasattr(data_obj, 'node_type'):
        node_types = data_obj.node_type.tolist()
        is_unsupporter = {i: (node_types[i] == 2) for i in range(data_obj.num_nodes)}
        edge_colors = ['red' if is_unsupporter.get(u) or is_unsupporter.get(v) else 'gray' for u, v in g.edges()]
        node_borders = ['red' if node_types[i] == 0 else 'white' for i in range(data_obj.num_nodes)]
        border_widths = [3 if node_types[i] == 0 else 1.5 for i in range(data_obj.num_nodes)]
    else:
        edge_colors = 'gray'
        node_borders = ['red' if i == 0 else 'white' for i in range(data_obj.num_nodes)]
        border_widths = [3 if i == 0 else 1.5 for i in range(data_obj.num_nodes)]

    nx.draw(
        g, ax=ax, labels=node_labels_dict, node_color=node_class_colors, cmap=cmap, edge_color=edge_colors,
        with_labels=True, node_size=800, font_size=12, font_color='black', width=1.5,
        edgecolors=node_borders, linewidths=border_widths
    )
    ax.set_title(title, fontsize=16)


def save_side_by_side_visualization(llm_data_obj, gnn_data_obj, num_classes, output_dir="comparison_visualizations"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    draw_graph_on_axis(axes[0], llm_data_obj, num_classes, "LLM-based Explanation")
    draw_graph_on_axis(axes[1], gnn_data_obj, num_classes, "GNNExplainer Explanation")
    
    target_node_id = llm_data_obj.org_nid[0].item()
    fig.suptitle(f"Comparison for Target Node {target_node_id}", fontsize=20)
    
    filename = os.path.join(output_dir, f"comparison_for_node_{target_node_id}.pdf")
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def parse_explanation(explanation_text, neighbors_in_dict, target_node_id):
    if "Support:" not in explanation_text and "Support :" not in explanation_text:
        return None
    pattern = re.compile(r"(?:\*\*)?(?:Product|Article)\s+(\d+):(?:\*\*)?.*?Support:\s+(YES|NO)", re.DOTALL | re.IGNORECASE)
    all_matches = pattern.findall(explanation_text)
    supporting_neighbors = [int(n_id) for n_id, stat in all_matches if stat.upper() == 'YES' and int(n_id) != target_node_id]
    unsupporting_neighbors = list(set(neighbors_in_dict) - set(supporting_neighbors))
    return {'supporting': supporting_neighbors, 'unsupporting': unsupporting_neighbors}


def process_and_create_subgraphs(file_path, main_graph_edge_index, main_graph_node_labels, main_graph_node_raw_text):
    try:
        with open(file_path, 'rb') as f:
            explanation_data = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error: {e}")
        return [], {}
    num_nodes_in_graph = main_graph_node_labels.size(0)
    flat_data = [item for sublist in explanation_data for item in sublist]
    grouped_data = defaultdict(list)
    for entry in flat_data:
        grouped_data[entry['target_node']].append(entry)
    subgraph_list = []
    final_explanations = {}
    for target_node, dict_list in grouped_data.items():
        all_supporters, all_unsupporters = [], []
        for entry in dict_list:
            result = parse_explanation(entry['explanation'], entry['neighbors'], target_node)
            if result is not None:
                all_supporters.extend(result['supporting'])
                all_unsupporters.extend(result['unsupporting'])
        unique_supporters = sorted(list(set(all_supporters)))
        if unique_supporters:
            final_explanations[target_node] = {}
            final_explanations[target_node]["raw_texts"] = {}
            valid_supporters = [i for i in unique_supporters if i < num_nodes_in_graph]
            unique_unsupporters = sorted(list(set(all_unsupporters) - set(unique_supporters)))
            valid_unsupporters = [i for i in unique_unsupporters if i < num_nodes_in_graph]
            if not valid_supporters:
                continue
            combined_text = "\n\n---\n\n".join([
                entry['explanation'] for entry in dict_list 
                if "Support:" in entry['explanation'] or "Support :" in entry['explanation']
            ])
            final_explanations[target_node]["explanation"] = combined_text
            subset_nodes = torch.tensor([target_node] + valid_supporters + valid_unsupporters, dtype=torch.long)
            node_type = torch.ones(len(subset_nodes), dtype=torch.long)
            node_type[0] = 0
            unsupporter_start_index = 1 + len(valid_supporters)
            node_type[unsupporter_start_index:] = 2
            sub_edge_index, _ = subgraph(subset_nodes, main_graph_edge_index, relabel_nodes=True)
            subgraph_node_labels = main_graph_node_labels[subset_nodes]
            final_explanations[target_node]["raw_texts"] = {
                n_idx.item(): main_graph_node_raw_text[n_idx.item()] for n_idx in subset_nodes
            }
            data_obj = Data(
                edge_index=sub_edge_index, num_nodes=len(subset_nodes),
                y=subgraph_node_labels, org_nid=subset_nodes,
                node_type=node_type
            )
            subgraph_list.append(data_obj)
    return subgraph_list, final_explanations


if __name__ == '__main__':
    processed_data_paths = "/workspace/LOGIC/output/gcn/data.pth"
    dataset = torch.load(processed_data_paths)
    main_edge_index = dataset["dataset"]._data.edge_index
    main_labels = dataset["dataset"]._data.y
    main_raw_text = dataset["dataset"]._data.raw_text
    num_classes = len(torch.unique(main_labels))

    llm_explanations_path = '/workspace/LOGIC/output/gcn/explanations.pth'
    final_llm_subgraphs, filtered_explanations = process_and_create_subgraphs(
        llm_explanations_path, 
        main_edge_index, 
        main_labels,
        main_raw_text
    )

    gnn_explainer_path = "/workspace/LOGIC/output/products-meta-llama/Meta-Llama-3.1-8B-Instruct-2025-09-15-19-17-12/GNNExplainer_explanations.pt"
    gnn_explainer_subgraphs = torch.load(gnn_explainer_path)

    if final_llm_subgraphs:
        print(f"\nSuccessfully identified {len(final_llm_subgraphs)} target node(s) with supporting neighbors.")
        print(f"Saving {len(final_llm_subgraphs)} side-by-side visualizations...")

        for llm_data_obj in final_llm_subgraphs:
            target_node_id = llm_data_obj.org_nid[0].item()
            
            if target_node_id < len(gnn_explainer_subgraphs):
                gnn_data_obj = gnn_explainer_subgraphs[target_node_id]
                save_side_by_side_visualization(llm_data_obj, gnn_data_obj, num_classes)
            else:
                print(f"Warning: No GNNExplainer subgraph found for target node {target_node_id}. Skipping visualization.")
        
        explanation_output_filename = 'filtered_explanations.json'
        with open(explanation_output_filename, 'w') as f:
            json.dump(filtered_explanations, f, indent=4)
        print(f"\nSaved all outputs.")
        
    else:
        print("\nNo target nodes with at least one supporting neighbor were found.")