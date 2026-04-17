import torch, random, os, logging, shutil
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.utils import to_dense_adj, subgraph, remove_self_loops, coalesce
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score
from sklearn.preprocessing import StandardScaler
import networkx as nx
from torch_geometric.utils import to_networkx, subgraph, k_hop_subgraph
from networkx.algorithms.approximation import steiner_tree
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.algorithm import PGExplainer
from torch_geometric.explain.config import ExplanationType, ModelMode, MaskType, ModelConfig
from TAGE.utils import Encoder
from TAGE.downstream import MLP, EndtoEnd, train_MLP
from TAGE.tagexplainer import TAGExplainer, MLPExplainer, XCollector
from utils import *
import ipdb
import pickle
import json
from tqdm import tqdm
import wandb


global_logger = logging.getLogger("global_logger")
eval_logger = logging.getLogger("eval_logger")



def test(model, dataset, validation = True):
    binary_task = False if dataset.num_classes > 2 else True
    f1 = BinaryF1Score() if binary_task else MulticlassF1Score(num_classes=dataset.num_classes, average="macro")
    test_loss, correct = 0, 0
    labels = []
    preds = []
    if validation:
        data_loader = dataset.val_loader
        n_samples = dataset.n_val
    else:
        data_loader = dataset.test_loader
        n_samples = dataset.n_test
    
    eval_ds = data_loader.dataset
    with torch.no_grad():
        model.eval()
        
        for i, batch_idxs in enumerate(data_loader):
            temp_labels = eval_ds.y[batch_idxs].to(model.device)
            test_out, _ = model(
                batch = eval_ds,
                decoder = True,
            )
            test_out = test_out[batch_idxs]
            test_loss += F.cross_entropy(test_out, temp_labels, reduction="sum")
            test_out = F.softmax(test_out, dim=1)
            labels.append(temp_labels)
            preds.append(test_out.argmax(dim=1))
            
        labels = torch.cat(labels)
        preds = torch.cat(preds)
        test_loss /= n_samples
        test_acc = int((labels == preds).sum()) / n_samples
        test_f1 = f1(preds.detach().cpu(), labels.detach().cpu())
    return test_loss, test_acc, test_f1.item()



def get_characterization(dataset, target_node, target_pred):
    if dataset.name_ == "amazon_products":
        system_message = "You are an expert research assistant skilled in reading Amazon product reviews and descriptions and analyzing co-purchase network, i.e. products bought together. Your goal is to assess whether the classification of a target product is supported by its co-purchase neighborhood."
        
        user_intro = f"""You are analyzing Amazon product review and its co-purchase neighborhood to understand why it has been classified under a specific category. Your task is to evaluate the embeddings of each neighboring product and determine whether it supports the classification of the target product. 
    The categories to distinguish from are {", ".join(w for w in dataset._data.label_names)} 
    Target Product ID: {target_node}
    Predicted Category: {target_pred}"""

        node_type = "Product"

    elif dataset.name_ == "liar":
        system_message = "You are an expert research assistant skilled in reading and analyzing political statements. Your goal is to assess whether the classification of a target statement is supported by its context."
        
        user_intro = f"""You are analyzing a statement and its context to understand why it has been classified under a specific category. Your task is to evaluate the embeddings of each contextual element and determine whether it supports the classification of the target statement.
    Target Statement ID: {target_node}
    Predicted Category: {target_pred} 
    """

        node_type = "Statement"

    elif dataset.name_ == "cora":
        system_message = "You are an expert research assistant skilled in reading scientific papers and analyzing citation networks. Your goal is to assess whether the classification of a target paper is supported by its citation neighborhood."
        
        user_intro = f"""You are analyzing a scientific paper and its citation neighborhood to understand why it has been classified under a specific category. Your task is to evaluate the embeddings of each neighboring paper and determine whether it supports the classification of the target paper.
    The categories to distinguish from are {", ".join(w for w in dataset._data.label_names)} 
    Target Paper ID: {target_node}
    Predicted Category: {target_pred} 
    Target Paper Embedding:
    """
        node_type = "Article"

    elif dataset.name_ == "wikics":
        system_message = "You are an expert research assistant skilled in reading scientific article and analyzing citation networks. Your goal is to assess whether the classification of a target article is supported by its citation neighborhood."
        
        user_intro = f"""You are analyzing a scientific article and its citation neighborhood to understand why it has been classified under a specific category. Your task is to evaluate the embeddings of each neighboring article and determine whether it supports the classification of the target paper.
    The categories to distinguish from are {", ".join(w for w in dataset._data.label_names)} 
    Target Article ID: {target_node}
    Predicted Category: {target_pred} 
    """
        node_type = "Article"
    
    else:
        raise Exception("The dataset isn't supported!")


    return system_message, user_intro, node_type



def build_inputs_embeds_vanilla_eos(
    embed_func,
    tokenizer,
    gnn_embeds,
    gnn_preds,
    target_node,
    selected_neighbors,
    dataset,
    with_chat_template,
    descriptive_prompt,
):
    """
    Build input embeddings that alternate between text and soft GNN embeddings.
    """
    device = gnn_embeds.device
    full_embeds = []

    system_message, user_intro, node_type = get_characterization(dataset, target_node, dataset._data.label_info[str(gnn_preds[target_node].item())])

    if with_chat_template:
        prompt_start = tokenizer.apply_chat_template([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_intro}
            ], tokenize=False, add_generation_prompt=False,
        )
    else:
        prompt_start = system_message + user_intro
    tokens_start = tokenizer(prompt_start, return_tensors="pt", add_special_tokens=False).to(device)
    embeds_start = embed_func(tokens_start.input_ids).squeeze(0)
    full_embeds.append(embeds_start)
    
    eos_embed = embed_func(torch.tensor([tokenizer.eos_token_id], device=device))
    full_embeds.append(eos_embed)

    target_text = f"\n\nTarget {node_type} Embedding Representation:\n<<BEGIN TARGET KEYWORDS>>"
    tokens_target_intro = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).to(device)
    embeds_target_intro = embed_func(tokens_target_intro.input_ids).squeeze(0)
    full_embeds.append(embeds_target_intro)
    
    # soft_emb_target = gnn_to_softprompt(gnn_embeddings[target_node].unsqueeze(0)).squeeze(0).to(dtype=torch.bfloat16, device=device)
    full_embeds.append(gnn_embeds[target_node])
    
    tokens_target_outro = tokenizer("<<END TARGET KEYWORDS>>", return_tensors="pt", add_special_tokens=False).to(device)
    embeds_target_outro = embed_func(tokens_target_outro.input_ids).squeeze(0)
    full_embeds.append(embeds_target_outro)
    
    full_embeds.append(eos_embed)

    neighbor_intro = f"\nNeighboring {node_type}s in the Network:\nEach {node_type} below is described by keywords.\n"
    tokens_neighbor_intro = tokenizer(neighbor_intro, return_tensors="pt", add_special_tokens=False).to(device)
    embeds_neighbor_intro = embed_func(tokens_neighbor_intro.input_ids).squeeze(0)
    full_embeds.append(embeds_neighbor_intro)
    
    for neighbor in selected_neighbors:
        neighbor_text = f"\n- {node_type} {neighbor}:\n<<BEGIN KEYWORDS>>"
        tokens_neighbor = tokenizer(neighbor_text, return_tensors="pt", add_special_tokens=False).to(device)
        embeds_neighbor = embed_func(tokens_neighbor.input_ids).squeeze(0)
        full_embeds.append(embeds_neighbor)
    
        # soft_emb_neighbor = gnn_to_softprompt(gnn_embeddings[neighbor].unsqueeze(0)).squeeze(0).to(dtype=torch.bfloat16, device=device)
        full_embeds.append(gnn_embeds[neighbor])
    
        tokens_outro = tokenizer("<<END KEYWORDS>>", return_tensors="pt", add_special_tokens=False).to(device)
        embeds_outro = embed_func(tokens_outro.input_ids).squeeze(0)
        full_embeds.append(embeds_outro)
    
        full_embeds.append(eos_embed)

    desciprtion_req = f"""\n\t- Finally justify your reasoning by **one sentence** for all the nieghboring nodes to **generally** clarify what are the common patterns among the supporting nodes ({node_type})s \
to explain the classification into of the node ({node_type})'{dataset._data.label_info[str(gnn_preds[target_node].item())]}.\n""" if descriptive_prompt else ""

    instructions_text = f"""
        **Instructions:**
        You are given a target node of type {node_type} and a list of neighboring nodes, each described by keywords. Base your reasoning only on the keywords and proximity to the target node.

        For each neighboring node ({node_type}):
        - Read the node's keywords.
        - Think step-by-step and see if the {node_type} is supporting or unsupporting for the prediction.
        - Choose a label 'YES' or 'NO' that whether the neighboring node ({node_type}) supports the classification of the Target {node_type} into category '{dataset._data.label_info[str(gnn_preds[target_node].item())]}'.{desciprtion_req}
        Use the following format for each neighbor:
        
        **Response Format:**\n

        {node_type} <ID1>:
        Support: YES/NO
        
        {node_type} <ID2>:
        Support: YES/NO

        {node_type} <ID3>:
        Support: YES/NO

        ...

        {"Reasoning: ..." if descriptive_prompt else ""}
    """

    # ipdb.set_trace()
    if with_chat_template:
        instructions_prompt = tokenizer.apply_chat_template([
            {"role": "user", "content": instructions_text}
            ], tokenize=False, add_generation_prompt=False,
        )
    else:
        instructions_prompt = instructions_text
    tokens_instructions = tokenizer(instructions_prompt, return_tensors="pt", add_special_tokens=False).to(device)

    embeds_instructions = embed_func(tokens_instructions.input_ids).squeeze(0)
    full_embeds.append(embeds_instructions)
    full_embeds.append(eos_embed)

    # if with_chat_template:
    #     assistant_header = tokenizer.apply_chat_template(
    #         [{"role": "assistant", "content": ""}], tokenize=False, add_generation_prompt=True
    #     )
    #     tokens_assistant = tokenizer(assistant_header, return_tensors="pt", add_special_tokens=False).to(device)
    #     embeds_assistant = embed_func(tokens_assistant.input_ids).squeeze(0)
    #     full_embeds.append(embeds_assistant)

    full_embeds = torch.cat(full_embeds, dim=0).unsqueeze(0)

    return full_embeds



def build_inputs_vanilla(
    gnn_preds,
    target_node,
    selected_neighbors,
    dataset,
    with_chat_template,
    descriptive_prompt
):

    system_message, user_intro, node_type = get_characterization(dataset, target_node, dataset._data.label_info[str(gnn_preds[target_node].item())])
    
    user_intro += f"""\n Target Product Keywords:\n<<BEGIN TARGET KEYWORDS>> {", ".join(dataset._data.words[target_node])} <<END TARGET KEYWORDS>> Neighboring Product in the Co-purchase Network:""" + "\n".join(
            [f"- Product {nid}: <<BEGIN KEYWORDS>>  {' , '.join(dataset._data.words[nid])}  <<END KEYWORDS>> " for nid in selected_neighbors]
        )

    desciprtion_req = f"""\n\Justify your reasoning by **one sentence** for all the nieghboring nodes to **generally** clarify what are the common patterns among the supporting nodes ({node_type})s \
to explain the classification into of the node ({node_type})'{dataset._data.label_info[str(gnn_preds[target_node].item())]}.\n""" if descriptive_prompt else ""

    instructions_text = f"""
        **Instructions:**
        You are given a target node of type {node_type} and a list of neighboring nodes, each described by keywords. Base your reasoning only on the keywords and proximity to the target node.

        For each neighboring node ({node_type}):
        - Read the node's keywords.
        - Think step-by-step and see if the {node_type} is supporting or unsupporting for the prediction.
        - Choose a label 'YES' or 'NO' that whether the neighboring node ({node_type}) supports the classification of the Target {node_type} into category '{dataset._data.label_info[str(gnn_preds[target_node].item())]}'.{desciprtion_req}
        Use the following format for each neighbor:
        
        **Response Format:**\n

        {node_type} <ID1>:
        Support: YES/NO
        
        {node_type} <ID2>:
        Support: YES/NO

        {node_type} <ID3>:
        Support: YES/NO

        ...

        {"Reasoning: ..." if descriptive_prompt else ""}
    """

    if with_chat_template:
        message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_intro + instructions_text}
        ]
    else:
        message = system_message + user_intro + instructions_text

    return message



def generate_exp_by_llm(
    dataset,
    model,
    tokenizer,
    eval_config,
    embed_func = None,
    generate_by = "embedding",
    gnn_embeds = None,
    gnn_preds = None,
    save_to = None,
    save_every = 10,
    with_chat_template = True,
    descriptive_prompt = False
):

    G = to_networkx(dataset._data, to_undirected=True)
    generated_texts = []

    for node_idx in range(eval_config["max_num_eval_nodes"]):
        k_hop_subgraph = nx.ego_graph(G, node_idx, radius=1)
        all_neighbors = set(k_hop_subgraph.nodes)
        all_neighbors.discard(node_idx)
        global_logger.info(
            f"Running for node {node_idx}/{eval_config['max_num_eval_nodes']}"
        )

        selected_neighbors = list(all_neighbors)

        node_texts = []
        for i in range(0, len(selected_neighbors), eval_config["llm_node_batch_size"]):
            neighbor_batch = selected_neighbors[i:i + eval_config["llm_node_batch_size"]]

            if generate_by == "embedding":
                
                full_embeds = build_inputs_embeds_vanilla_eos(
                    embed_func, tokenizer, gnn_embeds, gnn_preds,
                    node_idx, neighbor_batch, dataset, with_chat_template, descriptive_prompt
                )

                attention_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=full_embeds.device)
                max_safe_new_tokens = eval_config["max_position_embeddings"] - full_embeds.shape[1]

                outputs = model.generate(
                    inputs_embeds=full_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=min(max_safe_new_tokens, 1024),
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            else:

                message = build_inputs_vanilla(
                    gnn_preds, node_idx, neighbor_batch, dataset, with_chat_template, descriptive_prompt
                )

                if with_chat_template:
                    check_msg = " ".join([msg["content"] for msg in message])
                else:
                    check_msg = message
                tokenized_prompt = model.tokenizer(check_msg, return_tensors="pt")
                num_prompt_tokens = tokenized_prompt.input_ids.shape[1]
                max_safe_new_tokens = eval_config["max_position_embeddings"] - num_prompt_tokens
                # max_safe_new_tokens = eval_config["max_position_embeddings"]

                if max_safe_new_tokens > 0:
                    outputs = model(
                        message, 
                        max_new_tokens=min(max_safe_new_tokens, 1024), 
                        pad_token_id=tokenizer.eos_token_id
                    )
                else:
                    outputs = []

                try:
                    generated_text = outputs[0]['generated_text']
                except Exception:
                    generated_text = [{"content":"ERROR"}]

            node_texts.append({
                "target_node": node_idx,
                "neighbors": neighbor_batch,
                "explanation": generated_text
            })

        generated_texts.append(node_texts)

        if ((node_idx + 1) % save_every == 0 or (node_idx + 1) == eval_config["max_num_eval_nodes"]) and save_to is not None:

            torch.save(generated_texts, save_to)
            global_logger.info(f"Outputs saved to: {save_to}")

    return generated_texts



def extract_neighbor_support(explanation, node_type, threshold = 11701):
    support = []
    not_support = []
    
    pattern = re.compile(rf"{node_type} (\d+):.*?Support: (YES|NO)", re.DOTALL)
    matches = pattern.findall(explanation)
    
    for paper_id, support_status in matches:
        paper_id = int(paper_id)
        if paper_id > threshold:
            break
        if support_status == "YES":
            support.append(paper_id)
        else:
            not_support.append(paper_id)

    return support, not_support



def get_gnn_prediction_prob(gnn, data, subset_list, target_node_idx, original_pred_class):
    """
    Robust helper to run GNN on a subset.
    Enforces sorting to ensure alignment between x_sub and edge_index.
    """
    if len(subset_list) == 0:
        return 0.0

    subset_sorted = torch.tensor(sorted(list(subset_list)), dtype=torch.long, device=data.x.device)
    
    edge_index_sub, _ = subgraph(subset_sorted, data.edge_index, relabel_nodes=True)
    x_sub = data.x[subset_sorted]
    
    target_mapping = (subset_sorted == target_node_idx).nonzero(as_tuple=True)[0]
    
    if len(target_mapping) == 0:
        # Target node not in subset
        return 0.0
        
    idx_in_batch = target_mapping.item()
    
    exp_graph = Data(
        x=x_sub, 
        edge_index=edge_index_sub, 
        y=data.y[subset_sorted] if data.y is not None else None,
        org_nid=subset_sorted
    )

    with torch.no_grad():
        out, _ = gnn(batch=exp_graph)
        probs = F.softmax(out, dim=-1)
        target_prob = probs[idx_in_batch, original_pred_class].item()
        
    return target_prob



def eval_llm_explanations(
    dataset, 
    gnn,
    gnn_preds,
    generated_exps = None,
    generated_exps_path = "",
    generated_with_pipeline = False,
    num_eval_samples = 1,
    device="cpu"
):
    if generated_exps is None and generated_exps_path:
        with open(generated_exps_path, "rb") as f:
            generated_exps = torch.load(f)
    
    data = dataset._data.to(device)
    gnn.eval()

    _, _, node_type = get_characterization(dataset, 0, 0)

    parsed_explanations = []
    for i, batch in enumerate(generated_exps):
        support_papers = [i] 
        not_support_papers = [] 
        
        for entry in batch:
            explanation = entry['explanation']
            if not isinstance(explanation, str):
                explanation = explanation[-1]['content']

            neighbors = entry['neighbors']
            supports, not_supports = extract_neighbor_support(explanation, node_type, threshold=dataset.num_samples)
            
            supports = [s for s in supports if s in neighbors]
            not_supports = [ns for ns in not_supports if ns in neighbors]
            
            support_papers.extend(supports)
            not_support_papers.extend(not_supports)
            
        parsed_explanations.append((support_papers, not_support_papers))
    
    G = to_networkx(data, to_undirected=True)
    
    metrics = {
        "fidelity_scores": [],
        "prob_orig": [],
        "prob_suff": [],
        "prob_comp": [],
        "prob_rand": [],
        "prob_align": [],
        "sizes": []
    }

    for i in tqdm(range(num_eval_samples), desc="Evaluating GSPELL"):
        ego_graph = nx.ego_graph(G, i, radius=2)
        all_neighbors = set(ego_graph.nodes)
        
        supports, not_supports = parsed_explanations[i]
        s_plus_set = set([n for n in supports if n in all_neighbors])
        s_minus_set = set([n for n in not_supports if n in all_neighbors])
        
        subset_orig = list(all_neighbors)
        orig_pred_class = gnn_preds[i].item()
        p_orig = get_gnn_prediction_prob(gnn, data, subset_orig, i, orig_pred_class)
        
        s_list = list(s_plus_set)
        try:
            selected_nodes = list(steiner_tree(ego_graph, s_list).nodes)
        except:
            selected_nodes = []
        if len(selected_nodes) == 0:
            selected_nodes = s_list
            
        subset_suff = selected_nodes
        metrics["sizes"].append(len(subset_suff))
        
        # --- Binary Fidelity ---
        subset_sorted = torch.tensor(sorted(subset_suff), dtype=torch.long, device=device)
        edge_index_sub, _ = subgraph(subset_sorted, data.edge_index, relabel_nodes=True)
        
        target_idx = (subset_sorted == i).nonzero(as_tuple=True)[0].item()
        
        with torch.no_grad():
            out, _ = gnn(batch=Data(x=data.x[subset_sorted], edge_index=edge_index_sub))
            out_class = out.argmax(dim=-1)
            metrics["fidelity_scores"].append(out_class[target_idx].item() == orig_pred_class)
        
        p_suff = get_gnn_prediction_prob(gnn, data, subset_suff, i, orig_pred_class)

        # --- C. Comprehensiveness ---
        nodes_to_remove = s_plus_set - {i}
        subset_comp = list(all_neighbors - nodes_to_remove)
        p_comp = get_gnn_prediction_prob(gnn, data, subset_comp, i, orig_pred_class)

        # --- D. Random Baseline ---
        num_to_remove = len(nodes_to_remove)
        candidates_to_remove = list(all_neighbors - {i}) 
        if len(candidates_to_remove) >= num_to_remove:
            random_remove = set(random.sample(candidates_to_remove, num_to_remove))
            subset_rand = list(all_neighbors - random_remove)
        else:
            subset_rand = list(all_neighbors) 
        
        p_rand = get_gnn_prediction_prob(gnn, data, subset_rand, i, orig_pred_class)

        # --- E. Alignment ---
        nodes_to_remove_align = s_minus_set - {i}
        subset_align = list(all_neighbors - nodes_to_remove_align)
        p_align = get_gnn_prediction_prob(gnn, data, subset_align, i, orig_pred_class)

        metrics["prob_orig"].append(p_orig)
        metrics["prob_suff"].append(p_suff)
        metrics["prob_comp"].append(p_comp)
        metrics["prob_rand"].append(p_rand)
        metrics["prob_align"].append(p_align)

    output_dict = dict(
        avg_fid = np.mean(metrics["fidelity_scores"]),
        avg_size = np.mean(metrics["sizes"]),
        drop_comp = np.mean(np.array(metrics["prob_orig"]) - np.array(metrics["prob_comp"])),
        drop_rand = np.mean(np.array(metrics["prob_orig"]) - np.array(metrics["prob_rand"])),
        drop_align = np.mean(np.array(metrics["prob_orig"]) - np.array(metrics["prob_align"])),
    )

    eval_logger.info(f"--- Evaluation Results for {dataset.name_} ---")
    eval_logger.info(f"Fidelity (Sufficiency): {output_dict['avg_fid']:.4f}")
    eval_logger.info(f"Explanation Size: {output_dict['avg_size']:.4f}")
    eval_logger.info(f"Comprehensiveness (Prob Drop removing Expl): {output_dict['drop_comp']:.4f}")
    eval_logger.info(f"Random Baseline (Prob Drop removing Random): {output_dict['drop_rand']:.4f}")
    eval_logger.info(f"Alignment Check (Prob Drop removing 'NO' nodes): {output_dict['drop_align']:.4f}")
    
    return output_dict



def eval_gnnexplainer_explanations(
    dataset, 
    gnn,
    gnn_preds,
    num_eval_samples = 1,
    device="cpu",
    num_explainer_epochs = 200,
    save_to = None
):
    data = dataset._data.to(device)
    gnn = gnn.to(device)
    gnn.eval()
    
    gnn_explainer = Explainer(
        model=gnn,
        algorithm=GNNExplainer(epochs=num_explainer_epochs).to(device),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    G = to_networkx(data, to_undirected=True)

    metrics = {
        "fidelity": [],
        "size": [],
        "prob_orig": [],
        "prob_comp": [],
        "prob_rand": []
    }
    explaining_nodes = dict()

    for i in tqdm(range(num_eval_samples), desc="Evaluating GNNExplainer"):
        graph = nx.ego_graph(G, i, radius=2)
        all_neighbors = set(graph.nodes)

        explanation = gnn_explainer(
            x=data.x, 
            edge_index=data.edge_index,
            index=i,
            output_embeds=False
        )

        node_scores = explanation.node_mask.abs().sum(dim=1)
        sorted_indices = node_scores.argsort(descending=True)
        
        selected_nodes = [j.item() for j in sorted_indices if j.item() in all_neighbors]
        # k = max(1, len(all_neighbors) // 2)
        # selected_nodes = important_neighbors[:k]
        explaining_nodes[i] = ",".join([str(s_idx) for s_idx in selected_nodes if s_idx != i])
        
        if i not in selected_nodes:
            selected_nodes = [i] + selected_nodes
        
        try:
            s_plus = list(steiner_tree(graph, selected_nodes).nodes)
        except:
            s_plus = selected_nodes
        if len(s_plus) == 0: s_plus = selected_nodes
        
        # --- Fidelity ---
        if i not in s_plus: s_plus.append(i)
        subset_sorted = torch.tensor(sorted(s_plus), dtype=torch.long, device=device)
        edge_sub, _ = subgraph(subset_sorted, data.edge_index, relabel_nodes=True)
        target_idx = (subset_sorted == i).nonzero(as_tuple=True)[0].item()
        orig_pred = gnn_preds[i].item()
        
        with torch.no_grad():
            out, _ = gnn(batch=Data(x=data.x[subset_sorted], edge_index=edge_sub))
            out_class = out.argmax(dim=-1)
            metrics["fidelity"].append(out_class[target_idx].item() == orig_pred)
        
        metrics["size"].append(len(s_plus))
        
        # --- Comprehensiveness ---
        p_orig = get_gnn_prediction_prob(gnn, data, list(all_neighbors), i, orig_pred)
        
        s_plus_set = set(s_plus)
        nodes_to_remove = s_plus_set - {i}
        subset_comp = list(all_neighbors - nodes_to_remove)
        p_comp = get_gnn_prediction_prob(gnn, data, subset_comp, i, orig_pred)
        
        # --- Random Baseline ---
        num_to_remove = len(nodes_to_remove)
        candidates = list(all_neighbors - {i})
        if len(candidates) >= num_to_remove:
            random_remove = set(random.sample(candidates, num_to_remove))
            subset_rand = list(all_neighbors - random_remove)
        else:
            subset_rand = list(all_neighbors)
        p_rand = get_gnn_prediction_prob(gnn, data, subset_rand, i, orig_pred)
        
        metrics["prob_orig"].append(p_orig)
        metrics["prob_comp"].append(p_comp)
        metrics["prob_rand"].append(p_rand)

    if save_to is not None:
        with open(save_to, 'w') as json_file:
            json.dump(explaining_nodes, json_file, indent=4)

    output_dict = dict(
        avg_fid = np.mean(metrics["fidelity"]),
        avg_size = np.mean(metrics["size"]),
        drop_comp = np.mean(np.array(metrics["prob_orig"]) - np.array(metrics["prob_comp"])),
        drop_rand = np.mean(np.array(metrics["prob_orig"]) - np.array(metrics["prob_rand"])),
    )

    eval_logger.info(f"--- Evaluation Results for {dataset.name_} ---")
    eval_logger.info(f"Fidelity (Sufficiency): {output_dict['avg_fid']:.4f}")
    eval_logger.info(f"Explanation Size: {output_dict['avg_size']:.4f}")
    eval_logger.info(f"Comprehensiveness (Prob Drop removing Expl): {output_dict['drop_comp']:.4f}")
    eval_logger.info(f"Random Baseline (Prob Drop removing Random): {output_dict['drop_rand']:.4f}")

    return output_dict



def eval_tage_explanations(
    dataset, 
    gnn,
    gnn_preds,
    exp_config,
    num_eval_samples = 1,
    device="cpu",
    num_explainer_epochs = 200,
    top_k=100
):
    data = dataset._data.to(device)
    gnn = gnn.to(device)
    gnn.eval()
    gnn_preds = gnn_preds.to(device)

    enc_explainer = TAGExplainer(
        gnn, embed_dim=exp_config["gnn_h_dim"], device=device, 
        explain_graph=False, grad_scale=0.1, coff_size=0.05, coff_ent=0.002, loss_type='JSE'
    )
    enc_explainer.train_explainer_node(dataset.train_loader, batch_size=64, lr=5e-5, epochs=1)

    mlp_model = MLP(num_layer=2, emb_dim=exp_config["gnn_h_dim"], hidden_dim=600, out_dim=dataset.num_classes).to(device)
    mlp_model = train_MLP(gnn, mlp_model, dataset.train_loader, dataset.val_loader)
    mlp_explainer = MLPExplainer(mlp_model, device)

    G = to_networkx(data, to_undirected=True)
    metrics = {
        "fidelity": [],
        "size": [],
        "prob_orig": [],
        "prob_comp": []
    }

    all_nodes = data.edge_index.unique().tolist()
    
    for i in range(num_eval_samples):
        if i not in all_nodes: continue
        
        masked_data, subset, masked_subset, ego_mapping = enc_explainer(
            data, mlp_explainer, node_idx=i, top_k=top_k
        )
        orig_pred = gnn_preds[i].item()
        
        # TAGE uses masked_subset as important nodes. 
        s_nodes = list(masked_subset)
        
        # Consistency Logic
        ego_graph = nx.ego_graph(G, i, radius=2)
        all_neighbors = set(ego_graph.nodes)
        
        s_nodes = [n for n in s_nodes if n in all_neighbors]
        if i not in s_nodes: s_nodes.append(i)
        
        try:
            s_plus = list(steiner_tree(ego_graph, s_nodes).nodes)
        except:
            s_plus = s_nodes
        if len(s_plus) == 0: s_plus = s_nodes
        
        # --- Fidelity ---
        subset_sorted = torch.tensor(sorted(s_plus), dtype=torch.long, device=device)
        edge_sub, _ = subgraph(subset_sorted, data.edge_index, relabel_nodes=True)
        target_idx = (subset_sorted == i).nonzero(as_tuple=True)[0].item()
        
        with torch.no_grad():
            out, _ = gnn(batch=Data(x=data.x[subset_sorted], edge_index=edge_sub))
            out_class = out.argmax(dim=-1)
            metrics["fidelity"].append(out_class[target_idx].item() == orig_pred)
            
        metrics["size"].append(len(s_plus))
        
        # --- Comprehensiveness ---
        p_orig = get_gnn_prediction_prob(gnn, data, list(all_neighbors), i, orig_pred)
        
        nodes_to_remove = set(s_plus) - {i}
        subset_comp = list(all_neighbors - nodes_to_remove)
        p_comp = get_gnn_prediction_prob(gnn, data, subset_comp, i, orig_pred)
        
        metrics["prob_orig"].append(p_orig)
        metrics["prob_comp"].append(p_comp)

    output_dict = dict(
        avg_fid = np.mean(metrics["fidelity"]),
        avg_size = np.mean(metrics["size"]),
        drop_comp = np.mean(np.array(metrics["prob_orig"]) - np.array(metrics["prob_comp"])),
    )

    eval_logger.info(f"--- Evaluation Results for {dataset.name_} ---")
    eval_logger.info(f"Fidelity (Sufficiency): {output_dict['avg_fid']:.4f}")
    eval_logger.info(f"Explanation Size: {output_dict['avg_size']:.4f}")
    eval_logger.info(f"Comprehensiveness (Prob Drop removing Expl): {output_dict['drop_comp']:.4f}")

    return output_dict



def eval_pgexplainer_explanations(
    dataset, 
    gnn,
    gnn_preds,
    num_eval_samples = 1,
    device="cpu",
    num_explainer_epochs = 200,
):
    data = dataset._data.to(device)
    gnn = gnn.to(device)
    gnn.eval()
    gnn_preds = gnn_preds.to(device)
    
    explainer = Explainer(
        model=gnn,
        algorithm=PGExplainer(epochs=num_explainer_epochs, lr=0.003).to(device),
        explanation_type=ExplanationType.phenomenon,
        edge_mask_type=MaskType.object,
        model_config=ModelConfig(
            mode=ModelMode.multiclass_classification,
            task_level='node',
            return_type='log_probs',
        )
    )

    for epoch in range(num_explainer_epochs):
        for k, idx in enumerate(dataset.train_idxs):
            if k > 0: break
            loss = explainer.algorithm.train(
                epoch=epoch, model=gnn, x=data.x, edge_index=data.edge_index,
                target=gnn_preds, index=torch.tensor([idx]).to(device), output_embeds=False
            )

    G = to_networkx(data, to_undirected=True)
    metrics = {
        "fidelity": [],
        "size": [],
        "prob_orig": [],
        "prob_comp": []
    }

    for i in range(num_eval_samples):
        orig_pred = gnn_preds[i].item()
        explanation = explainer(x=data.x, edge_index=data.edge_index, index=i, target=gnn_preds[i], output_embeds=False)
        edge_mask = explanation.edge_mask
        edge_index_all = explanation.edge_index
        
        subset_indices, _, _, _ = k_hop_subgraph(i, 2, data.edge_index, relabel_nodes=False, num_nodes=dataset.num_samples)
        ego_nodes = set(subset_indices.tolist())
        mask_in_ego = [j for j, (u, v) in enumerate(edge_index_all.t().tolist()) if u in ego_nodes and v in ego_nodes]
        if not mask_in_ego: continue
        
        edge_mask_filtered = edge_mask[mask_in_ego]
        edge_index_filtered = edge_index_all[:, mask_in_ego]
        k_edges = max(1, int(edge_mask_filtered.size(0) * 0.1))
        top_indices = edge_mask_filtered.topk(k_edges).indices
        edge_index_topk = edge_index_filtered[:, top_indices]
        nodes_in_subgraph = torch.unique(edge_index_topk)
        subset_expl = [i] + [n.item() for n in nodes_in_subgraph if n.item() != i]
        
        ego_graph = nx.ego_graph(G, i, radius=2)
        try:
            s_plus = list(steiner_tree(ego_graph, subset_expl).nodes)
        except:
            s_plus = subset_expl
        if len(s_plus) == 0: s_plus = subset_expl
        
        # --- Fidelity ---
        subset_sorted = torch.tensor(sorted(s_plus), dtype=torch.long, device=device)
        edge_sub, _ = subgraph(subset_sorted, data.edge_index, relabel_nodes=True)
        target_idx = (subset_sorted == i).nonzero(as_tuple=True)[0].item()
        
        with torch.no_grad():
            out, _ = gnn(batch=Data(x=data.x[subset_sorted], edge_index=edge_sub))
            out_class = out.argmax(dim=-1)
            metrics["fidelity"].append(out_class[target_idx].item() == orig_pred)
            
        metrics["size"].append(len(s_plus))
        
        # --- Comprehensiveness ---
        p_orig = get_gnn_prediction_prob(gnn, data, list(ego_nodes), i, orig_pred)
        
        nodes_to_remove = set(s_plus) - {i}
        subset_comp = list(ego_nodes - nodes_to_remove)
        p_comp = get_gnn_prediction_prob(gnn, data, subset_comp, i, orig_pred)
        
        metrics["prob_orig"].append(p_orig)
        metrics["prob_comp"].append(p_comp)

    output_dict = dict(
        avg_fid = np.mean(metrics["fidelity"]),
        avg_size = np.mean(metrics["size"]),
        drop_comp = np.mean(np.array(metrics["prob_orig"]) - np.array(metrics["prob_comp"])),
    )

    eval_logger.info(f"--- Evaluation Results for {dataset.name_} ---")
    eval_logger.info(f"Fidelity (Sufficiency): {output_dict['avg_fid']:.4f}")
    eval_logger.info(f"Explanation Size: {output_dict['avg_size']:.4f}")
    eval_logger.info(f"Comprehensiveness (Prob Drop removing Expl): {output_dict['drop_comp']:.4f}")

    return output_dict



def eval_gnn_explanations(
    dataset, 
    gnn,
    gnn_preds,
    method,
    num_eval_samples = 1,
    device="cpu",
):
    data = dataset._data.to(device)
    gnn = gnn.to(device)
    gnn.eval()

    G = to_networkx(data, to_undirected=True)
    
    metrics = {
        "fidelity": [],
        "size": [],
        "drop_comp": []
    }

    for _ in range(5): 
        trial_metrics = {
            "fidelity": [],
            "size": [],
            "prob_orig": [],
            "prob_comp": []
        }
    
        for i in range(num_eval_samples):
            orig_pred = gnn_preds[i].item()
            ego_graph = nx.ego_graph(G, i, radius=2)
            all_neighbors = set(ego_graph.nodes)
            neighbors_only = all_neighbors.copy()
            neighbors_only.discard(i)

            if len(neighbors_only) == 0: continue

            if method == "random":
                s = random.sample(list(neighbors_only), int(len(neighbors_only)/2))
                s = [i] + s
                selected_nodes = list(steiner_tree(ego_graph, s).nodes)
                if not selected_nodes: selected_nodes = s
            elif method == "node":
                selected_nodes = [i]
            else:
                raise NotImplementedError

            s_plus = selected_nodes
            
            # --- Fidelity ---
            subset_sorted = torch.tensor(sorted(s_plus), dtype=torch.long, device=device)
            edge_sub, _ = subgraph(subset_sorted, data.edge_index, relabel_nodes=True)
            target_idx = (subset_sorted == i).nonzero(as_tuple=True)[0].item()
            
            with torch.no_grad():
                out, _ = gnn(batch=Data(x=data.x[subset_sorted], edge_index=edge_sub))
                out_class = out.argmax(dim=-1)
                trial_metrics["fidelity"].append(out_class[target_idx].item() == orig_pred)
            
            trial_metrics["size"].append(len(s_plus))
            
            # --- Comprehensiveness ---
            p_orig = get_gnn_prediction_prob(gnn, data, list(all_neighbors), i, orig_pred)
            
            nodes_to_remove = set(s_plus) - {i}
            subset_comp = list(all_neighbors - nodes_to_remove)
            p_comp = get_gnn_prediction_prob(gnn, data, subset_comp, i, orig_pred)
            
            trial_metrics["prob_orig"].append(p_orig)
            trial_metrics["prob_comp"].append(p_comp)

        metrics["fidelity"].append(np.mean(trial_metrics["fidelity"]))
        metrics["size"].append(np.mean(trial_metrics["size"]))
        metrics["drop_comp"].append(
            np.mean(np.array(trial_metrics["prob_orig"]) - np.array(trial_metrics["prob_comp"]))
        )

    output_dict = dict(
        avg_fid = np.mean(metrics["fidelity"]),
        avg_size = np.mean(metrics["size"]),
        drop_comp = np.mean(metrics["drop_comp"]),
    )

    eval_logger.info(f"--- Evaluation Results for {dataset.name_} ---")
    eval_logger.info(f"Fidelity (Sufficiency): {output_dict['avg_fid']:.4f}")
    eval_logger.info(f"Explanation Size: {output_dict['avg_size']:.4f}")
    eval_logger.info(f"Comprehensiveness (Prob Drop removing Expl): {output_dict['drop_comp']:.4f}")

    return output_dict