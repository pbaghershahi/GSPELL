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
from tqdm import tqdm


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

    desciprtion_req = f"""\n\t3. For better understanding, justify your reasoning by **one sentence** to **generally** clarify what are the common patterns among the supporting {node_type}s \
to explain the classification into '{dataset._data.label_info[str(gnn_preds[target_node].item())]}. Give this reasoning after marking all neighbors at the end by Reasoning:<YOURE_REASONING_GOES_HERE>'. \n""" if descriptive_prompt else "\n"

    instructions_text = f"""
        Instructions:
        You are given a target {node_type} and a list of neighboring {node_type}s, each described by keywords.
        
        For each neighboring {node_type}:
        1. Write **one sentence** summarizing the main topics or ideas captured in its keywords.
        2. Clearly state whether this {node_type} supports the classification of the Target {node_type} into category '{dataset._data.label_info[str(gnn_preds[target_node].item())]}'.{desciprtion_req}
        Use the following format for each neighbor:
        
        {node_type} <ID>:
        Summary: <One sentence summary of the {node_type}'s keywords>.
        Support: YES or NO — Does this {node_type} support the classification into '{dataset._data.label_info[str(gnn_preds[target_node].item())]}'?
        
        Base your reasoning only on the keywords and proximity to the target {node_type}.
        
        Start your analysis below:
    """
    tokens_instructions = tokenizer(instructions_text, return_tensors="pt", add_special_tokens=False).to(device)
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
    with_chat_template
):

    system_message = "You are an expert research assistant skilled in reading Amazon product reviews and descriptions and analyzing co-purchase network, i.e. products bought together. Your goal is to assess whether the classification of a target product is supported by its co-purchase neighborhood."
    
    user_intro = f"""You are analyzing Amazon product review and its co-purchase neighborhood to understand why it has been classified under a specific category. Your task is to evaluate the embeddings of each neighboring product and determine whether it supports the classification of the target product. 
The categories to distinguish from are {", ".join(w for w in dataset._data.label_names)} 
Target Product ID: {target_node}
Predicted Category: {dataset._data.label_info[str(gnn_preds[target_node].item())]} 

Target Product Keywords:\n<<BEGIN TARGET KEYWORDS>> {", ".join(dataset._data.words[target_node])} <<END TARGET KEYWORDS>>

Neighboring Product in the Co-purchase Network:
""" + "\n".join(
            [f"- Product {nid}: <<BEGIN KEYWORDS>>  {' , '.join(dataset._data.words[nid])}  <<END KEYWORDS>> " for nid in selected_neighbors]
        ) + f"""

    Instructions:
    You are given a target product and a list of neighboring products, each described by keywords.
    
    For each neighboring product:
    1. Write **one sentence** summarizing the main topics or ideas captured in its keywords.
    2. Clearly state whether this product supports the classification of the Target Product into category '{dataset._data.label_info[str(gnn_preds[target_node].item())]}'.
    
    Use the following format for each neighbor:
    
    Product <ID>:
    Summary: <One sentence summary of the product's keywords>.
    Support: YES or NO — Does this product support the classification into '{dataset._data.label_info[str(gnn_preds[target_node].item())]}'?
    
    Base your reasoning only on the keywords and proximity to the target product.
    
    Start your analysis below:
    """

    if with_chat_template:
        message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_intro}
        ]
    else:
        message = system_message + user_intro

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
                    gnn_preds, node_idx, neighbor_batch, dataset, with_chat_template
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

    _, _, node_type = get_characterization(dataset, 0, 0)

    explanations = []
    for i, batch in enumerate(generated_exps):
        support_papers = [i]
        not_support_papers = []
        hallucinate_papers = []
    
        for entry in batch:

            explanation = entry['explanation']
            if not isinstance(explanation, str):
                explanation = explanation[-1]['content']

            neighbors = entry['neighbors']
    
            # Get support info from explanation
            supports, not_supports = extract_neighbor_support(explanation, node_type, threshold=dataset.num_samples)
    
            supports = [s for s in supports if s in neighbors]
            not_supports = [ns for ns in not_supports if ns in neighbors]
            found = supports + not_supports
    
            # Add parsed results
            support_papers.extend(supports)
            not_support_papers.extend(not_supports)
    
            # Check for hallucinated neighbors (those not found in explanation)
            hallucinate = [n for n in neighbors if n not in found]
            hallucinate_papers.extend(hallucinate)
    
        explanations.append((support_papers, not_support_papers, hallucinate_papers))
    
    G = to_networkx(data, to_undirected=True)
    
    all_fidelities = []
    all_sizes = []
    exp_graphs = []
    for i in range(num_eval_samples):
        graph = nx.ego_graph(G, i, radius=2)
        a, b, c = explanations[i]
        s = list(set(a))
        selected_nodes = list(steiner_tree(graph, s).nodes)
        subset = torch.tensor(selected_nodes, dtype=torch.long)
        
        if len(selected_nodes) == 0:
            selected_nodes = s
        all_sizes.append(len(selected_nodes))

        subset = torch.tensor(selected_nodes, dtype=torch.long)
        d, e = subgraph(subset, data.edge_index, relabel_nodes=True)
            
        x_sub = data.x[subset]
        x_pred = gnn_preds[subset]

        gnn.eval()
        exp_graph = Data(
            x=x_sub, 
            edge_index=d, 
            y=data.y[subset],
            org_nid=subset,
            raw_text=[data.raw_text[n_idx.item()] for n_idx in subset] if len(data.raw_text) != 0 else []
        )
        exp_graphs.append(exp_graph)

        with torch.no_grad():
            out, _ = gnn(batch = exp_graph)
            out = out.argmax(dim=-1)
            all_fidelities.append(out[0] == x_pred[0])
    
    all_sizes = torch.as_tensor(all_sizes, dtype=torch.float)
    all_fidelities = torch.stack(all_fidelities)
    
    eval_logger.info(f"Evalaution on {dataset.name_}: Test Fidelity: {all_fidelities.sum()/all_fidelities.size(0):.3f} -- Test Size: {all_sizes.mean():.3f}")



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
    gnn.device = device
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

    all_fidelities = []
    all_sizes = []
    exp_graphs = []
    for i in tqdm(range(num_eval_samples)):
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

        important_neighbors = [j.item() for j in sorted_indices if j.item() in all_neighbors]
        k = len(all_neighbors) // 2
        selected_nodes = important_neighbors[:k]
        
        # Make sure target node is first in the list
        selected_nodes = [i] + [x for x in selected_nodes if x != i]
        all_sizes.append(len(selected_nodes))

        # if len(selected_nodes) > max_expln_subgraph_size:
        #     selected_nodes = selected_nodes[:max_expln_subgraph_size]

        subset = torch.tensor(selected_nodes, dtype=torch.long, device=data.x.device)
        d, e = subgraph(subset, data.edge_index, relabel_nodes=True)
            
        x_sub = data.x[subset]
        x_pred = gnn_preds.to(data.x.device)[subset]

        gnn.eval()
        exp_graph = Data(
            x=x_sub, 
            edge_index=d, 
            y=data.y[subset],
            org_nid=subset,
            raw_text=[data.raw_text[n_idx.item()] for n_idx in subset] if len(data.raw_text) != 0 else []
        )
        exp_graphs.append(exp_graph)

        with torch.no_grad():
            out, _ = gnn(batch = exp_graph)
            out = out.argmax(dim=-1)
            all_fidelities.append(out[0] == x_pred[0])

    all_sizes = torch.as_tensor(all_sizes, dtype=torch.float)
    all_fidelities = torch.stack(all_fidelities)

    if save_to is not None:
        torch.save(exp_graphs, save_to)
        global_logger.info(f"Outputs saved to: {save_to}")
    
    eval_logger.info(f"Evalaution on {dataset.name_}: Test Fidelity: {all_fidelities.sum()/all_fidelities.size(0):.3f} -- Test Size: {all_sizes.mean():.3f}")



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
    gnn.device = device
    gnn_preds = gnn_preds.to(device)

    enc_explainer = TAGExplainer(
        gnn, 
        embed_dim=exp_config["gnn_h_dim"], 
        device=device, 
        explain_graph=False, 
        grad_scale=0.1, 
        coff_size=0.05, 
        coff_ent=0.002, 
        loss_type='JSE'
    )

    enc_explainer.train_explainer_node(dataset.train_loader, batch_size=64, lr=5e-5, epochs=1)

    mlp_model = MLP(
        num_layer=2, 
        emb_dim = exp_config["gnn_h_dim"], 
        hidden_dim = 600, 
        out_dim = dataset.num_classes
    ).to(device)
    mlp_model = train_MLP(gnn, mlp_model, dataset.train_loader, dataset.val_loader)
    mlp_explainer = MLPExplainer(mlp_model, device)

    # x_collector = XCollector()
    # train_ds = dataset.train_loader.dataset
    # for i, batch_idxs in enumerate(dataset.train_loader):
    #     batch_idxs = batch_idxs.to(device)
    #     for node_idx in batch_idxs:
            # walks, masks, related_preds = enc_explainer(
            #     train_ds, 
            #     mlp_explainer, 
            #     node_idx=node_idx, 
            #     top_k=top_k
            # )
    #         fidelity = related_preds[0]['origin'] - related_preds[0]['maskout']

    #         print(f'explain graph {i} node {node_idx}'+' fidelity %.4f'%fidelity, end='\r')
    #         x_collector.collect_data(masks, related_preds)

    # fid, fid_std = x_collector.fidelity
    # spa, spa_std = x_collector.sparsity

    # print(f'Fidelity: {fid:.4f} ±{fid_std:.4f}\nSparsity: {spa:.4f} ±{spa_std:.4f}')
    all_nodes = data.edge_index.unique()
    all_fidelities = []
    all_sizes = []
    for i in range(num_eval_samples):
        if i not in all_nodes:
            continue

        masked_data, subset, masked_subset, ego_mapping = enc_explainer(
            data, 
            mlp_explainer, 
            node_idx=i, 
            top_k=top_k
        )

        all_sizes.append(len(masked_subset))

        x_pred = gnn_preds[subset]

        gnn.eval()
        with torch.no_grad():
            out, _ = gnn(batch = masked_data)
            out = out.argmax(dim=-1)
            all_fidelities.append(out[ego_mapping] == x_pred[ego_mapping])
    
    all_sizes = torch.as_tensor(all_sizes, dtype=torch.float)
    all_fidelities = torch.stack(all_fidelities)
    
    eval_logger.info(f"Evalaution on {dataset.name_}: Test Fidelity: {all_fidelities.sum()/all_fidelities.size(0):.3f} -- Test Size: {all_sizes.mean():.3f}")
    


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
    gnn.device = device
    data = data.to(device)
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
        print(f"Traing Explainer -- Epoch: {epoch}/{num_explainer_epochs}")
        for k, idx in enumerate(dataset.train_idxs):
            if k > 0:
                break
            loss = explainer.algorithm.train(
                epoch=epoch,
                model=gnn,
                x=data.x,
                edge_index=data.edge_index,
                target=gnn_preds,
                index=torch.tensor([idx]).to(device),
                output_embeds=False
            )

    all_fidelities = []
    all_sizes = []

    for i in range(num_eval_samples):

        global_logger.info(
            f"Running for node {i}/{num_eval_samples}"
        )

        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            index=i,
            target=gnn_preds[i],
            output_embeds=False
        )
        edge_mask = explanation.edge_mask
        edge_index_all = explanation.edge_index
        subset, _, _, _ = k_hop_subgraph(
            node_idx=i,
            num_hops=2,
            edge_index=data.edge_index,
            relabel_nodes=False,
            num_nodes=dataset.num_samples
        )
        ego_nodes = set(subset.tolist())
        mask_in_ego = [
            j for j, (u, v) in enumerate(edge_index_all.t().tolist())
            if u in ego_nodes and v in ego_nodes
        ]
        if not mask_in_ego:
            continue  # skip if no edges in 2-hop
        edge_index_filtered = edge_index_all[:, mask_in_ego]
        edge_mask_filtered = edge_mask[mask_in_ego]
        k = max(1, int(edge_mask_filtered.size(0) * 0.1))  ## thresholding for size is here
        top_indices = edge_mask_filtered.topk(k).indices
        edge_index_topk = edge_index_filtered[:, top_indices]
        nodes_in_subgraph = torch.unique(edge_index_topk)
        subset_expl = [i] + [n.item() for n in nodes_in_subgraph if n.item() != i]
        subset = torch.tensor(subset_expl, dtype=torch.long, device=data.x.device)
        d, e = subgraph(subset, data.edge_index, relabel_nodes=True)
        all_sizes.append(len(subset_expl))

        x_sub = data.x[subset]
        x_pred = gnn_preds.to(data.x.device)[subset]

        gnn.eval()
        with torch.no_grad():
            out, _ = gnn(batch = Data(x=x_sub, edge_index=d))
            out = out.argmax(dim=-1)
            all_fidelities.append(out[0] == x_pred[0])
        
    all_sizes = torch.as_tensor(all_sizes, dtype=torch.float)
    all_fidelities = torch.stack(all_fidelities)

    eval_logger.info(f"Evalaution on {dataset.name_}: Test Fidelity: {all_fidelities.sum()/all_fidelities.size(0):.3f} -- Test Size: {all_sizes.mean():.3f}")


    
def eval_gnn_explanations(
    dataset, 
    gnn,
    gnn_preds,
    method,
    num_eval_samples = 1,
    device="cpu",
):

    data = dataset._data.to(device)

    all_sizes = []
    all_fidelities = []

    for _ in range(5):  # Run 5 trials
        trial_size = []
        trial_fidelity = []

        G = to_networkx(data, to_undirected=True)
    
        for i in range(num_eval_samples):
            graph = nx.ego_graph(G, i, radius=2)
            k_hop_subgraph = nx.ego_graph(G, i, radius=2)
            all_neighbors = set(k_hop_subgraph.nodes)
            all_neighbors.discard(i)

            if len(all_neighbors) == 0:
                continue  # skip if no neighbors to sample

            if method == "random":
                s = random.sample(all_neighbors, int(len(all_neighbors)/2))
                s = [i] + s
                selected_nodes = list(steiner_tree(graph, s).nodes)
                if not selected_nodes:
                    selected_nodes = s
            elif method == "node":
                selected_nodes = [i]
            else:
                raise NotImplementedError

            trial_size.append(len(selected_nodes))

            subset = torch.tensor(selected_nodes, dtype=torch.long)
            d, e = subgraph(subset, data.edge_index, relabel_nodes=True)

            x_sub = data.x[subset]
            x_pred = gnn_preds[subset]
            gnn.eval()
            with torch.no_grad():
                out, _ = gnn(batch = Data(x=x_sub, edge_index=d))
                out = out.argmax(dim=-1)
                trial_fidelity.append((out[0] == x_pred[0]).item())

        all_sizes.append(np.mean(trial_size))
        all_fidelities.append(np.mean(trial_fidelity))
    
    eval_logger.info(f"Evalaution: Test Fidelity: {np.mean(all_fidelities):.3f} -- Test Size: {np.mean(all_sizes):.3f}")



def generic_rbf_kernel(X: torch.Tensor, Y: torch.Tensor, sigma_sq: float = 1.0, max_pairs=1000000) -> torch.Tensor:

    dist_sq = torch.cdist(X, Y, p=2) ** 2
    m, n = dist_sq.shape
    total_pairs = m * n

    if total_pairs > max_pairs:
        sigma_sq = 1.0
    else:
        sigma_sq = max(dist_sq.median() + 1e-9, sigma_sq)

    kernel_matrix = torch.exp(-dist_sq / (2 * sigma_sq))
    return kernel_matrix


    
def mmd(set1: torch.Tensor, set2: torch.Tensor, sigma_sq=1.0, iw=None, max_size=10000):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of samples.

    MMD is a distance metric between distributions. An MMD of zero means the
    distributions are identical. This implementation uses the biased estimator
    of MMD^2.

    Args:
        set1 (torch.Tensor): Samples from the source distribution, tensor of shape (N, D).
        set2 (torch.Tensor): Samples from the target distribution, tensor of shape (M, D).
        sigma_sq (float): The RBF kernel bandwidth.
        iw (torch.Tensor): importance weights.

    Returns:
        torch.Tensor: A scalar tensor representing the MMD value.
    """

    # Ensure the tensors have the same number of feature dimensions
    assert set1.shape[1] == set2.shape[1], \
        f"The feature dimensions of the two sets must be the same. " \
        f"Got {set1.shape[1]} and {set2.shape[1]}."

    rng = np.random.default_rng()
    sets = [set1, set2]
    for i, x in enumerate(sets):
        if x.shape[0] > max_size:
            rand_idxs = rng.choice(x.shape[0], size=max_size, replace=False)
            sets[i] = x[rand_idxs]
    set1, set2 = sets

    # Compute the kernel matrices for within and between sets
    k_xx = generic_rbf_kernel(set1, set1, sigma_sq)
    k_yy = generic_rbf_kernel(set2, set2, sigma_sq)
    k_xy = generic_rbf_kernel(set1, set2, sigma_sq)

    if iw is None:
        # Calculate the MMD^2 statistic
        # MMD^2 = E[k(x, x')] - 2*E[k(x, y)] + E[k(y, y')]
        mmd_sq = k_xx.mean() - 2 * k_xy.mean() + k_yy.mean()
    else:
        # Calculate the MMD^2 statistic
        # MMD^2 = E[β_iβ_jk(x_i, x_j)] - 2*E[β_ik(x_i, y_j)] + E[k(y, y')]
        weighted_k_xx = iw[None, :] * iw[:, None] * k_xx
        weighted_k_xy = iw[:, None] * k_xy
        mmd_sq = weighted_k_xx.mean() - 2 * weighted_k_xy.mean() + k_yy.mean()

    # The MMD value can be negative due to estimation noise with the biased
    # estimator, so we clamp it at 0 before taking the square root.
    return torch.sqrt(torch.clamp(mmd_sq, min=0))