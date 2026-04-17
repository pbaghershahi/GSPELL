import torch, random, os, ipdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils import *
from data_utils import *
from model import *
from eval_utils import test
import ipdb

logger = logging.getLogger("global_logger")


def pretrain_gnn(
    dataset,
    model_config,
    optimizer_config,
    training_config,
    eval_step = 1,
    save_model = True, 
    pretext_task = "classification",
    model_dir = "./pretrained",
):
    model = PretrainedModel(**model_config)
    model.to(model.device)

    if pretext_task == "classification":
        obj_fun = nn.CrossEntropyLoss()
    else:
        raise Exception("Pretext task is not implemented yet!")
    optimizer = Adam(model.parameters(), lr = optimizer_config["lr"], weight_decay = optimizer_config["weight_decay"])
    scheduler = StepLR(optimizer, step_size = optimizer_config["scheduler_step_size"], gamma = optimizer_config["scheduler_gamma"])
    
    test_loss, test_acc, test_f1 = test(model, dataset, validation = False)
    logger.info(f'GNN Before Pretraining: -- Test Loss: {test_loss:.3f} -- Test ACC: {test_acc:.3f} -- Test F1-score: {test_f1:.3f}')

    train_ds = dataset.train_loader.dataset
    num_epochs = training_config["num_epochs"]
    for epoch in range(num_epochs):
        model.train()
        for i, batch_idxs in enumerate(dataset.train_loader):
            optimizer.zero_grad()
            temp_labels = train_ds.y[batch_idxs].to(model.device)
            scores, _ = model(
                batch = train_ds,
                decoder = True,
            )
            loss = obj_fun(
                scores[batch_idxs],
                temp_labels
            )
            loss.backward()
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if epoch % eval_step == 0 and epoch > 0:
            val_loss, val_acc, val_f1 = test(model, dataset, validation = True)
            logger.info(
                f"Epoch: {epoch}/{num_epochs} -- Train Loss: {loss:.3f} -- " +
                f"Validation Loss: {val_loss:.3f} -- Validation ACC: {val_acc:.3f} -- Validation F1: {val_f1:.3f}"
            )

    results = {}
    test_loss, results["pretrained_test_acc"], results["pretrained_test_f1"] = test(model, dataset, validation = False)
    logger.info(
        f"GNN After Pretraining: -- Train Loss: {loss:.3f} "
        f"-- Test Loss: {test_loss:.3f} "
        f"-- Test ACC: {results['pretrained_test_acc']:.3f} "
        f"-- Test F1: {results['pretrained_test_f1']:.3f}"
    )

    if save_model:
        model_path = os.path.join(model_dir, f"{model_config['gnn_type']}.pth")
        torch.save(
            {
                'model_state_dict': model.state_dict(),
            }, model_path
        )
        logger.info(f"Model saved to: {model_path}")
    else:
        model_path = "Won't be stored"
    return model, model_path, results



def context_loss(soft_prompts, keyword_embeddings):
    # Mean pool the soft prompt across tokens
    pooled_prompt = soft_prompts.mean(dim=1)
    pooled_prompt = F.normalize(pooled_prompt, dim=-1)
    keyword_embeddings = F.normalize(keyword_embeddings, dim=-1)
    # Cosine similarity loss (maximize similarity = minimize negative cosine)
    return - (pooled_prompt * keyword_embeddings).sum(dim=-1).mean()



def contrastive_loss_from_gnn(soft_prompts, gnn_embeddings, temperature=0.5):
    soft_prompts = soft_prompts.mean(dim=1)
    soft_prompts = F.normalize(soft_prompts, dim=-1)
    gnn_embeddings = F.normalize(gnn_embeddings, dim=-1)

    # Compute similarity matrices
    soft_sim = torch.matmul(soft_prompts, soft_prompts.T)  # (B, B)
    gnn_sim = torch.matmul(gnn_embeddings, gnn_embeddings.T)  # (B, B)
    gnn_sim = gnn_sim / temperature
    gnn_sim_probs = F.softmax(gnn_sim, dim=1)

    # Cross-entropy between soft similarities and GNN similarity "labels"
    loss = -torch.sum(gnn_sim_probs * F.log_softmax(soft_sim, dim=1), dim=1).mean()
    return loss



def mutual_info_loss(x, y, temperature=0.5):
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)

    labels = torch.arange(x_norm.shape[0])

    sim_mat = (x_norm @ y_norm.T) / temperature

    return F.cross_entropy(sim_mat, labels.to(sim_mat.device))

    

def pretrain_projector(
    dataset,
    embed_func,
    tokenizer,
    gnn_embeds,
    gnn_logits,
    projector_config,
    optimizer_config,
    training_config,
    gnn_preds = None,
    use_bow = False,
    eval_step = 50,
    model_dir = None, 
):

    projector = GNNToSoftPrompt(**projector_config)
    projector.to(gnn_embeds.device)

    optimizer = Adam(projector.parameters(), lr = optimizer_config["lr"], weight_decay = optimizer_config["weight_decay"])
    scheduler = StepLR(optimizer, step_size = optimizer_config["scheduler_step_size"], gamma = optimizer_config["scheduler_gamma"])

    llm_emebds = []

    for i in range(len(dataset)):
        prompt_content = ""
        if use_bow:
            prompt_content += ", ".join(dataset._data.words[i])
            prompt_content += ", " + dataset._data.label_info[str(gnn_preds[i].item())]
        else:
            prompt_content = dataset._data.raw_text[i]
        
        if dataset.name_ == "wikics":
            tokenized_prompt = tokenizer(prompt_content, is_split_into_words=True, return_tensors="pt", add_special_tokens=False)
        else:
            tokenized_prompt = tokenizer(prompt_content, return_tensors="pt", add_special_tokens=False)

        prompt_embeds = embed_func(
            tokenized_prompt["input_ids"].to(torch.long).to(gnn_embeds.device)
        ).squeeze(0)
        prompt_embeds = prompt_embeds.mean(dim=0)
        llm_emebds.append(prompt_embeds)

    llm_emebds = torch.stack(llm_emebds, dim=0)

    gnn_embeds = F.normalize(gnn_embeds, dim=-1).to(gnn_embeds.device)
    llm_emebds = F.normalize(llm_emebds, dim=-1).to(gnn_embeds.device).detach()

    num_epochs = training_config["num_epochs"]
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        soft_prompts, gnn_proj_logits = projector(gnn_embeds)

        contrastive = contrastive_loss_from_gnn(soft_prompts, gnn_embeds, temperature=training_config["temperature"])
        context = context_loss(soft_prompts, llm_emebds)

        mi_loss = torch.tensor(0.0)
        if gnn_proj_logits is not None:
            mi_loss = mutual_info_loss(gnn_proj_logits, gnn_logits)

        loss = training_config["proj_contrastive_w"] * contrastive \
        + training_config["proj_context_w"] * context \
        + training_config["proj_mutualinfo_w"] * mi_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % eval_step == 0:
            logger.info(f"Step {i}: Total Loss = {loss.item():.3f} | Contrastive = {contrastive.item():.3f} | Context = {context.item():.3f} | MI = {mi_loss.item():.3f}")

    if model_dir:
        model_path = os.path.join(model_dir, "projector.pth")
        torch.save(
            {
                'model_state_dict': projector.state_dict(),
            }, model_path
        )
        logger.info(f"Model saved to: {model_path}")

    return projector, model_path