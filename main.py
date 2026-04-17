import os, logging
import sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).resolve().parents[0])
sys.path.insert(0, ROOT_PATH)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from datetime import datetime
import functools
import argparse
import ipdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import shutil
import json, yaml
import itertools
import networkx as nx
import re
from tqdm import tqdm
from torch_geometric.utils import to_networkx, subgraph
import random
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPT2Tokenizer, GPT2Model
import transformers
import gc
from collections import defaultdict, Counter
from nltk.corpus import stopwords
import nltk
import pickle
from math import ceil
import torch.nn.functional as F
from networkx.algorithms.approximation import steiner_tree
from TAGLAS import get_dataset
from datetime import datetime
from train_utils import *
from eval_utils import *
from utils import *
from model import PretrainedModel, GNNToSoftPrompt
import wandb
from scipy import stats
from huggingface_hub import login

hugging_face_token = os.environ.get('HUGGINGFACE_TOKEN')
login(token=hugging_face_token)


def main(args):

    if args.config_from_file:

        flag_to_dest_map = {}
        for action in parser._actions:
            for option_string in action.option_strings:
                flag_to_dest_map[option_string] = action.dest

        input_dests = set()
        for arg in sys.argv[1:]:
            if arg in flag_to_dest_map:
                input_dests.add(flag_to_dest_map[arg])

        with open(args.config_from_file, 'r') as infile:
            all_args = vars(args)
            file_args = yaml.safe_load(infile) or {}

        merged_args = {
            key: file_args[key] if (key in file_args and key not in input_dests) else value
            for key, value in all_args.items()
        }
        
        merged_args.update({k: v for k, v in file_args.items() if k not in merged_args})
        args = argparse.Namespace(**merged_args)

    data_root_dir = os.path.join(ROOT_PATH, "data")
    exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    run_group_id = f"{args.method}_{args.dataset}_{exec_name}"

    if args.output_path_dir is None:
        ouputs_path = Path(ROOT_PATH) / "outputs"
        model_path = Path(ROOT_PATH) / "pretrained"
        log_dir = Path(ROOT_PATH) / "logs"
    else:
        ouputs_path = Path(args.output_path_dir) / "outputs"
        model_path = Path(args.output_path_dir) / "pretrained"
        log_dir = Path(args.output_path_dir) / "logs"

    if args.write_new_output:
        output_dir = os.path.join(
            ouputs_path,
            f"{args.dataset}-{args.llm_model}-{exec_name}/"
        )
        model_dir = os.path.join(
            model_path,
            f"{args.dataset}-{args.llm_model}-{exec_name}/"
        )
    else:
        output_dir = os.path.join(
            ouputs_path,
            f"{args.dataset}-{args.llm_model}/"
        )
        model_dir = os.path.join(
            model_path,
            f"{args.dataset}-{args.llm_model}/"
        )

    if args.not_verbose:
        global_logger = DummyLogger()
        eval_logger = DummyLogger()
    else:

        general_log_dir = Path(log_dir) / "general"
        evaluations_log_dir = Path(log_dir) / "evaluation"
        os.makedirs(general_log_dir, exist_ok=True)
        os.makedirs(evaluations_log_dir, exist_ok=True)

        log_file_paths = [
            general_log_dir / f"global_{args.method}_{args.dataset}_{exec_name}.log", 
            evaluations_log_dir / f"eval_{args.method}_{args.dataset}_{exec_name}.log"
        ]
        global_logger = setup_logger(
            name="global_logger", 
            level=logging.INFO, 
            log_file=log_file_paths[0],
            stream_handler=True,
        )
        eval_logger = setup_logger(
            name="eval_logger", 
            level=logging.INFO, 
            log_file=log_file_paths[1],
            stream_handler=True,
            formatter=logging.Formatter(fmt='%(message)s')
        )
        global_logger.info(f"Logging to: {log_file_paths}")

    arg_seeds = np.random.randint(1000, 5000, (args.total_iters,)) if len(args.seed) == 0 else args.seed
    total_iters = len(arg_seeds)
    global_logger.info(args)

    if args.output_path_dir is None:
        ouputs_path = Path(ROOT_PATH) / "outputs"
        model_path = Path(ROOT_PATH) / "pretrained"
    else:
        ouputs_path = Path(args.output_path_dir) / "outputs"
        model_path = Path(args.output_path_dir) / "pretrained"

    if args.write_new_output:
        output_dir = os.path.join(
            ouputs_path,
            f"{args.dataset}-{args.llm_model}-{exec_name}/"
        )
        model_dir = os.path.join(
            model_path,
            f"{args.dataset}-{args.llm_model}-{exec_name}/"
        )
    else:
        output_dir = os.path.join(
            ouputs_path,
            f"{args.dataset}-{args.llm_model}/"
        )
        model_dir = os.path.join(
            model_path,
            f"{args.dataset}-{args.llm_model}/"
        )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    json_out_dir = os.path.join(output_dir, "json_outs")
    os.makedirs(json_out_dir, exist_ok=True)
    json_out_path = os.path.join(json_out_dir, f"{args.method}_{args.dataset}_{exec_name}.json")

    global_logger.info(f"Seeds: {arg_seeds}")
    global_logger.info(f"Saving outputs to:\n{output_dir}")
    global_logger.info("#"*100)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.download_nltk_stopwords:
        nltk.download('stopwords')

    if not args.no_wandb:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=wandb_api_key)

    all_results = {}

    def update_results(temp_results):
        for key, value in temp_results.items():
            if key in all_results:
                all_results[key].append(value)
            else:
                all_results[key] = [value]

    for i in range(total_iters):
        if not args.no_wandb:
            run = wandb.init(
                project="CertainGNN",
                group=run_group_id,
                name=f"seed_{i}_{exec_name}",
                config=vars(args),
                reinit=True # Allows multiple wandb.init() in one script
            )

        global_logger.info(f"Started round {i+1}/{total_iters} of experiments!")
        fix_seed(arg_seeds[i])

        if args.dataset == "products":
            dataset = get_node_dataset(
                dataset_name = args.dataset,
                train_test_split = args.train_test_split,
                batch_size = args.gnn_batch_size,
                normal_mode = args.normal_mode,
                seed = arg_seeds[i],
                sample_size = args.dataset_sample_size,
                random_sampling = args.dataset_random_sampling,
                num_workers = args.gnn_dataloader_num_workers,
                use_bow = args.use_bow,
                num_hops = args.neighborhood_max_hops,
                max_num_frequent_words = args.max_num_frequent_words
            )
        elif args.dataset == "cora":
            dataset = get_node_dataset(
                dataset_name = args.dataset,
                train_test_split = args.train_test_split,
                batch_size = args.gnn_batch_size,
                normal_mode = args.normal_mode,
                seed = arg_seeds[i],
                sample_size = args.dataset_sample_size,
                random_sampling = args.dataset_random_sampling,
                num_workers = args.gnn_dataloader_num_workers,
                use_bow = args.use_bow,
                num_hops = args.neighborhood_max_hops,
                max_num_frequent_words = args.max_num_frequent_words
            )
        elif args.dataset == "wikics":
            dataset = get_node_dataset(
                dataset_name = args.dataset,
                train_test_split = args.train_test_split,
                batch_size = args.gnn_batch_size,
                normal_mode = args.normal_mode,
                seed = arg_seeds[i],
                sample_size = args.dataset_sample_size,
                random_sampling = args.dataset_random_sampling,
                num_workers = args.gnn_dataloader_num_workers,
                use_bow = args.use_bow,
                num_hops = args.neighborhood_max_hops,
                max_num_frequent_words = args.max_num_frequent_words
            )
        elif args.dataset == "liar":
            dataset = get_node_dataset(
                dataset_name = args.dataset,
                train_test_split = args.train_test_split,
                batch_size = args.gnn_batch_size,
                normal_mode = args.normal_mode,
                seed = arg_seeds[i],
                sample_size = args.dataset_sample_size,
                random_sampling = args.dataset_random_sampling,
                num_workers = args.gnn_dataloader_num_workers,
                use_bow = args.use_bow,
                num_hops = args.neighborhood_max_hops,
                max_num_frequent_words = args.max_num_frequent_words
            )
        else:
            raise NotImplementedError

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pretrained_gnn_paths = args.pretrained_gnn_path if len(args.pretrained_gnn_path) > 0 else []
        processed_data_paths = args.processed_data_path if len(args.processed_data_path) > 0 else []
        pretrained_proj_paths = args.pretrained_projector_path if len(args.pretrained_projector_path) > 0 else []
        generated_exp_paths = args.generated_exp_path if len(args.generated_exp_path) > 0 else []
        all_results = {}

        model_config = dict(
            gnn_type = args.gnn_type,
            in_channels = dataset.n_feats,
            hidden_channels = args.gnn_h_dim,
            out_channels = dataset.num_classes,
            gnn_num_hid_layers = args.gnn_num_hid_layers, 
            dropout = args.gnn_dropout,
            with_bn = False,
            decoder_type = args.gnn_decoder_type,
            device = device,
            with_last_dropout = args.gnn_with_last_dropout,
            mode = args.gnn_task_mode
        )
        optimizer_config = dict(
            lr = args.gnn_lr,
            scheduler_step_size = args.gnn_step_size,
            scheduler_gamma = args.gnn_gamma,
            weight_decay = args.gnn_weight_decay
        )
        training_config = dict(
            num_epochs = args.gnn_num_epochs
        )

        if len(pretrained_gnn_paths) == 0:
            global_logger.info(f"Pretraining {model_config['gnn_type']} on {args.dataset} started for {args.gnn_num_epochs} epochs")
            gnn, p_path, p_results = pretrain_gnn(
                dataset,
                model_config,
                optimizer_config,
                training_config,
                eval_step = args.gnn_eval_step,
                save_model = True,
                pretext_task = "classification",
                model_dir = model_dir,
            )
            pretrained_gnn_paths.append(p_path)
        else:
            p_results = {}
            gnn = PretrainedModel(**model_config).to(device)
            load_model(gnn, read_checkpoint=True, pretrained_path=pretrained_gnn_paths[i])
            _, p_results["pretrained_test_acc"], p_results["pretrained_test_f1"] = test(gnn, dataset, validation = False)
            global_logger.info(
                f"Reading GNN model from: {pretrained_gnn_paths[i]}\n"
                f"GNN After Pretraining: "
                f"-- Test ACC: {p_results['pretrained_test_acc']:.3f} "
                f"-- Test F1: {p_results['pretrained_test_f1']:.3f}"
            )

        gnn.eval()

        if len(processed_data_paths) == 0:
            with torch.no_grad():
                gnn_logits, gnn_embeds = gnn(batch=dataset._data)
                gnn_preds = gnn_logits.argmax(dim=1)
            outdict = dict(
                dataset = dataset,
                gnn_embeds = gnn_embeds.detach().to("cpu"),
                gnn_logits = gnn_logits.detach().to("cpu"),
                gnn_preds = gnn_preds.detach().to("cpu")
            )
            path_to_saved_data = os.path.join(output_dir, "data.pth")
            torch.save(outdict, path_to_saved_data)
            global_logger.info(f"Data and GNN preds saved to {path_to_saved_data}")
            processed_data_paths.append(path_to_saved_data)
        else:
            global_logger.info(f"Reading Data and GNN preds from: {processed_data_paths[i]}")
            output_dict = torch.load(processed_data_paths[i])
            dataset = output_dict["dataset"]
            gnn_preds, gnn_embeds, gnn_logits = output_dict["gnn_preds"].to(device), output_dict["gnn_embeds"].to(device), output_dict["gnn_logits"].to(device)

        eval_config = dict(
            max_num_eval_nodes = args.max_num_eval_nodes,
            llm_node_batch_size = args.llm_node_batch_size,
            save_every = args.save_generation_every
        )

        proj_embeds = None
        if args.method == "gspell":

            tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
            llm = AutoModelForCausalLM.from_pretrained(
                args.llm_model,
                torch_dtype = torch.bfloat16,
                device_map = "auto"
            )

            llm_type = args.llm_model.split("/")[0]
            if llm_type in ["meta-llama", "mistralai", "microsoft"]:
                embed_func = llm.model.embed_tokens
                with_chat_template = True
                eval_config["max_position_embeddings"] = llm.config.max_position_embeddings
            elif llm_type in ["openai"]:
                embed_func = llm.transformer.wte
                with_chat_template = False
                eval_config["max_position_embeddings"] = llm.config.n_positions
            elif llm_type in ["EleutherAI"]:
                embed_func = llm.transformer.wte
                with_chat_template = False
                eval_config["max_position_embeddings"] = llm.config.max_position_embeddings

            projector_config = dict(
                gnn_h_dim = args.gnn_h_dim, 
                num_tokens = args.num_tokens, 
                llm_h_dim = llm.config.hidden_size,
                gnn_out_dim = dataset.num_classes if args.projector_with_backward else None
            )
            optimizer_config = dict(
                lr = args.projector_lr,
                scheduler_step_size = args.projector_step_size,
                scheduler_gamma = args.projector_gamma,
                weight_decay = args.projector_weight_decay
            )
            training_config = dict(
                num_epochs = args.projector_num_epochs,
                temperature = args.contrastive_temperature,
                proj_contrastive_w = args.projector_contrastive_w,
                proj_context_w = args.projector_context_w,
                proj_mutualinfo_w = args.projector_mutualinfo_w
            )

            if len(pretrained_proj_paths) == 0:
                projector, p_path = pretrain_projector(
                    dataset,
                    embed_func,
                    tokenizer,
                    gnn_embeds,
                    gnn_logits,
                    projector_config,
                    optimizer_config,
                    training_config,
                    gnn_preds = gnn_preds,
                    use_bow = args.use_bow,
                    eval_step = 50,
                    model_dir = model_dir,
                )
                pretrained_proj_paths.append(p_path)
            else:
                global_logger.info(f"Reading projector model from: {pretrained_proj_paths[i]}")
                projector = GNNToSoftPrompt(**projector_config).to(device)
                load_model(projector, read_checkpoint=True, pretrained_path=pretrained_proj_paths[i])
            
            projector.eval()
            proj_embeds, _ = projector(gnn_embeds)
            proj_embeds = proj_embeds.to(dtype=torch.bfloat16)

            start_time = time.time()
            if len(generated_exp_paths) <= i:
                gen_exp_save_path = os.path.join(output_dir, "GSPELL_explanations.pt")
                generated_exps = generate_exp_by_llm(
                    dataset = dataset,
                    model = llm,
                    tokenizer = tokenizer,
                    gnn_embeds = proj_embeds,
                    gnn_preds = gnn_preds,
                    eval_config = eval_config,
                    embed_func = embed_func,
                    generate_by = "embedding",
                    save_to = gen_exp_save_path,
                    save_every = 20,
                    with_chat_template = with_chat_template,
                    descriptive_prompt = args.add_descriptive_prompt
                )
                generated_exp_paths.append(gen_exp_save_path)
            else:
                generated_exps = None
                gen_exp_save_path = generated_exp_paths[i]

            eval_results = eval_llm_explanations(
                dataset = dataset, 
                gnn = gnn,
                gnn_preds = gnn_preds,
                generated_exps = generated_exps,
                generated_exps_path = gen_exp_save_path,
                generated_with_pipeline = False,
                num_eval_samples = args.max_num_eval_nodes
            )
            end_time = time.time()
        
        elif args.method == "llm":

            tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
            llm = transformers.pipeline(
                "text-generation",
                model=args.llm_model,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                # token = key
            )

            llm_type = args.llm_model.split("/")[0]
            if llm_type in ["meta-llama", "mistralai", "microsoft"]:
                with_chat_template = True
                eval_config["max_position_embeddings"] = llm.model.config.max_position_embeddings
            elif llm_type in ["openai"]:
                with_chat_template = False
                eval_config["max_position_embeddings"] = llm.model.config.n_positions
            elif llm_type in ["EleutherAI"]:
                with_chat_template = False
                eval_config["max_position_embeddings"] = llm.model.config.max_position_embeddings
            
            start_time = time.time()
            if len(generated_exp_paths) <= i:
                gen_exp_save_path = os.path.join(output_dir, "LLM_explanations.pt")
                generated_exps = generate_exp_by_llm(
                    dataset = dataset,
                    model = llm,
                    tokenizer = tokenizer,
                    gnn_preds = gnn_preds,
                    eval_config = eval_config,
                    generate_by = "tokens",
                    save_to = gen_exp_save_path,
                    save_every = 20,
                    with_chat_template = with_chat_template,
                    descriptive_prompt = args.add_descriptive_prompt
                )
                generated_exp_paths.append(gen_exp_save_path)
            else:
                generated_exps = None
                gen_exp_save_path = generated_exp_paths[i]

            eval_results = eval_llm_explanations(
                dataset = dataset, 
                gnn = gnn,
                gnn_preds = gnn_preds,
                generated_exps = generated_exps,
                generated_exps_path = gen_exp_save_path,
                generated_with_pipeline = True,
                num_eval_samples = args.max_num_eval_nodes
            )
            end_time = time.time()

        elif args.method == "random":

            start_time = time.time()
            eval_results = eval_gnn_explanations(
                dataset = dataset, 
                gnn = gnn,
                gnn_preds = gnn_preds,
                method = args.method,
                num_eval_samples = args.max_num_eval_nodes,
                device = device,
            )
            end_time = time.time()

        elif args.method == "node":

            start_time = time.time()
            eval_results = eval_gnn_explanations(
                dataset = dataset, 
                gnn = gnn,
                gnn_preds = gnn_preds,
                method = args.method,
                num_eval_samples = args.max_num_eval_nodes,
                device = device,
            )
            end_time = time.time()

        elif args.method == "gnnexplainer":

            start_time = time.time()
            eval_results = eval_gnnexplainer_explanations(
                dataset = dataset, 
                gnn = gnn,
                gnn_preds = gnn_preds,
                device = device,
                num_eval_samples = args.max_num_eval_nodes,
                save_to = os.path.join(output_dir, "GNNExplainer_explanations.json"),
            )
            end_time = time.time()

        elif args.method == "pgexplainer":

            start_time = time.time()
            eval_results = eval_pgexplainer_explanations(
                dataset = dataset, 
                gnn = gnn,
                gnn_preds = gnn_preds,
                device = device,
                num_eval_samples = args.max_num_eval_nodes,
                num_explainer_epochs = 30
            )
            end_time = time.time()

        elif args.method == "tage":

            exp_config = dict(
                gnn_h_dim = args.gnn_h_dim
            )

            start_time = time.time()
            eval_results = eval_tage_explanations(
                dataset = dataset, 
                gnn = gnn,
                gnn_preds = gnn_preds,
                exp_config = exp_config,
                device = device,
                num_eval_samples = args.max_num_eval_nodes,
                num_explainer_epochs = 200,
            )
            end_time = time.time()
        
        spent_time = (end_time - start_time)/args.max_num_eval_nodes
        global_logger.info(f"Elapsed: {spent_time:.4f} seconds")
        eval_results |= {"spent_time": spent_time}
        update_results(eval_results)

        if not args.no_wandb:
            wandb.log(eval_results)
            run.finish()

    output_dict = {
        "args": vars(args),
        "results": dict()
    }

    log_content = "\n" + "="*20 + " Final Results " + "="*20 + "\n"
    log_content += "Method: " + args.method.upper() + "\n"
    log_content += "Dataset: " + args.dataset.upper() + "\n"

    for key, value in all_results.items():
        if isinstance(value[0], np.ndarray):
            all_results[key] = stats.describe(np.vstack(value), ddof=0)
        else:
            all_results[key] = stats.describe(np.array(value), ddof=0)

    for key, value in all_results.items():
        v_mean = np.atleast_1d(value.mean)
        v_std = np.atleast_1d(np.sqrt(value.variance))

        mean_content = ", ".join([f"{v:.3f}" for v in v_mean])
        std_content = ", ".join([f"{v:.3f}" for v in v_std])

        metric = ' '.join(key.split('_')).upper()
        log_content += f"{metric}: \n\tMean: {mean_content}\n\tStd: {std_content}\n"

        output_dict["results"][key] = {
            "mean": [round(float(v), 3) for v in v_mean],
            "std": [round(float(v), 3) for v in v_std]
        }

    global_logger.info(log_content)

    with open(json_out_path, "w") as f:
        json.dump(output_dict, f, indent=4)

    if not args.no_wandb:
        final_summary_run = wandb.init(
            project="CertainGNN",
            group=run_group_id,
            name=f"SUMMARY_{exec_name}",
            job_type="summary",
            config=vars(args)
        )
        wandb.log({"final_results": output_dict["results"]})
        final_summary_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str)
    parser.add_argument('-lm', '--llm-model', type=str, help=(
        "Choose from: "
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
        "meta-llama/Llama-2-7b-chat-hf"
        "openai-community/gpt2"
        "mistralai/Mistral-7B-Instruct-v0.2"
        "EleutherAI/gpt-neo-2.7B"
        "microsoft/Phi-3-mini-4k-instruct"
        )
    )
    parser.add_argument("-dns", "--download_nltk_stopwords", action="store_true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("-ub", "--use_bow", action="store_true")
    parser.add_argument("-gdnw", "--gnn_dataloader_num_workers", type=int, help="Number of graph dataloader workers")
    parser.add_argument("-dss", "--dataset_sample_size", type=int, help="Graph dataset sample size")
    parser.add_argument("-drs", "--dataset_random_sampling", action="store_true")
    parser.add_argument("-pgp", "--pretrained_gnn_path", nargs="*", default=[], type=str, help="Paths to the pretrained model")
    parser.add_argument("-pdp", "--processed_data_path", nargs="*", default=[], type=str, help="Paths to the processed data")
    parser.add_argument("-ppp", "--pretrained_projector_path", nargs="*", default=[], type=str, help="Paths to the pretrained model")
    parser.add_argument("-gep", "--generated_exp_path", nargs="*", default=[], type=str, help="Paths to the generated explanations")
    parser.add_argument("-gwlp", "--gnn_with_last_dropout", action="store_true")
    parser.add_argument("-gtm", "--gnn_task_mode", type=str)
    parser.add_argument("-gwd", "--gnn_weight_decay", type=float, help="Rate of regularization")
    parser.add_argument("-gt", "--gnn_type", type=str, help="Type of base GNN: [gcn, gat, gin, sage]")
    parser.add_argument("-gdt", "--gnn_decoder_type", type=str, help="GNN decoder: [linear, gnn]")
    parser.add_argument("-gnh", "--gnn_num_hid_layers", type=int, help="Number of layers of the base GNN")
    parser.add_argument("-gne", "--gnn_num_epochs", type=int, help="Number of epochs for pretraining")
    parser.add_argument("-ges", "--gnn_eval_step", type=int, help="Evaluation step for pretrained model")
    parser.add_argument("-ghd", "--gnn_h_dim", type=int, help="Hidden dim of the GNN")
    parser.add_argument("-gl", "--gnn_lr", type=float, help="Learning rate for pretraining the gnn")
    parser.add_argument("-gss", "--gnn_step_size", type=int, help="Learning rate step size for pretraining the gnn")
    parser.add_argument("-gg", "--gnn_gamma", type=float, help="Learning rate gamma for pretraining the gnn")
    parser.add_argument("-gbs", "--gnn_batch_size", type=int, help="Batch size for pretraining")
    parser.add_argument("-gn", "--gnn_dropout", type=float, help="Dropout for GNN")
    parser.add_argument("-pl", "--projector_lr", type=float, help="Learning rate for pretraining the projector")
    parser.add_argument("-pss", "--projector_step_size", type=int, help="Learning rate step size for pretraining the projector")
    parser.add_argument("-pwd", "--projector_weight_decay", type=float, help="Rate of regularization")
    parser.add_argument("-pg", "--projector_gamma", type=float, help="Learning rate gamma for pretraining the projector")
    parser.add_argument("-pne", "--projector_num_epochs", type=int, help="Number of epochs for pretraining the projector")
    parser.add_argument("-ct", "--contrastive_temperature", type=float, help="contrastive learning temperature")
    parser.add_argument("-pcrw", "--projector_contrastive_w", type=float, help="contrastive loss weights")
    parser.add_argument("-pcew", "--projector_context_w", type=float, help="context loss weights")
    parser.add_argument("-pmiw", "--projector_mutualinfo_w", type=float, help="mutual information loss weights")
    parser.add_argument("-pwb", "--projector_with_backward", action="store_true")
    parser.add_argument("-nv", "--not_verbose", action="store_true")
    parser.add_argument("-wno", "--write_new_output", action="store_true")
    parser.add_argument("-tt", "--total_iters", type=int, help="Total number of trials with random initialization of datasets")
    parser.add_argument("--seed", nargs="*", type=int, default=[], help="Seed for random")
    parser.add_argument("-cff", "--config_from_file", type=str, default="", help="Config file to read from")
    parser.add_argument("-ld", "--log_dir", default=None, type=str)
    parser.add_argument("-lbs", "--llm_batch_size", type=int)
    parser.add_argument("-mnen", "--max_num_eval_nodes", type=int)
    parser.add_argument("-mnfw", "--max_num_frequent_words", type=int)
    parser.add_argument("-nmh", "--neighborhood_max_hops", type=int)
    parser.add_argument("-lnbs", "--llm_node_batch_size", type=int)
    parser.add_argument("-sge", "--save_generation_every", type=int)
    parser.add_argument("-nt", "--num_tokens", type=int)
    parser.add_argument("-tts", "--train_test_split", nargs="*", type=float, help="[train_percentage, test_percentage]")
    parser.add_argument("-nm", "--normal_mode", type=str, help="Config file to save to")
    parser.add_argument("-adp", "--add_descriptive_prompt", action="store_true")
    parser.add_argument("-nwdb", "--no_wandb", action="store_true")
    parser.add_argument("-opd", "--output_path_dir", default=None, type=str)
    args = parser.parse_args()
    main(args)