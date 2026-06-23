import torch

from tqdm import tqdm
import random
import numpy as np
import os
import argparse

from scenario_datasets import build_dataset
from scenario_datasets.utils import build_data_loader
from scenario_datasets.collections import CIFAR100, MNIST
from utils import *

from models.factory import get_model_loader, get_trainer

LORA_MAPPING = {
    "q": ['q_proj'],
    "k": ['k_proj'],
    "v": ['v_proj'],
    "qv": ['q_proj', 'v_proj'],
    "qk": ['q_proj', 'k_proj'],
    "kv": ['k_proj', 'v_proj'],
    'o': ['out_proj'],
    'qkv': ['q_proj', 'k_proj', 'v_proj'],
    'qkvo': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
    'in': ['ffn_in'],
    'out': ['ffn_out'],
    'inout': ['ffn_in', 'ffn_out'],
    'kvinout': ['k_proj', 'v_proj', 'ffn_in', 'ffn_out'],
    'qkvinout': ['q_proj', 'k_proj', 'v_proj', 'ffn_in', 'ffn_out'],
    'qkvoinout': ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'ffn_in', 'ffn_out'],
}

###########################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="MoRAM continual fine-tuning")

    # Run & logging
    parser.add_argument("--method", type=str, default="moram", help="Training method (factory entrypoint)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=str, default=None, help="Directory to save LoRA checkpoints")
    parser.add_argument("--load", type=str, default=None, help="Optional checkpoint path for loading")

    # Data
    parser.add_argument(
        "--datasets",
        type=list,
        default=[
            "aircraft", "caltech101", "dtd", "eurosat", "oxford_flowers",
            "food101", "mnist", "oxford_pets", "stanford_cars", "sun397",
        ],
        help="Task sequence (dataset names)",
    )
    parser.add_argument("--data_dir", type=str, default="datasets", help="Dataset root directory")
    parser.add_argument("--num_shots", type=int, default=16, help="Shots per class for few-shot training")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--augmentation_time", type=int, default=1, help="Augmentation repeats per epoch step")
    parser.add_argument("--mtil_eval", default=False, action="store_true", help="Alternate eval / label indexing")

    # Optimization
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations per task")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing")
    parser.add_argument("--warmup_length", type=int, default=0, help="LR warmup steps before cosine decay")

    # Backbone & LoRA placement
    parser.add_argument("--backbone_type", type=str, default="ViT-B-16", help="open_clip model name")
    parser.add_argument("--pretrained_weight", type=str, default="openai", help="open_clip pretrained tag")
    parser.add_argument("--zero_shot", action="store_true", help="Load backbone only (no LoRA)")
    parser.add_argument("--target_encoder", type=str, default="vision", choices=("vision", "text", "all"))
    parser.add_argument(
        "--target_modules_abbrev",
        type=str,
        default="qkvo",
        help="Which linear groups get MoRAM (see LORA_MAPPING in main.py)",
    )
    parser.add_argument("--target_modules", type=list, default=None, help="Override: full module name list")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")

    # MoRAM self-activation
    parser.add_argument("--topk", type=int, default=16, help="Top-k ranks ")
    parser.add_argument("--prune_thre", type=float, default=0.5, help="Gate / prune threshold")
    parser.add_argument("--temp", type=float, default=0.05, help="MoRAM temperature")

    return parser.parse_args()

def setup_logging(logfilename):
    import sys
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)d] - %(message)s")

    # Create file handler
    file_handler = logging.FileHandler(logfilename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def main(cfg):
    
    # Construct log directory and filename
    log_dir = "logs/{}/{}_{}/{}/{}/rank_{}/".format(
        cfg.method,
        cfg.backbone_type,
        cfg.pretrained_weight,
        cfg.target_encoder,
        cfg.target_modules_abbrev,
        cfg.rank)
    os.makedirs(log_dir, exist_ok=True)
    logfilename = os.path.join(log_dir, f"{cfg.save.split('/')[-1]}eval.log")

    # Set up logging
    setup_logging(logfilename)
    
    
    seed = cfg.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_sequence = cfg.datasets
    logging.info(f"Multi-task dataset sequence: {dataset_sequence}")

    # Results
    fusion_acc_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))

    cfg.previous_class_num = 0
    current_class_names = []
    cfg.seen_classes = []

    """
    Loading model
    """
    logging.info('Loading pretrained CLIP model...')
    
    model_loader = get_model_loader(cfg)
    
    model, train_preprocess, val_preprocess, tokenizer = model_loader(vars(cfg))

    """
    Training on dataset sequence
    """
    for task_id, train_dataset in enumerate(dataset_sequence):
        logging.info(f"------------------ Start training on task-{task_id + 1}: dataset-{train_dataset}. ---------------------")
        if train_dataset == "cifar100":
            dataset = CIFAR100(num_shots=cfg.num_shots, preprocess=train_preprocess, val_transform=val_preprocess,
                            batch_size=cfg.batch_size, location=cfg.data_dir)
        elif train_dataset == "mnist":
            dataset = MNIST(num_shots=cfg.num_shots, preprocess=train_preprocess, val_transform=val_preprocess,
                            batch_size=cfg.batch_size, location=cfg.data_dir)
        else:
            dataset = build_dataset(train_dataset, cfg.data_dir, cfg.num_shots, val_preprocess)

        current_class_names += dataset.classnames
        cfg.increment = len(dataset.classnames)
        cfg.current_class_num = len(current_class_names)

        if train_dataset == "cifar100" or train_dataset == "mnist":
            train_loader = dataset.train_loader
        else:
            train_loader = build_data_loader(data_source=dataset.train_x, batch_size=cfg.batch_size, tfm=train_preprocess,
                                            is_train=True, shuffle=True,                                             augmentation_time=cfg.augmentation_time)

        trainer = get_trainer(cfg)
        dataset_name = train_dataset
        
        trainer(cfg, model, tokenizer, dataset_name, dataset, train_loader)

        cfg.trained_class_num = cfg.current_class_num

        evaluation(cfg, dataset_sequence, task_id, val_preprocess, fusion_acc_table, model, tokenizer)

    upper_triangle_no_diag = np.triu(fusion_acc_table, k=1)
    masked_matrix = np.ma.masked_equal(upper_triangle_no_diag, 0)
    transfer_acc = np.mean(masked_matrix, axis=0)
    transfer_avg_acc = np.mean(transfer_acc)
    avg_acc = np.mean(fusion_acc_table, axis=0)
    avg_avg_acc = np.mean(avg_acc)
    logging.info(f'average transfer acc: {transfer_avg_acc}')
    logging.info(f'average average acc: {avg_avg_acc}')
    logging.info(f'average last acc: {np.mean(fusion_acc_table[-1, :])}')

def evaluation(cfg, dataset_sequence, task_id, val_preprocess, fusion_acc_table, model, tokenizer):
    model.eval()
    model = model.cuda()

    all_texts = []
    for test_id, test_dataset in enumerate(dataset_sequence):
        if test_dataset == "cifar100":
            dataset = CIFAR100(num_shots=-1, preprocess=None, val_transform=None, batch_size=cfg.batch_size,
                                location=cfg.data_dir)
        elif test_dataset == "mnist":
            dataset = MNIST(num_shots=-1, preprocess=None, val_transform=None, batch_size=cfg.batch_size,
                            location=cfg.data_dir)
        else:
            dataset = build_dataset(test_dataset, cfg.data_dir, cfg.num_shots, val_preprocess)
        all_texts += [dataset.template[0].format(l) for l in dataset.classnames]
    with torch.no_grad():
        all_texts = tokenizer(all_texts).cuda()
        try:
            all_embeddings = model.encode_text(all_texts)
            all_embeddings = all_embeddings / all_embeddings.norm(dim=-1, keepdim=True)
            all_embeddings = all_embeddings.cuda()
        except:
            all_embeddings, _ = model.encode_text(all_texts)
            all_embeddings = all_embeddings / all_embeddings.norm(dim=-1, keepdim=True)
            all_embeddings = all_embeddings.cuda()

    tested_cls_num = 0
    for test_id, test_dataset in enumerate(dataset_sequence):
        logging.info(f"Evaluating on dataset-{test_id + 1}: {test_dataset}")
        if test_dataset == "cifar100":
            test_set = CIFAR100(num_shots=-1, preprocess=None, val_transform=val_preprocess, batch_size=cfg.batch_size, 
                                location=cfg.data_dir)
        elif test_dataset == "mnist":
            test_set = MNIST(num_shots=-1, preprocess=None, val_transform=val_preprocess, batch_size=cfg.batch_size,
                                location=cfg.data_dir)
        else:
            test_set = build_dataset(test_dataset, cfg.data_dir, cfg.num_shots, val_preprocess)

        if cfg.mtil_eval:
            if test_dataset == "imagenet": 
                texts = ["a photo of a {}.".format(classname) for classname in test_set.classnames]
            else:
                texts = [test_set.template[0].format(classname) for classname in test_set.classnames]
            with torch.no_grad():
                texts = tokenizer(texts).cuda()  # tokenize
                try:
                    class_embeddings, _ = model.encode_text(texts)  # embed with text encoder
                except: 
                    class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                all_embeddings = class_embeddings.cuda()

        if test_dataset == "cifar100" or test_dataset == "mnist" or test_dataset == "imagenet" or test_dataset == "places365":
            test_loader = test_set.test_loader
        else:
            test_loader = build_data_loader(data_source=test_set.test, batch_size=cfg.batch_size, is_train=False,
                                            tfm=val_preprocess, shuffle=False)        

        top1, top5, test_num = 0.0, 0.0, 0.0

        for data in tqdm(test_loader, desc=f'Evaluating on dataset-{test_id + 1}: {test_dataset}',
                        total=len(test_loader), unit='batch'):
            if test_dataset == 'imagenet' :   
                inputs = data["images"].cuda()
                targets = data["labels"].cuda()
            else:
                inputs, targets = data
                inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
            test_num += inputs.size(0)

            if not cfg.mtil_eval:
                targets += tested_cls_num

            with torch.no_grad():
                out, _, _ = model(inputs, None)
                out = out / out.norm(dim=-1, keepdim=True)
                outputs = model.logit_scale.exp() * out @ all_embeddings.t()

            # Zero-shot acc
            acc1, acc5 = cls_acc(outputs, targets, topk=(1, 5))
            top1 += acc1
            top5 += acc5

        top1, top5 = (top1 / test_num) * 100, (top5 / test_num) * 100
        logging.info(f"top-1 acc for dataset-{test_id + 1}: {test_dataset}: {top1}")

        fusion_acc_table[task_id, test_id] = top1

        tested_cls_num += len(test_set.classnames)
                
    logging.info(fusion_acc_table)
    if cfg.save is not None:
        results_path = cfg.save.split("/")[-1]
        save_path = os.path.join("./results", f"{results_path}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        outfile = os.path.join(save_path, f"rail_results.npy")
        np.save(outfile, fusion_acc_table)
        logging.info(f"Results saved to {outfile}")


if __name__ == "__main__":
    args = parse_arguments()
    args.target_modules = LORA_MAPPING[args.target_modules_abbrev]

    logging.info(args)
    main(args)
