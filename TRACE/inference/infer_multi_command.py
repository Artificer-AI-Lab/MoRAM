# !/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import ast
import copy
import os
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import math
import sys
from tqdm import tqdm

import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten, eval_amazon, eval_yelp, eval_agnews, eval_dbpedia, eval_yahoo, eval_BoolQA, eval_QQP  # to be continued

CHOICE_TASK_LABELS = {
    "C-STANCE": tuple("ABC"),
    "FOMC": tuple("ABC"),
    # NumGLUE tasks are math problems requiring numerical answers, NOT multiple choice
    # "NumGLUE-cm": tuple("ABCD"),  # REMOVED - math task
    # "NumGLUE-ds": tuple("ABCD"),  # REMOVED - math task
    "BoolQA": ("A", "B"),
    "QQP": ("A", "B"),
    "amazon": tuple("ABCD"),
    "yelp": tuple("ABCD"),
    "agnews": tuple("ABCD"),
    "dbpedia": tuple("ABCDEFGH"),
    "yahoo": tuple("ABCDEFGHIJ"),
}
CHOICE_TASK_REGEX = {
    task: re.compile(rf"\b([{''.join(labels)}])\b", re.IGNORECASE)
    for task, labels in CHOICE_TASK_LABELS.items()
}


# # add flash attention
# from utils.flash_attention.llama_flash_att import replace_llama_attn_with_flash_attn
# from utils.flash_attention.bloom_flash_att import replace_bloom_attn_with_flash_attn
#
# replace_llama_attn_with_flash_attn()
# replace_bloom_attn_with_flash_attn()

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset. A single data path.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='./tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--inference_model_path",
        type=str,
        help=
        "Path to inference model.",
        required=True,
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    # inference params
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=256,
        help="The maximum answer length.",
    )
    parser.add_argument(
        "--meetingbank_max_prompt_len",
        type=int,
        default=4096,
        help="Max prompt tokens for MeetingBank (must match training).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Generate temperature params.",
    )
    
    parser.add_argument(
        "--inference_batch",
        type=int,
        default=4,
        help="Inference batch size.",
    )
    #  add other inference params
    parser.add_argument(
        "--inference_tasks",
        type=str,
        default='all',
        help='Datasets to be used.'
    )
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    
    
    parser.add_argument('--inference_output_path',
                        type=str,
                        default=None,
                        help="Where to store inference results.")
    parser.add_argument('--CL_method',
                        default=None,
                        help='continual learning method used')
    
    parser.add_argument('--start_round',
                        default=0,
                        type=int,
                        help='which round (task) to start')
    
    parser.add_argument(
        "--lora_depth",
        type=int,
        default=-1,
        help="max depth of lora. -1 means no limit.",
    )
    parser.add_argument(
        "--moram_router_temp",
        type=float,
        default=0.01,
        help="MoRAM: softmax temperature on sparse router logits (must match training).",
    )
    parser.add_argument(
        "--moram_infer_lora_a_thresh",
        type=float,
        default=0.0,
        help="MoRAM inference: gate on L2-normed rank projections (>=), frozen and current-task ranks. 0 disables.",
    )
    parser.add_argument(
        "--moram_topk",
        type=int,
        default=None,
        help="MoRAM: optional router top-k override (omit to use adapter checkpoint).",
    )

    parser.add_argument(
        "--gpus",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--master_port",
        type=int,
        required=True
    )
    # parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    return args


def save_inference_results(output_dir: str,
                           evaluation_result: dict,
                           sources_sequences: list,
                           raw_predictions: list,
                           normalized_predictions: list,
                           ground_truths: list,
                           round_idx: int,
                           task_idx: int,
                           task_name: str):
    df = {
        "eval": evaluation_result,
        "prompts": sources_sequences,
        "results": normalized_predictions,
        "raw_results": raw_predictions,
        "labels": ground_truths
    }
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(
        output_dir,
        f"results-{round_idx}-{task_idx}-{task_name}.json"
    )
    with open(file_path, "w+", encoding="utf-8") as file:
        json.dump(df, file, ensure_ascii=False)


def postprocess_predictions(task: str, predictions: list) -> list:
    if task not in CHOICE_TASK_LABELS:
        return predictions
    labels = CHOICE_TASK_LABELS[task]
    pattern = CHOICE_TASK_REGEX[task]
    normalized = []
    for pred in predictions:
        text = str(pred).strip()
        match = pattern.search(text)
        if match:
            normalized.append(match.group(1).upper())
            continue
        first_valid = next((ch.upper() for ch in text if ch.upper() in labels), "")
        if first_valid:
            normalized.append(first_valid)
        else:
            normalized.append(text)
    return normalized


def main():
    args = parse_args()
    gpus = args.gpus.split(',')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print('CUDA_VISIBLE_DEVICES:', os.environ["CUDA_VISIBLE_DEVICES"])
    total_rank = len(gpus)
    
    # set_random_seed(args.seed)
    # device = torch.device("cuda")
    
    # set evaluation batch size
    # only support bs = 1, cause right padding training logic
    
    inference_tasks = (args.inference_tasks).split(',')
    task_num = len(inference_tasks)
    print("task_num: ", task_num, "inference_tasks: ", inference_tasks)
    # task_num: 8
    # inference_tasks: ['C-STANCE', 'FOMC', 'MeetingBank', 'Py150', 'ScienceQA', 'NumGLUE-cm', 'NumGLUE-ds', '20Minuten']
    start_round = copy.deepcopy(int(args.start_round))
    
    # del the start_round property in args:
    del args.start_round
    
    for round in range(start_round, task_num):  # load models and adapters of a new round in continual learning
        inference_model_path = os.path.join(args.inference_model_path, str(round))
        # print("Inference Model Path: " + inference_model_path, "local_rank" + args.local_rank)
        print("Inference Model Path: " + inference_model_path, "local_rank" + str(args.local_rank))
        
        # use command line of "deepspeed infer_part.py" to get the results:
        args.round = round
        args.total_rank = total_rank
        ranks = list(range(total_rank))
        # all_results_dic = run_inference(current_rank=, args)
        results = []
        # run one command:
        # run_inference(0, args)
        
        with ProcessPoolExecutor(max_workers=total_rank) as executor:
            futures = {executor.submit(run_inference, current_rank, args): current_rank for current_rank in ranks}
            
            for future in as_completed(futures):
                current_rank = futures[future]
                try:
                    result = future.result()
                    # print(f"RANK {current_rank} finished, result: {str(result)[:200]}")
                    # print blue:
                    print("\033[34m" + f"RANK {current_rank} finished, result: {str(result)[:200]}..." + "\033[0m")
                    
                    results.append((current_rank, result))
                except Exception as e:
                    print(f"RANK {current_rank} encountered an error: {e}")
        
        # for gpu_id, output in results:
        #     print(f"Output from RANK {gpu_id}: {output}")
        
        for inference_task_id in range(round + 1):  # evaluation for previous tasks in a single round
            inference_task = inference_tasks[inference_task_id]
            
            # sources_sequences, predicted_sequences, ground_truths = prediction(model, infer_dataloader)
            sources_sequences, predicted_sequences, ground_truths = [], [], []
            for rank, result in results:
                sources_sequences += ast.literal_eval(result[inference_task]['sources_sequences'])
                predicted_sequences += ast.literal_eval(result[inference_task]['predicted_sequences'])
                ground_truths += ast.literal_eval(result[inference_task]['ground_truths'])
            print(f'Task {inference_task_id} gathered, total len: {len(sources_sequences)}')
            
            base_dir = os.path.join('./check_output',
                                    f'{inference_task}_{args.CL_method}')
            os.makedirs(base_dir, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(base_dir, f'results_{timestamp}.json')

            # Prepare data to save
            normalized_predictions = postprocess_predictions(inference_task, predicted_sequences)

            save_data = {
                'sources_sequences': sources_sequences,
                'predicted_sequences': normalized_predictions,
                'raw_predicted_sequences': predicted_sequences,
                'ground_truths': ground_truths,
                'task_id': inference_task_id,
                'total_samples': len(sources_sequences)
            }

            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            print(f'Results saved to: {output_file}')

            if len(sources_sequences) == 0:
                warning = {"warning": "no predictions collected"}
                print(f"Warning: no predictions gathered for {inference_task}; skipping metric computation.")
                save_inference_results(
                    args.inference_output_path,
                    warning,
                    sources_sequences,
                    predicted_sequences,
                    normalized_predictions,
                    ground_truths,
                    round,
                    inference_task_id,
                    inference_task
                )
                continue

            # Get Accuracy/ROUGE/BLEU/...
            # The evaluation result is stored in a dictionary. e.g. {"accuracy": .., "rouge-L": ..}
            if inference_task == "ScienceQA":
                evaluation_result = eval_ScienceQA.eval(normalized_predictions, ground_truths)
            elif inference_task == "MeetingBank":
                evaluation_result = eval_MeetingBank.eval(normalized_predictions, ground_truths)
            elif inference_task == "C-STANCE":
                evaluation_result = eval_CStance.eval(normalized_predictions, ground_truths)
            elif inference_task == "Papyrus-f":
                evaluation_result = eval_PapyrusF.eval(normalized_predictions, ground_truths)
            elif inference_task == "Py150":
                evaluation_result = eval_Py150.eval(normalized_predictions, ground_truths)
            elif inference_task == "FOMC":
                evaluation_result = eval_FOMC.eval(normalized_predictions, ground_truths)
            elif inference_task == "NumGLUE-cm":
                evaluation_result = eval_NumGLUE_cm.eval(normalized_predictions, ground_truths)
            elif inference_task == "NumGLUE-ds":
                evaluation_result = eval_NumGLUE_ds.eval(normalized_predictions, ground_truths)
            elif inference_task == "20Minuten":
                evaluation_result = eval_20Minuten.eval(sources_sequences, normalized_predictions, ground_truths)
            elif inference_task == "amazon":
                evaluation_result = eval_amazon.eval(normalized_predictions, ground_truths)
            elif inference_task == "yelp":
                evaluation_result = eval_yelp.eval(normalized_predictions, ground_truths)
            elif inference_task == "agnews":
                evaluation_result = eval_agnews.eval(normalized_predictions, ground_truths)
            elif inference_task == "dbpedia":
                evaluation_result = eval_dbpedia.eval(normalized_predictions, ground_truths)
            elif inference_task == "yahoo":
                evaluation_result = eval_yahoo.eval(normalized_predictions, ground_truths)
            elif inference_task == "BoolQA":
                evaluation_result = eval_BoolQA.eval(normalized_predictions, ground_truths)
            elif inference_task == "QQP":
                evaluation_result = eval_QQP.eval(normalized_predictions, ground_truths)
            else:
                # default using accuracy
                evaluation_result = eval_QQP.eval(normalized_predictions, ground_truths)
            
            # if args.global_rank <= 0:  # only one process is running
            print("***** Saving inference results *****")
            save_inference_results(
                args.inference_output_path,
                evaluation_result,
                sources_sequences,
                predicted_sequences,
                normalized_predictions,
                ground_truths,
                round,
                inference_task_id,
                inference_task
            )


def run_inference(current_rank, args):
    # random sleep 0-3s:
    time.sleep(current_rank * 2 + random.random())
    
    args.current_rank = current_rank

    gpu_list = args.gpus.strip().split(',')
    current_gpu = gpu_list[current_rank]
    
    dir_path = args.output_json_file_path = args.inference_model_path + '/intermediate_predictions'
    print("mkdir -p " + dir_path)
    os.system("mkdir -p " + dir_path)
    
    args.output_json_file_path = args.inference_model_path + '/intermediate_predictions/round{}_rank{}'.format(args.round, current_rank) + '.json'
    command = [
        "deepspeed",
        f"--include=localhost:{current_gpu}",
        "--master_port",
        str(args.master_port + current_rank),
        "inference/infer_part.py",
        "--deepspeed"
    ]

    for key, value in vars(args).items():
        if key in ["master_port", "gpus"]:
            continue
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        elif value is None:
            continue
        else:
            command.extend([f"--{key}", str(value)])
    
    print("\033[31m" + " ".join(command) + "\033[0m")
    print('')

    attempts = 0
    while attempts < 3:
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            if completed.stdout:
                print(f"[rank {current_rank}] stdout: {completed.stdout.strip()[:2000]}")
            if completed.stderr:
                print(f"[rank {current_rank}] stderr: {completed.stderr.strip()[:2000]}")
        except subprocess.CalledProcessError as err:
            attempts += 1
            print(f"Fail for {attempts} times. Error: {err.stderr or err}")
            time.sleep(2)
            continue

        if not os.path.exists(args.output_json_file_path):
            attempts += 1
            print(f"Fail for {attempts} times. Error: missing output file {args.output_json_file_path}")
            time.sleep(2)
            continue

        with open(args.output_json_file_path, "r") as file:
            payload = file.read().strip()

        if not payload:
            attempts += 1
            print(f"Fail for {attempts} times. Error: empty inference result at {args.output_json_file_path}")
            time.sleep(2)
            continue

        try:
            return ast.literal_eval(payload)
        except Exception as parse_error:
            attempts += 1
            print(f"Fail for {attempts} times. Error parsing result: {parse_error}")
            time.sleep(2)

    raise RuntimeError(f"Inference command failed after {attempts} attempts for rank {current_rank}")


if __name__ == "__main__":
    main()
