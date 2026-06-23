import argparse
import json

import numpy as np

# task_metric = {
#     "C-STANCE"   : "accuracy",
#     "FOMC"       : "accuracy",
#     "MeetingBank": "rouge-L",
#     "Py150"      : "similarity",
#     "ScienceQA"  : "accuracy",
#     "NumGLUE-cm" : "accuracy",
#     "NumGLUE-ds" : "accuracy",
#     "20Minuten"  : "sari"
# }
task_metric = {
    "C-STANCE"   : "accuracy",
    "FOMC"       : "accuracy",
    "MeetingBank": "rouge-L",
    "Py150"      : "similarity",
    "ScienceQA"  : "accuracy",
    "NumGLUE-cm" : "accuracy",
    "NumGLUE-ds" : "accuracy",
    "20Minuten"  : "sari",
    # Add new tasks here
    "yelp"       : "accuracy",
    "amazon" : "accuracy",
    "dbpedia": "accuracy",
    "yahoo"  : "accuracy",
    "agnews"     : "accuracy",
    "MNLI"   : "accuracy",
    "QQP"    : "accuracy",
    "RTE"    : "accuracy",
    "SST-2"  : "accuracy",
    "WiC"    : "accuracy",
    "CB"     : "accuracy",
    "COPA"   : "accuracy",
    "BoolQA": "accuracy",
    "MultiRC": "accuracy",
    "IMDB"   : "accuracy",
    # Add new tasks here
}


# MNLI,CB,WIC,COPA,QQP,BoolQA,RTE,IMDB,yelp,amazon,SST-2,dbpedia,agnews,MultiRC,yahoo

# C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten,dbpedia,amazon,yahoo,agnews,yelp,BoolQA,QQP


def _scaled_metric_from_json(tmp_json, eval_task, current_file_name):
    """
    Return primary metric on 0–100 scale, or np.nan if missing / empty run.
    """
    ev = tmp_json.get("eval")
    if not isinstance(ev, dict):
        print(f"\tWarning: {current_file_name}: invalid or missing 'eval'")
        return np.nan

    mkey = task_metric[eval_task]
    if mkey not in ev:
        msg = ev.get("warning", f"missing metric key '{mkey}'")
        print(f"\tWarning: {current_file_name}: {msg}; eval keys={list(ev.keys())}")
        return np.nan

    try:
        if mkey == "sari":
            metric = ev[mkey][0]["sari"]
            metric = metric / 100.0
        elif mkey == "similarity":
            metric = ev[mkey]
            metric = metric / 100.0
        else:
            metric = ev[mkey]
    except (KeyError, TypeError, IndexError) as e:
        print(f"\tWarning: {current_file_name}: cannot read metric '{mkey}' ({e}); keys={list(ev.keys())}")
        return np.nan

    return float(metric) * 100.0


def parse_results_dir(dir_path, inference_tasks):
    num_tasks = len(inference_tasks)
    results = np.zeros((num_tasks, num_tasks))
    all_results = []
    
    for task_id, task in enumerate(inference_tasks):
        for eval_id in range(task_id + 1):
            current_file_name = f"results-{task_id}-{eval_id}-{inference_tasks[eval_id]}.json"
            print('\t' + current_file_name)
            # read json file:
            with open(f"{dir_path}/{current_file_name}", "r") as f:
                tmp_json = json.load(f)
            metric = _scaled_metric_from_json(
                tmp_json, inference_tasks[eval_id], current_file_name
            )
            print(metric)
            results[eval_id, task_id] = metric
            all_results.append(metric)
    
    # # print a beautiful table:
    # print('\t\t\t' + '\t'.join(inference_tasks))
    # for i in range(num_tasks):
    #     formatted_results = [f"{x:.2f}" for x in results[i, :]]
    #     if len(inference_tasks[i]) < 8:
    #         print('\t', end='')
    #     print(inference_tasks[i] + '\t\t' + '\t\t'.join(formatted_results))
    
    table_str = ''
    table_str += ('Results of {}'.format(dir_path)) + '\n'
    column_headers = inference_tasks
    row_headers = inference_tasks
    num_tasks = len(inference_tasks)

    def _fmt_cell(x):
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return "nan"
        return f"{x:.2f}"

    formatted_results = [[_fmt_cell(x) for x in row] for row in results]
    max_row_header_width = max(len(rh) for rh in row_headers)
    col_widths = []
    for i in range(num_tasks):
        max_data_width = max(len(r[i]) for r in formatted_results) if num_tasks > 0 else 0
        max_col_width = max(len(column_headers[i]), max_data_width)
        col_widths.append(max_col_width)
    header_str = " " * max_row_header_width + "\t" + "\t".join(c.ljust(col_widths[i]) for i, c in enumerate(column_headers))
    table_str += (header_str) + '\n'
    table_str += ("-" * (max_row_header_width + sum(col_widths) + (num_tasks) * 4)) + '\n'
    for i in range(num_tasks):
        row_str = row_headers[i].ljust(max_row_header_width)
        row_data_str = "\t".join(formatted_results[i][j].rjust(col_widths[j]) for j in range(num_tasks))
        table_str += (row_str + "\t" + row_data_str) + '\n'
    
    table_str += (f"All Average: {np.nanmean(all_results):.4f}") + '\n'
    table_str += (f"Last Average: {np.nanmean(results[:, num_tasks - 1]):.4f}") + '\n'
    
    # calculate the BWT (Backward Transfer):
    BWT = 0.0
    bwt_n = 0
    for i in range(num_tasks):
        r_ii = results[i, i]
        r_last = results[i, num_tasks - 1]
        if np.isnan(r_ii) or np.isnan(r_last):
            continue
        BWT += min(r_last - r_ii, 0)
        bwt_n += 1
    BWT = BWT / bwt_n if bwt_n else float("nan")
    
    table_str += (f"BWT: {BWT:.4f}") + '\n'
    
    print(table_str)
    # write the table_str to a txt file:
    with open(f"{dir_path}/final_results_new.txt", "w") as f:
        f.write(table_str)
        


if __name__ == '__main__':
    # inference_tasks = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
    # inference_tasks = ["ScienceQA", "NumGLUE-cm", "NumGLUE-ds"]
    # inference_tasks = ["agnews","dbpedia","yelp", "yahoo", "amazon"]
    # inference_tasks = ["dbpedia", "amazon", "yahoo", "agnews"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        # required=True,
                        help='Path to the training dataset, a single data path.')
    parser.add_argument('--inference_tasks',
                        type=str,
                        default='C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten',
                        # required=True,
                        help='The tasks to be evaluated, separated by a comma.')
    
    args = parser.parse_args()
    
    inference_tasks = str(args.inference_tasks).split(',')
    dir_path = args.data_path
    # dir_path = "./outputs_LLM-CL/cl/Mistral-7B-Instruct-v0.3/Tree_LoRA_1224_123449/predictions"
    
    parse_results_dir(dir_path, inference_tasks)
