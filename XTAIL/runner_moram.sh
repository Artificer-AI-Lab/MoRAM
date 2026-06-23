#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

data_dir="${XTAIL_DATA_DIR:-./datasets}"

method="moram"
rank=16
enc=all
mod=qkvoinout

lr=5e-4
iters=500

temp=0.01
thre=0.2
topk=$rank

prefix="top${topk}_temp${temp}_thre${thre}_"
exp_no=${prefix}${method}_RAIL_iter${iters}_lr${lr}_${enc}_${mod}_r${rank}
model_ckpt_path=rail_ckpt/${exp_no}

CUDA_VISIBLE_DEVICES='0' \
python main.py --data_dir $data_dir \
    --lr=${lr} \
    --iterations $iters \
    --save $model_ckpt_path \
    --rank $rank \
    --target_encoder $enc \
    --target_modules_abbrev $mod \
    --method $method \
    --temp $temp \
    --topk $topk \
    --prune_thre $thre \
