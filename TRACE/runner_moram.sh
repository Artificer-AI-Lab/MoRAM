#!/usr/bin/env bash
# End-to-end TRACE: train -> infer -> collect metrics (MoRAM).
# Run from the TRACE/ directory after activating your environment.

set -euo pipefail
cd "$(dirname "$0")"

# Optional: custom Hugging Face cache (defaults to ~/.cache/huggingface if unset)
if [ -n "${HF_HOME:-}" ]; then
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export TRANSFORMERS_CACHE="${HF_HOME}"
fi
export TOKENIZERS_PARALLELISM=false

now=$(date +"%m%d_%H%M%S")
gpu_nodes="${TRACE_GPUS:-0,1,2,3}"
master_port="${TRACE_MASTER_PORT:-25011}"

model_name="${TRACE_MODEL:-google/Gemma-2B-it}"
# Filesystem-safe tag (HF model ids contain "/")
model_tag="${model_name//\//__}"

epochs="2,1,3,2,1,2,2,3"
lr=5e-4

moram_rank=8
moram_topk=$moram_rank
moram_router_temp=0.03
moram_infer_lora_a_thresh=0.2
dataset="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten"
epochs_tag="${epochs//,/-}"
run_tag="MoRAM_rank${moram_rank}_topk${moram_topk}_lr${lr}_epochs${epochs_tag}_temp${moram_router_temp}_thre${moram_infer_lora_a_thresh}_${now}"
output_root="./outputs_LLM-CL/cl/${model_tag}/${run_tag}"

echo "===== Training MoRAM on TRACE benchmark ====="
deepspeed --include=localhost:$gpu_nodes --master_port "$master_port" training/main.py \
    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name $dataset \
    --model_name_or_path $model_name \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --learning_rate $lr \
    --weight_decay 0. \
    --num_train_epochs $epochs \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method MoRAM \
    --moram_rank $moram_rank \
    --moram_topk $moram_topk \
    --moram_router_temp $moram_router_temp \
    --moram_infer_lora_a_thresh $moram_infer_lora_a_thresh \
    --output_dir $output_root

echo "===== Inference with trained MoRAM adapters ====="
python inference/infer_multi_command.py \
    --gpus $gpu_nodes \
    --master_port "$master_port" \
    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
    --inference_tasks $dataset \
    --model_name_or_path $model_name \
    --inference_model_path $output_root \
    --inference_batch 1 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --CL_method MoRAM \
    --moram_topk $moram_topk \
    --moram_router_temp $moram_router_temp \
    --moram_infer_lora_a_thresh $moram_infer_lora_a_thresh \
    --inference_output_path ${output_root}/predictions

echo "===== Collecting evaluation metrics ====="
python inference/collect_results.py \
    --inference_tasks $dataset \
    --data_path ${output_root}/predictions

echo "Done. Outputs saved to ${output_root}"
