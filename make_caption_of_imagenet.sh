#!/bin/bash
num_gpus=8
num_jobs=0
pids=()
jsonl_paths=(
  "/home/oshita/vlm/LLaVA/sub_1.jsonl"
  "/home/oshita/vlm/LLaVA/sub_2.jsonl"
  "/home/oshita/vlm/LLaVA/sub_3.jsonl"
  "/home/oshita/vlm/LLaVA/sub_4.jsonl"
  "/home/oshita/vlm/LLaVA/sub_5.jsonl"
  "/home/oshita/vlm/LLaVA/sub_6.jsonl"
  "/home/oshita/vlm/LLaVA/sub_7.jsonl"
  "/home/oshita/vlm/LLaVA/sub_8.jsonl"
  )


for index in "${!jsonl_paths[@]}"; do
  jsonl_path="${jsonl_paths[$index]}"

  gpu_id1=$((num_jobs % num_gpus))
  # echo "GPU ID: $gpu_id"
  # pidにはプロセス番号が格納される
  CUDA_VISIBLE_DEVICES=$gpu_id1,$gpu_id2 python -m llava.serve.cli_for_cap --model-path liuhaotian/llava-v1.6-34b --jsonl_path $jsonl_path &

  ((num_jobs+=1))
  if ((num_jobs >= num_gpus)); then
      for pid in "${pids[@]}"; do
          wait $pid
      done
      num_jobs=0
      pids=()
  fi
done