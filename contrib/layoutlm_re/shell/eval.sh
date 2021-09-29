# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=6

export PYTHONPATH=../../:$PYTHONPATH

python3 eval.py \
    --model_type "layoutxlm" \
    --model_name_or_path "./layoutxlm-base-paddle/checkpoint-best/" \
    --max_seq_length 512 \
    --eval_data_dir "zh.val/" \
    --eval_label_path "zh.val/xfun_normalize_val.json" \
    --label_map_path "../layoutlm_ser/labels/labels_ser.txt" \
    --per_gpu_eval_batch_size 8 \
    --output_dir 'output/re'