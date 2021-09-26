# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=7

export PYTHONPATH=../../:$PYTHONPATH

# python3.7 -m paddle.distributed.launch \
#     --gpus="1" \
python3 train.py \
    --model_type "layoutxlm" \
    --model_name_or_path "./layoutxlm-base-paddle/" \
    --max_seq_length 512 \
    --train_data_dir "zh.train/" \
    --train_label_path "zh.train/xfun_normalize_train.json" \
    --eval_data_dir "zh.val/" \
    --eval_label_path "zh.val/xfun_normalize_val.json" \
    --label_map_path "../layoutlm_ser/labels/labels_ser.txt" \
    --num_train_epochs 200 \
    --eval_steps 50 \
    --save_steps 500 \
    --output_dir "output/re/"  \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --evaluate_during_training \
    --seed 2048