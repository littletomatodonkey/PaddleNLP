export CUDA_VISIBLE_DEVICES=5

export PYTHONPATH=../../:$PYTHONPATH

# python3.7 -m paddle.distributed.launch --gpus '0,1,2,3' train_xlm.py \
python3.7 train_xlm.py \
    --model_type "layoutxlm" \
    --model_name_or_path "./layoutxlm-base-paddle/" \
    --max_seq_length 512 \
    --train_data_dir "dataset/" \
    --train_label_path "dataset/json_all224.txt" \
    --num_train_epochs 20 \
    --eval_steps 10 \
    --save_steps 30000 \
    --output_dir "output_layoutxlm_v1/" \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
