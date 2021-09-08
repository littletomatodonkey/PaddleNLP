export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=../../:$PYTHONPATH

python3.7 train.py \
    --model_type "layoutxlm" \
    --model_name_or_path "./layoutxlm-base-paddle/" \
    --do_lower_case \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --num_train_epochs 200 \
    --logging_steps 50 \
    --save_steps 500 \
    --output_dir "output_v8_paddle_data/" \
    --labels "./SROIE/anno/labels.txt" \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --evaluate_during_training
