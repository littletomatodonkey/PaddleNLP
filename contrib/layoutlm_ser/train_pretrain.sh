export CUDA_VISIBLE_DEVICES=4,5,6,7

export PYTHONPATH=../../:$PYTHONPATH
# python3.7 train_pretrain.py \
# python3.7 -m paddle.distributed.launch --gpus '4,5,6,7' train_pretrain.py \

python3.7 -m paddle.distributed.launch --gpus '4,5,6,7' train_pretrain.py \
    --model_type "layoutxlm-pp" \
    --model_name_or_path "./layoutxlm-pp-base-paddle/" \
    --do_lower_case \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --num_train_epochs 200 \
    --logging_steps 10 \
    --save_steps 10000 \
    --output_dir "output/" \
    --labels "./SROIE/anno/labels.txt" \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --evaluate_during_training
