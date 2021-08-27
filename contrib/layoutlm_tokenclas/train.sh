export CUDA_VISIBLE_DEVICES=3

export PYTHONPATH=../../:$PYTHONPATH

python3.7 train.py \
    --data_dir "./SROIE/anno" \
    --model_type "layoutlm" \
    --model_name_or_path "./layoutlm-base-uncased-paddle" \
    --do_lower_case \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --num_train_epochs 50 \
    --logging_steps 10 \
    --save_steps 500 \
    --output_dir "output/" \
    --labels "./SROIE/anno/labels.txt" \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --evaluate_during_training
