export CUDA_VISIBLE_DEVICES=0,1,2,3

export PYTHONPATH=../../:$PYTHONPATH
# python3.7 train_funsd.py \
#     --data_dir "./data/XFUN_v1.0_data/zh/" \
#     --model_type "layoutlm" \
#     --model_name_or_path "./layoutlm-base-uncased-paddle" \
#     --do_lower_case \
#     --max_seq_length 512 \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 50 \
#     --logging_steps 10 \
#     --save_steps 500 \
#     --output_dir "output/" \
#     --labels "./data/XFUN_v1.0_data/zh/labels.txt" \
#     --per_gpu_train_batch_size 16 \
#     --per_gpu_eval_batch_size 16 \
#     --evaluate_during_training

# python3.7 train_funsd_bert.py \
#     --data_dir "./data/XFUN_v1.0_data/zh/" \
#     --model_type "bert" \
#     --model_name_or_path "simbert-base-chinese" \
#     --do_lower_case \
#     --max_seq_length 512 \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 20 \
#     --logging_steps 10 \
#     --save_steps 500 \
#     --output_dir "output/" \
#     --labels "./data/XFUN_v1.0_data/zh/labels.txt" \
#     --per_gpu_train_batch_size 16 \
#     --per_gpu_eval_batch_size 16 \
#     --evaluate_during_training

# python3.7 train_funsd_common.py \
#     --data_dir "./data/XFUN_v1.0_data/zh/" \
#     --model_type "erniegram" \
#     --model_name_or_path "./success_models/erniegram_yanbaov1/checkpoint-40000/" \
#     --do_lower_case \
#     --max_seq_length 512 \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 300 \
#     --logging_steps 10 \
#     --save_steps 2000 \
#     --output_dir "output/" \
#     --labels "./data/XFUN_v1.0_data/zh/labels.txt" \
#     --per_gpu_train_batch_size 16 \
#     --per_gpu_eval_batch_size 16 \
#     --evaluate_during_training 

#base: "erniegram" "ernie-gram-zh"
# python3.7 train_funsd_common.py \
# python3.7 -m paddle.distributed.launch --gpus '0,1,2,3' train_funsd_common.py \
python3.7 train_funsd_common.py \
    --data_dir "./data/XFUN_v1.0_data/zh/" \
    --model_type "erniegram" \
    --model_name_or_path "ernie-gram-zh" \
    --do_lower_case \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 1000 \
    --logging_steps 10 \
    --save_steps 2000 \
    --output_dir "output/" \
    --labels "./data/XFUN_v1.0_data/zh/labels.txt" \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --evaluate_during_training 
