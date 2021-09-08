export CUDA_VISIBLE_DEVICES=0,1,2,3

export PYTHONPATH=../../:$PYTHONPATH
# python3.7 train_pretrain_model.py \
#     --data_dir "./data/yanbao_v1/sample_box.txt" \
#     --model_type "erniegram" \
#     --model_name_or_path "ernie-gram-zh" \
#     --do_lower_case \
#     --max_seq_length 512 \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 300 \
#     --logging_steps 10 \
#     --save_steps 2000 \
#     --output_dir "output/" \
#     --per_gpu_train_batch_size 16 \
#     --per_gpu_eval_batch_size 16 \
#     --evaluate_during_training 

# python3.7 -m paddle.distributed.launch --gpus '0,1,2,3' train_pretrain_model.py \
#     --data_dir "./data/yanbao_v1/train_box.txt" \
#     --model_type "erniegram" \
#     --model_name_or_path "ernie-gram-zh" \
#     --do_lower_case \
#     --max_seq_length 512 \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 300 \
#     --logging_steps 10 \
#     --save_steps 5000 \
#     --output_dir "output/" \
#     --per_gpu_train_batch_size 16 \
#     --per_gpu_eval_batch_size 16 \
#     --evaluate_during_training

# python3.7 train_pretrain_model.py \
#     --data_dir "./data/yanbao_v1/yanbao_v1_train.h5" \
#     --model_type "erniegram" \
#     --model_name_or_path "ernie-gram-zh" \
#     --do_lower_case \
#     --max_seq_length 512 \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 300 \
#     --logging_steps 10 \
#     --save_steps 30 \
#     --output_dir "output/" \
#     --per_gpu_train_batch_size 16 \
#     --per_gpu_eval_batch_size 16 \
#     --evaluate_during_training

#    --model_name_or_path "ernie-gram-zh" \

python3.7 -m paddle.distributed.launch --gpus '0,1,2,3' train_pretrain_model.py \
    --data_dir "./data/yanbao_v1/yanbao_v1_train.h5" \
    --model_type "erniegram" \
    --model_name_or_path "./erniegram-yanbao-5000/" \
    --do_lower_case \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --num_train_epochs 300 \
    --logging_steps 10 \
    --save_steps 5000 \
    --output_dir "output/" \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --evaluate_during_training 