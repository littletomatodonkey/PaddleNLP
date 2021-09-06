export CUDA_VISIBLE_DEVICES=3
python3.7 run_xfun_ser.py \
        --model_name_or_path "./layoutxlm-base-paddle/" \
        --output_dir ./output/ \
        --do_train \
        --do_eval \
        --lang zh \
        --max_steps 1000 \
        --warmup_ratio 0.1 \
