# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=1

export PYTHONPATH=../../:$PYTHONPATH

python3.7 infer_ser.py \
    --model_type "layoutxlm" \
    --model_name_or_path "./output_v9/checkpoint-500/" \
    --max_seq_length 512 \
    --output_dir "output_res_v1/" \
    --infer_imgs "zh.val/img/" \
    --ocr_json_path "zh.val/xfun_normalize_val.json"
