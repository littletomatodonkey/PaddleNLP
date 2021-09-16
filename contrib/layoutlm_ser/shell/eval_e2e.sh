
gt="./zh.val/xfun_normalize_val.json"

dt="output_e2e_v1/infer_results.txt"

python3.7 eval/eval_end2end.py \
    --gt_json_path="${gt}" \
    --pred_json_path="${dt}" \
    --ignore_background=True \
    --ignore_ser_prediction=True
