import paddle
import numpy as np
layoutxmlpp_sample_path = "./layoutxmlpp_sample.pdparams"
layoutxmlpp_dict = paddle.load(layoutxmlpp_sample_path)
layoutxml_model_path = "../layoutlm_ser/layoutxlm-base-paddle/model_state.pdparams"
layoutxml_dict = paddle.load(layoutxml_model_path)
ppyolov2_model_path = "publaynet_ppyolov2/ppyolov2_r50vd_dcn_365e_publaynet_pretrained.pdparams"
ppyolov2_dict = paddle.load(ppyolov2_model_path)

for key in layoutxml_dict:
    print("layoutxml_dict:", key)

for key in ppyolov2_dict:
    print("ppyolov2_dict:", key)
    
for key in layoutxmlpp_dict:
    sub_keys = key.split(".")
    text_subkey = ''
    det_subkey = ''
    if sub_keys[0] == "layoutlmv2" and sub_keys[1] == "visual":
        if sub_keys[2] == "resnet":
            det_subkey = 'backbone'
        elif sub_keys[2] == "fpn":
            det_subkey = 'neck'
        for sno in range(3, len(sub_keys)):
            det_subkey = det_subkey + "." + sub_keys[sno]
    elif sub_keys[0] == "layoutlmv2":
        text_subkey = sub_keys[1]
        for sno in range(2, len(sub_keys)):
            text_subkey = text_subkey + "." + sub_keys[sno]
        
    if text_subkey in layoutxml_dict:
        layoutxmlpp_dict[key] = layoutxml_dict[text_subkey]
    elif det_subkey in ppyolov2_dict:
        layoutxmlpp_dict[key] = ppyolov2_dict[det_subkey]
    else:
        print("layoutxmlpp_dict:", key, text_subkey, det_subkey)
paddle.save(layoutxmlpp_dict, "model_state.pdparams")