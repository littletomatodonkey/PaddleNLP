# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.
import numpy as np
import cv2
import paddle
from resnet import ResNet
from yacs.config import CfgNode
from yolo_fpn import PPYOLOPAN
def read_config(fp=None):
    with open(fp, "r") as fin:
        cfg = CfgNode().load_cfg(fin)
    cfg.freeze()
    return cfg
    
def main():
    img = cv2.imread("./road554.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     resize_h, resize_w = 640, 640
    resize_h, resize_w = 224, 224
    im_shape = img.shape[0:2]
    im_scale_y = resize_h / im_shape[0]
    im_scale_x = resize_w / im_shape[1]
    img_new = cv2.resize(img, None, None, 
                         fx=im_scale_x, fy=im_scale_y, 
                         interpolation=2)
    img_new = img_new.astype(np.float32, copy=False)
    mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
    std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]
    img_new = img_new / 255.0
    img_new -= mean
    img_new /= std
    img_new = img_new.transpose((2, 0, 1))
    img_new = img_new[np.newaxis, :]
    img_paddle = paddle.to_tensor(img_new)
    print(img_paddle.numpy()[0, 0, 0, 0:10])
    sys.exit(-1)

    cfg = read_config("./ppyolov2_r50vd_dcn.yml")
    depth = cfg.ResNet.depth
    variant = cfg.ResNet.variant
    return_idx = cfg.ResNet.return_idx
    dcn_v2_stages = cfg.ResNet.dcn_v2_stages
    freeze_at = cfg.ResNet.freeze_at
    freeze_norm = cfg.ResNet.freeze_norm
    norm_decay = cfg.ResNet.norm_decay
    norm_type = cfg.norm_type
    resnet = ResNet(depth=depth, variant=variant, return_idx=return_idx,
                   dcn_v2_stages=dcn_v2_stages, freeze_at=freeze_at,
                   freeze_norm=freeze_norm, norm_decay=norm_decay,
                   norm_type=norm_type)

    drop_block = cfg.PPYOLOPAN.drop_block
    block_size = cfg.PPYOLOPAN.block_size
    keep_prob = cfg.PPYOLOPAN.keep_prob
    spp = cfg.PPYOLOPAN.spp        
    fpn = PPYOLOPAN(drop_block=drop_block, block_size=block_size,
                    keep_prob=keep_prob, spp=spp)
    info = paddle.load("./publaynet_ppyolov2/ppyolov2_r50vd_dcn_365e_publaynet_pretrained.pdparams")
    resnet_state_dict = resnet.state_dict()
    fpn_state_dict = fpn.state_dict()

    trans_info = {}
    for key in resnet_state_dict:
        trans_info[key] = info["backbone." + key]
    for key in fpn_state_dict:
        trans_info[key] = info["neck." + key]

    resnet.load_dict(trans_info)
    fpn.load_dict(trans_info)
    resnet.eval()
    fpn.eval()

    body_feats = resnet({'image':img_paddle})
    outs = fpn(body_feats)
    for lno in range(len(outs)):
        print("New:", outs[lno].shape, outs[lno][0, 0, 0, 0:10])

if __name__ == "__main__":
    main()