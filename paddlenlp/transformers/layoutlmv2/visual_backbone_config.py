from yacs.config import CfgNode


def read_config(fp="visual_backbone.yaml"):
    with open(fp, "r") as fin:
        cfg = CfgNode().load_cfg(fin)
    cfg.freeze()
    return cfg
