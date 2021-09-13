from .. import PretrainedModel, register_base_model
from ..layoutlmv2_pp.modeling import LayoutLMv2PPModel, LayoutLMv2PPForTokenClassification, LayoutLMv2PPForPretraining

__all__ = [
    "LayoutXLMPPModel",
    "LayoutXLMPPForTokenClassification",
    "LayoutXLMPPForPretraining",
]


@register_base_model
class LayoutXLMPPModel(LayoutLMv2PPModel):
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "layoutxlm-base": {
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 0,
            "coordinate_size": 128,
            "eos_token_id": 2,
            "fast_qkv": False,
            "gradient_checkpointing": False,
            "has_relative_attention_bias": False,
            "has_spatial_attention_bias": False,
            "has_visual_segment_embedding": True,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "image_feature_pool_shape": [7, 7, 256],
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-05,
            "max_2d_position_embeddings": 1024,
            "max_position_embeddings": 514,
            "max_rel_2d_pos": 256,
            "max_rel_pos": 128,
            "model_type": "layoutlmv2",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "output_past": True,
            "pad_token_id": 1,
            "shape_size": 128,
            "rel_2d_pos_bins": 64,
            "rel_pos_bins": 32,
            "type_vocab_size": 1,
            "vocab_size": 250002,
            "tokenizer_class": "XLMRobertaTokenizer",
            "init_class": "LayoutXLMPPModel"
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "layoutxlm-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/layoutxlm-base.pdparams",
        }
    }
    base_model_prefix = "layoutlmv2"


@register_base_model
class LayoutXLMPPForTokenClassification(LayoutLMv2PPForTokenClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@register_base_model
class LayoutXLMPPForPretraining(LayoutLMv2PPForPretraining):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)