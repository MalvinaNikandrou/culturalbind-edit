import re
from dataclasses import dataclass
from typing import Literal, Optional

import yaml

from ...common.hparams import HyperParams
from ...common.registry import Registry


@Registry.register_implementation("method_name_mapping", "ike")
class IKEHyperParams(HyperParams):

    # Module templates
    device: int
    alg_name: Literal["IKE"]
    model_name: str
    sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Method
    k: int = 16  # K icl examples
    use_icl_examples: bool = True

    model_parallel: bool = False


@Registry.register_implementation("method_name_mapping", "ike_multimodal")
class IKEMultimodalHyperParams(HyperParams):
    # Method
    k: int  # K icl examples

    # Module templates
    device: int
    name: str
    alg_name: Literal["IKE"]
    sentence_model_name: str
    eval_only: bool
    dtype: str

    ## Multimodal
    task_name: str
    qformer_checkpoint: str
    qformer_name_or_path: str
    state_dict_file: str

    # Image_dir
    coco_image: str
    rephrase_image: str
    exact_match: bool = True
    test_generation: bool = False
    sequential_edit: bool = False
    pretrained_ckpt: Optional[str] = None
    compute_post_eval_metrics: bool = False

    @property
    def run_name(self):
        return f"IKE-{self.model_name}"

    @property
    def requires_training(self):
        return True
