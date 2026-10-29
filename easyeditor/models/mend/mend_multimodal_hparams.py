import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

import yaml

from ...common.hparams import HyperParams
from ...common.registry import Registry


@Registry.register_implementation("method_name_mapping", "mend_multimodal")
class MENDMultimodalHparams(HyperParams):

    # Multimodal
    qformer_name_or_path: str
    state_dict_file: str

    # Image_dir
    coco_image: str
    rephrase_image: str

    # Model
    name: str
    inner_params: list[str]
    dtype: str

    # Method
    alg_name: str
    lr: float
    lr_lr: float
    lr_scale: float
    debug: bool
    cedit: float
    iedit: float
    cloc: float
    cbase: float
    dropout: float
    train_base: bool
    no_grad_layers: Any
    one_sided: bool
    n_hidden: int
    hidden_dim: Any
    init: str
    norm: bool
    combine: bool
    x_only: bool
    delta_only: bool
    act: str
    rank: int
    mlp_class: str
    shared: bool

    # Train
    device: str
    model_save_pt: int
    silent: bool
    log_interval: int
    eval_log_interval: int
    final_eval: bool
    val_interval: int
    early_stop_patience: int
    early_stop_key: str
    eval_only: bool
    half: bool
    save: bool
    verbose: bool

    val_batch_size: int
    accumulate_bs: int
    val_steps: int
    opt: str
    grad_clip: float

    alg_name: Literal["MEND"]

    exact_match: bool = True
    sequential_edit: bool = False
    test_generation: bool = False
    archive: str | None = None
    batch_size: int = 1
    max_length: int = 30
    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None
    model_parallel: bool = False
    qformer_checkpoint: Optional[str] = None
    freeze_qformer: bool = True
    pretrained_ckpt: Optional[str] = None
    compute_post_eval_metrics: bool = False

    @property
    def run_name(self):
        inner_param_layers = [re.findall(r"\d+", inner_param) for inner_param in self.inner_params]
        # Flatten the list of lists and convert to a set to remove duplicates
        unique_layers = set(layer for sublist in inner_param_layers for layer in sublist)
        # Sort the unique layers
        unique_layers = sorted(unique_layers, key=int)
        layers = "-".join(unique_layers)
        return f"{self.alg_name}-{self.model_name}-layers{layers}-maxiters{self.max_iters}-lr{self.lr}"

    @property
    def requires_training(self):
        return True
