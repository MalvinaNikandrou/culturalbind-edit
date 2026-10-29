import ast
from dataclasses import dataclass
from typing import Any, Literal, Optional

import yaml

from ...common.hparams import HyperParams
from ...common.registry import Registry


@Registry.register_implementation("method_name_mapping", "finetune_multimodal")
class FTMultimodalHyperParams(HyperParams):
    # Multimodal
    state_dict_file: str

    # Image_dir
    coco_image: str
    rephrase_image: str

    # Model
    name: str
    norm_constraint: float
    weight_decay: float

    # Layer selection
    layers: list[int]
    rewrite_module_tmp: str
    train_base: bool
    freeze_language_model: bool
    freeze_vision_encoder: bool
    freeze_projector: bool
    dropout: float

    # Method
    grad_clip: float
    num_steps: int
    kl_factor: float

    # Loss weights
    cedit: float
    iedit: float
    cloc: float
    cbase: float
    init_std: float
    dtype: str

    # Module templates
    objective_optimization: str

    # Output
    save: bool
    model_save_pt: int

    # Train
    silent: bool
    log_interval: int
    eval_log_interval: int
    final_eval: bool
    val_interval: int
    early_stop_patience: int
    early_stop_key: str
    debug: bool
    eval_only: bool
    half: bool
    verbose: bool

    val_batch_size: int
    accumulate_bs: int
    val_steps: int
    opt: str

    locality_enabled: bool = False
    no_grad_layers: Optional[list[int]] = None
    batch_size: int = 1
    max_length: int = 8192
    max_iters: int = 25
    model_parallel: bool = False
    archive: Any = None
    alg_name: Literal["FT"] = "FT"

    @property
    def run_name(self):
        layers = "-".join([str(layer) for layer in self.layers])
        return f"{self.alg_name}-{self.model_name}_layers{layers}_seed{self.seed}_bs{self.batch_size}_lr{self.edit_lr}_{self.rewrite_module_tmp}"

    @property
    def requires_training(self):
        return False
