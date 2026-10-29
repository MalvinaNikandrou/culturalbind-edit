import ast
from dataclasses import dataclass
from typing import Any, Optional, get_origin

import yaml
from pydantic import field_validator

from ...common.hparams import HyperParams
from ...common.registry import Registry


@Registry.register_implementation("method_name_mapping", "dpo_multimodal")
class DPOMultimodalHyperParams(HyperParams):
    task_name: str

    # Method
    alpha: float
    beta: float
    weight_decay: float
    num_steps: int
    grad_clip: float

    # Model
    name: str
    dtype: str
    half: bool

    # Layer selection
    rewrite_module_tmp: str

    # Output
    save: bool

    # Train
    opt: str
    device: str
    debug: bool
    eval_only: bool
    verbose: bool

    val_batch_size: int

    batch_size: int = 1
    max_length: int = 8192
    time_limit: Optional[float] = None
    alg_name: str = "DPO"
    early_stop: bool = True
    patience: int = 1

    sequential_edit: bool = False
    exact_match: bool = True
    test_generation: bool = False

    @property
    def run_name(self):
        layers = "-".join([str(layer) for layer in self.layers])
        return f"{self.alg_name}-{self.model_name}_layers{layers}_alpha{self.alpha}_earlystop{self.early_stop}_seed{self.seed}_bs{self.batch_size}_lr{self.edit_lr}"

    @property
    def requires_training(self):
        return False
