from dataclasses import dataclass
from typing import Literal, Optional

import yaml

from ...common.hparams import HyperParams
from ...common.registry import Registry


@Registry.register_implementation("method_name_mapping", "grace")
class GraceHyperParams(HyperParams):
    # Experiments

    edit_lr: int
    n_iter: int
    # Method
    eps: float
    dist_fn: str
    val_init: str
    val_train: str
    val_reg: str
    reg: str
    replacement: str
    eps_expand: str
    num_pert: str
    dropout: float

    # Module templates
    inner_params: list[str]
    device: int
    alg_name: Literal["GRACE"]
    model_name: str

    # Defaults
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False


@Registry.register_implementation("method_name_mapping", "grace_multimodal")
class GraceMultimodalHyperParams(HyperParams):
    # Experiments
    task_name: str

    n_iter: int
    # Method
    eps: float
    dist_fn: str
    val_init: str
    val_train: str
    val_reg: str
    reg: str
    replacement: str
    eps_expand: str
    num_pert: int
    dropout: float

    # Module templates
    layers: list[int]
    rewrite_module_tmp: str
    device: int
    name: str
    alg_name: Literal["GRACE"]
    dtype: str

    # Defaults
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False
    sequential_edit: bool = False
    exact_match: bool = True
    test_generation: bool = False
    compute_post_eval_metrics: bool = False

    @property
    def run_name(self):
        return f"GRACE-{self.model_name}"

    @property
    def requires_training(self):
        return False

    @property
    def inner_params(self) -> list[str]:
        return [self.rewrite_module_tmp.format(layer) for layer in self.layers]
