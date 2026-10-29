from dataclasses import dataclass
from typing import Any, Optional

import yaml

from ...common.hparams import HyperParams
from ...common.registry import Registry


@Registry.register_implementation("method_name_mapping", "mend")
class MENDHyperParams(HyperParams):
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    inner_params: list[str]

    archive: Any

    # Method
    alg_name: str
    lr: float
    lr_lr: float
    lr_scale: float
    debug: bool
    cedit: float
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
    device: int
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

    alg_name: str
    model_name: str
    device: int

    batch_size: int = 1
    max_length: int = 40
    max_output_length: Optional[int] = None
    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None

    model_parallel: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config["alg_name"] == "MEND") or print(
            f"MENDHyperParams can not load from {hparams_name_or_path}, " f'alg_name is {config["alg_name"]} '
        )
        return cls(**config)
