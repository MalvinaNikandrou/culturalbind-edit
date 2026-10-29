from typing import Literal

from ...common.hparams import HyperParams
from ...common.registry import Registry


@Registry.register_implementation("method_name_mapping", "dpo")
class DPOHyperParams(HyperParams):
    # Method
    lora_type: str
    layers: list[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float
    target_modules: list[str]
    rank: int
    lora_alpha: float
    lora_dropout: float
    # Module templates

    device: int
    alg_name: Literal["DPO"]
    model_name: str

    # Defaults
    batch_size: int = 128
    max_length: int = 40
    model_parallel: bool = False
    alpha: float = 0.1
    beta: float = 0.1
