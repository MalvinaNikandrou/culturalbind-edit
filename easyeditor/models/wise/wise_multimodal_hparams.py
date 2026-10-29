import ast
from typing import Literal, Optional, Union

from pydantic import field_validator

from ...common.hparams import HyperParams
from ...common.registry import Registry


@Registry.register_implementation("method_name_mapping", "wise_multimodal")
class WISEMultimodalHyperParams(HyperParams):
    hidden_act: str

    # Experiments
    sequential_edit: bool
    edit_lr: float
    n_iter: int
    # Method
    objective_optimization: str
    mask_ratio: float
    act_margin: list[float]  # [alpha, beta, gamma]

    act_ratio: float
    merge_freq: int
    retrieve: bool
    replay: bool
    save_freq: Union[int, None]
    merge_alg: str
    norm_constraint: float
    # Module templates
    rewrite_module_tmp: str
    layers: list[int]
    weights: Union[float, None]
    densities: Union[float, None]

    device: int
    name: str
    alg_name: Literal["WISE"]
    dtype: str

    # Multimodal
    task_name: str
    qformer_checkpoint: str
    qformer_name_or_path: str
    state_dict_file: str

    # Image_dir
    coco_image: str
    rephrase_image: str
    file_type: str  # single image or video

    # Defaults
    batch_size: int = 1
    max_length: int = 30
    training_locality: Literal["text", "multimodal"] = "text"
    model_parallel: bool = False
    use_chat_template: bool = False
    exact_match: bool = True
    test_generation: bool = False
    pretrained_ckpt: Optional[str] = None
    compute_post_eval_metrics: bool = False

    # Save and Load
    save_path: str = None
    load_path: str = None

    @property
    def run_name(self):
        return f"WISE-{self.model_name}"

    @property
    def requires_training(self):
        return False

    @field_validator("act_margin", mode="before")
    @classmethod
    def check_act_margin(cls, value: list[float]) -> list[float]:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if len(value) != 3:
            raise ValueError("act_margin must be a list of three floats: [alpha, beta, gamma]")
        return value

    @property
    def alpha(self) -> float:
        return self.act_margin[0]

    @property
    def beta(self) -> float:
        return self.act_margin[1]

    @property
    def gamma(self) -> float:
        return self.act_margin[2]

    @property
    def inner_params(self) -> list[str]:
        return [self.rewrite_module_tmp.format(layer) for layer in self.layers]
