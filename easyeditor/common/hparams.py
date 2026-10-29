import json
from typing import Optional

import yaml
from pydantic import BaseModel, field_validator


class HyperParams(BaseModel):
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    alg_name: str
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    seed: int = 42
    edit_lr: Optional[float] = None
    layers: Optional[list[int]] = None
    eval_only: bool
    test_generation: bool = False
    sequential_edit: bool = False
    results_root_dir: str = "results"
    device: str = "cuda"
    exact_match: bool = True

    model_config = {
        "extra": "forbid",  # Raise error on unknown fields
        "str_strip_whitespace": True,
        "arbitrary_types_allowed": True,
    }

    @field_validator("layers", mode="before")
    @classmethod
    def parse_layers(cls, v):
        if isinstance(v, str):
            import ast

            return ast.literal_eval(v)
        return v

    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)

        return cls(**data)

    def construct_float_from_scientific_notation(config: dict):
        for key, value in config.items():
            if isinstance(value, str):
                if value.lower() == "none":
                    config[key] = None
                else:
                    try:
                        # Convert scalar to float if it is in scientific notation format
                        config[key] = float(value)
                    except ValueError:
                        pass

        return config

    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[dict] = None):
        if not path.endswith(".yaml"):
            path = f"{path}.yaml"

        with open(path) as f:
            config = yaml.safe_load(f)

        if overrides:
            config.update(overrides)

        return cls(**config)  # Pydantic handles all the coercion

    def to_dict(config) -> dict:
        return config.model_dump()

    @property
    def run_name(self):
        return f"{self.alg_name}-{self.model_name}_layers{self.layers}_seed{self.seed}_lr{self.edit_lr}_sequential{self.sequential_edit}"

    @property
    def results_dir(self):
        return f"{self.results_root_dir}/{self.run_name}"
