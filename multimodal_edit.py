import argparse
from dataclasses import dataclass
from typing import Optional
import wandb

from easyeditor import Registry, run_multimodal_ke_training, run_multimodal_ke_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Run multimodal editing experiment")
    parser.add_argument("--project_name", type=str, default="multimodal_knowledge_editing_generation_eval")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity (defaults to your wandb default)")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="wandb mode; if unset, wandb uses the WANDB_MODE environment variable (default: online)",
    )
    parser.add_argument("--knowledge_editing_method", type=str)
    parser.add_argument("--knowledge_editing_hparams_config", type=str)
    parser.add_argument("--task_name", type=str, default="crope", choices=["crope"])
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--train_size", type=int)
    parser.add_argument("--eval_dataset", type=str)
    parser.add_argument("--eval_size", type=int)
    parser.add_argument("--edit_lr", type=float)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return ExperimentHparams(**vars(args))


@dataclass
class ExperimentHparams:
    project_name: str
    knowledge_editing_method: str
    knowledge_editing_hparams_config: str
    task_name: str
    wandb_entity: Optional[str] = None
    wandb_mode: Optional[str] = None
    train_dataset: Optional[str] = None
    train_size: Optional[int] = None
    eval_dataset: Optional[str] = None
    eval_size: Optional[int] = None
    layers: Optional[list[int]] = None
    edit_lr: Optional[float] = None
    options: Optional[list[str]] = None


def parse_user_config(hparams: ExperimentHparams) -> dict:
    options = hparams.options
    if options is None:
        return {}
    user_config = {}
    for option in options:
        key, value = option.split("=")
        user_config[key] = value
    if hparams.edit_lr is not None:
        user_config["edit_lr"] = hparams.edit_lr
    if hparams.layers is not None:
        user_config["layers"] = hparams.layers
    return user_config

    
def init_experiment(experiment_hparams: ExperimentHparams):
    method_config_class = Registry.get_method(experiment_hparams.knowledge_editing_method)
    dataset_class = Registry.get_dataset(experiment_hparams.task_name)
    # Make options into a dictionary
    user_config = parse_user_config(experiment_hparams)
    method_config = method_config_class.from_yaml(experiment_hparams.knowledge_editing_hparams_config, overrides=user_config)
    if experiment_hparams.train_dataset is None or method_config.eval_only:
        train_dataset = None
    else:
        train_dataset = dataset_class(
            experiment_hparams.train_dataset, split="train", size=experiment_hparams.train_size, config=method_config
        )
    if experiment_hparams.eval_dataset is None:
        eval_dataset = None
        test_dataset = None
    else:
        eval_dataset = dataset_class(
            experiment_hparams.eval_dataset, split="valid", size=experiment_hparams.eval_size, config=method_config
        )
        test_dataset = dataset_class(
            experiment_hparams.eval_dataset, split="test", size=experiment_hparams.eval_size, config=method_config
        )
    tags = []
    if experiment_hparams.eval_size is not None:
        tags.append(f"eval_size:{experiment_hparams.eval_size}")
    if hasattr(method_config, "training_locality"):
        tags.append(method_config.training_locality)
    wandb.init(
        project=experiment_hparams.project_name,
        name=method_config.run_name,
        config=method_config,
        entity=experiment_hparams.wandb_entity,
        mode=experiment_hparams.wandb_mode,
        tags=tags,
    )

    if method_config.requires_training and not method_config.eval_only:
        run_multimodal_ke_training(config=method_config, train_ds=train_dataset, eval_ds=eval_dataset, test_ds=test_dataset)
    else:
        run_multimodal_ke_evaluation(config=method_config, train_ds=train_dataset, eval_ds=eval_dataset, test_ds=test_dataset)


if __name__ == "__main__":
    exp = parse_args()
    trainer = init_experiment(exp)
