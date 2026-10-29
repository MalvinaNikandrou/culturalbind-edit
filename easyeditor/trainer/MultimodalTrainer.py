import logging
import time
from typing import Any, Optional, Tuple, Type, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.utils import ModelOutput

import wandb
from easyeditor.common.hparams import HyperParams
from easyeditor.evaluate.metrics import (
    EditSuccessMetric,
    EntityOverlapMetric,
    LocalityMetric,
    compute_locality_accuracy,
)
from easyeditor.evaluate.post_edit_crope_metrics import PostEvalBackgroundMetric, PostEvalRecognitionMetric
from easyeditor.trainer.algs.editable_model import EditableModel
from easyeditor.trainer.BaseTrainer import *
from easyeditor.trainer.losses import kl_loc_loss
from easyeditor.trainer.utils import RunningStatAverager, safe_backward

LOG = logging.getLogger(__name__)

ConfigType = Type[HyperParams]
EditModelType = Type[EditableModel]
EditModelOutput = Union[ModelOutput, torch.Tensor]
# loss_total, info_dict
EditStepOutput = Tuple[torch.Tensor, dict[str, float]]


class MultimodalTrainer(BaseTrainer):
    def __init__(self, config: ConfigType, train_set: Dataset, val_set: Dataset, test_set: Dataset = None) -> None:
        super().__init__(config, train_set, val_set, test_set)

        self.model.set_processor(self._get_processor())

        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]

        self.metrics = {
            "reliability": EditSuccessMetric(metric="edit_reliability", output_dir=config.results_dir),
            "edit_generality": EditSuccessMetric(metric="edit_generality", output_dir=config.results_dir),
            "edit_generality_image": EditSuccessMetric(metric="edit_generality_image", output_dir=config.results_dir),
            "loc": LocalityMetric(metric="loc", output_dir=config.results_dir),
            "loc_image": LocalityMetric(metric="loc_image", output_dir=config.results_dir),
            "text_portability": EntityOverlapMetric(metric="portability", output_dir=config.results_dir),
            "image_portability": EntityOverlapMetric(metric="portability_image", output_dir=config.results_dir),
            "text_portability_consistency": LocalityMetric(
                metric="text_portability_consistency", output_dir=config.results_dir, batch_key="portability"
            ),
            "image_portability_consistency": LocalityMetric(
                metric="image_portability_consistency", output_dir=config.results_dir, batch_key="portability_image"
            ),
            "background": EntityOverlapMetric(metric="background", output_dir=config.results_dir),
        }

        self.post_eval_metrics = {
            "reliability": PostEvalRecognitionMetric(
                metric="edit_reliability",
                predictions_file=self.metrics["reliability"].output_file,
                output_dir=config.results_dir,
            ),
            "text_portability": PostEvalBackgroundMetric(
                metric="text_portability",
                predictions_file=self.metrics["text_portability"].output_file,
                output_dir=config.results_dir,
            ),
            "image_portability": PostEvalBackgroundMetric(
                metric="image_portability",
                predictions_file=self.metrics["image_portability"].output_file,
                output_dir=config.results_dir,
            ),
            "background": PostEvalBackgroundMetric(
                metric="background",
                predictions_file=self.metrics["background"].output_file,
                output_dir=config.results_dir,
            ),
        }

    def edit_step(self, batch: dict, log: bool = False, training: bool = True) -> EditStepOutput:
        info_dict = {}
        split = "train" if training else "eval"
        self.model.train(training)
        # 1. Get base model outputs for locality metrics
        pre_edit_loc_logits = self.get_pre_edit_logits(batch, key="loc")
        pre_edit_image_loc_logits = self.get_pre_edit_logits(batch, key="loc_image")

        # print("before", training, self.model.model.language_model.layers[self.config.layers[0]].mlp.down_proj.weight[5])
        # 2. Do the edit
        start = time.time()
        # Prepare the batch for the edit step
        edit_batch = self.model.prepare_edit_batch(batch, key="edit_reliability")
        edited_model, model_info = self.model.edit(edit_batch, batch["cond"])
        edit_time = time.time() - start

        # if self.model.weights_copy is None:
        #     print(f"\nafter editing = {self.model.model.language_model.layers[35].mlp.down_proj.weight}")

        # else:
        #     close = torch.allclose(self.model.weights_copy[f"model.language_model.layers.35.mlp.down_proj.weight"], self.model.model.language_model.layers[35].mlp.down_proj.weight)
        #     print(f"\nafter editing the weights are the same = {close}")

        # 3. Get the post-edit outputs and metrics
        with torch.set_grad_enabled(training):
            # Editing text generality loss
            loss_edit = self.compute_edit_loss(
                edited_model, batch, key="edit_generality", split=split, info_dict=info_dict
            )

            # Editing image generality loss
            loss_image_edit = self.compute_edit_loss(
                edited_model, batch, key="edit_generality_image", split=split, info_dict=info_dict
            )

            # Text Locality
            loss_locality, _ = self.compute_locality_loss_and_accuracy(
                edited_model,
                batch,
                base_logits=pre_edit_loc_logits,
                key="loc",
                split=split,
                info_dict=info_dict,
                topk=1,
                log=log,
            )
            # Image Locality
            loss_image_locality, _ = self.compute_locality_loss_and_accuracy(
                edited_model,
                batch,
                base_logits=pre_edit_image_loc_logits,
                key="loc_image",
                split=split,
                info_dict=info_dict,
                topk=1,
                log=log,
            )

            # Editing reliability metrics
            self.compute_reliability_metrics(edited_model, batch, split=split, log=log, info_dict=info_dict)

            # Generality metrics
            self.compute_generality_metrics(
                edited_model, batch, split=split, log=log, info_dict=info_dict, metric_key="edit_generality"
            )
            self.compute_generality_metrics(
                edited_model, batch, split=split, log=log, info_dict=info_dict, metric_key="edit_generality_image"
            )

            # Portability metrics
            self.compute_portability_metrics(edited_model, batch, split=split, log=log, info_dict=info_dict)

        loss_total_edit = self.combine_edit_losses(
            loss_edit=loss_edit,
            loss_locality=loss_locality,
            loss_edit_image=loss_image_edit,
            loss_image_locality=loss_image_locality,
            batch=batch,
            split=split,
            info_dict=info_dict,
        )

        if training:
            safe_backward(loss_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)
        self.model.restore_weights(edited_model.model)

        # if self.model.weights_copy is None:
        #     print(f"\nafter restoring = {self.model.model.language_model.layers[35].mlp.down_proj.weight}")
        # else:
        #     close = torch.allclose(self.model.weights_copy[f"model.language_model.layers.35.mlp.down_proj.weight"], self.model.model.language_model.layers[35].mlp.down_proj.weight)
        #     print(f"after reset the weights are the same = {close}")

        loss_total = self.compute_total_loss(loss_total_edit, info_dict)
        # Log other metrics
        info_dict[f"time/{split}_edit"] = edit_time
        info_dict[f"memory/{split}_alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict[f"memory/{split}_res_max"] = torch.cuda.max_memory_reserved()
        info_dict.update(model_info)
        wandb.log(info_dict)
        return loss_total, info_dict

    def train_step(self, batch: dict) -> dict[str, float]:
        training = self.config.requires_training
        _, info_dict = self.edit_step(batch, training=training)

        if training and self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict["grad"] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f"lr/lr{lr_idx}"] = lr.item()

        return info_dict

    def validate(self, steps: Optional[int] = None, log: bool = False) -> dict[str, float]:
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager()
        tokenizer = self._get_tokenizer()
        start_time = time.time()
        for val_step, batch in tqdm(enumerate(self.val_loader), desc="Evaluating", total=steps):
            if val_step >= steps:
                break
            cond = tokenizer.batch_decode(batch["cond"], skip_special_tokens=True)
            batch["sample_ids"] = [cc.split("---")[0].strip() for cc in cond]
            _, info_dict = self.edit_step(batch, log=log, training=False)
            averager.add(info_dict)

            if log and (val_step + 1) % self.config.log_interval == 0:
                self._inline_validation_log(val_step, averager.average(), start_time, steps)

        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps
        self._inline_validation_log(val_step, stats, start_time, steps)

        return stats

    def final_eval(self, val_info: dict[str, float]) -> dict[str, float]:
        if not self.config.final_eval:
            return val_info
        # Load the best model
        if not self.config.eval_only:
            if (not self.config.debug) or self.config.save:
                if self.config.model_parallel:
                    archive = torch.load(self.save_path, weights_only=False)
                    LOG.info(f"Loading best model from step {archive['step']}, elapsed time {archive['elapsed_time']}")
                    self.model.load_state_dict(archive["model"])
                else:
                    archive = torch.load(self.save_path, map_location="cpu", weights_only=False)
                    LOG.info(f"Loading best model from step {archive['step']}, elapsed time {archive['elapsed_time']}")
                    self.model.to("cpu")
                    self.model.load_state_dict(archive["model"])
                    self.model.to(self.config.device)
        val_steps = self.config.val_steps if self.config.debug else None
        if self.config.alg_name == "MALMEN":
            val_info = self.model.valid(
                log=True, steps=val_steps, config=self.config, loader=self.test_loader, val_set=self.test_set
            )
        else:
            self.val_loader = self.test_loader
            self.val_set = self.test_set
            val_info = self.validate(log=True, steps=val_steps)
        self.echo(self.global_iter, val_info, pretty=True)
        del self.model
        # clear cuda cache
        torch.cuda.empty_cache()
        # print gpu memory
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
        if self.config.compute_post_eval_metrics:
            for metric_name, metric in self.post_eval_metrics.items():
                value = metric.compute()
                wandb.log({f"{metric_name}/post_eval_acc": value})

        return val_info

    @torch.no_grad()
    def get_pre_edit_logits(self, batch, key: str) -> torch.Tensor:
        if "pre_edit_labels" in batch[key]:
            pre_edit_labels = batch[key].pop("pre_edit_labels")
        else:
            pre_edit_labels = None

        pre_edit_logits = self._get_logits_from_model_outputs(self.model(batch[key]))

        if pre_edit_labels is not None:
            batch[key]["pre_edit_labels"] = pre_edit_labels

        return pre_edit_logits

    def compute_total_loss(self, loss_total_edit: torch.Tensor, info_dict) -> torch.Tensor:
        # TODO: BASE LOSS IS NOT IMPLEMENTED
        # loss_base = torch.tensor(0.0)
        # loss_total = loss_total_edit + self.config.cbase * loss_base
        info_dict["loss/total"] = loss_total_edit.item()
        return loss_total_edit

    def combine_edit_losses(
        self,
        loss_edit: torch.Tensor,
        loss_locality: torch.Tensor,
        loss_edit_image: torch.Tensor,
        loss_image_locality: torch.Tensor,
        batch: dict,
        split: str,
        info_dict: dict[str, float],
    ) -> torch.Tensor:
        """Compute the total loss for the edit step."""
        self._is_loss_nan("edit_generality", loss_edit, batch)
        self._is_loss_nan("edit_image_generality", loss_edit_image, batch)
        self._is_loss_nan("loc", loss_locality, batch)
        self._is_loss_nan("loc_image", loss_image_locality, batch)

        if self.config.alg_name == "SERAC_MULTI":
            loss_total_edit = (
                self.config.cedit * loss_edit
                + self.config.iedit * loss_edit_image
                + self.config.cloc * (loss_locality + loss_image_locality)  # loss_locality
            )
        else:
            loss_total_edit = (
                self.config.cedit * loss_edit
                + self.config.iedit * loss_edit_image
                + self.config.cloc * (loss_locality + loss_image_locality)
            )
        info_dict[f"loss/{split}_total_edit"] = loss_total_edit.item()
        return loss_total_edit

    def compute_edit_loss(
        self, edited_model: EditModelType, batch: dict, key: str, split: str, info_dict: dict[str, float]
    ) -> torch.Tensor:
        """Compute the edit loss (usually cross-entropy) for the given key."""
        if key in {"portability", "portability_image", "background"} and isinstance(batch[key], list):
            num_samples = len(batch[key])
            post_edit_dict = {
                metric: torch.tensor(0.0, device=self.config.device) for metric in ["nll", "acc", "log_prob", "prob"]
            }
            for idx in range(num_samples):
                post_edit_outputs_i = self._get_post_edit_dict(
                    edited_model, dict_to(batch[key][idx], self.config.device)
                )
                for metric in post_edit_dict:
                    post_edit_dict[metric] += post_edit_outputs_i[metric] / num_samples
        else:
            post_edit_dict = self._get_post_edit_dict(edited_model, batch[key])
        info_dict[f"loss/{split}_{key}"] = post_edit_dict["nll"].item()
        info_dict[f"{key}/{split}_acc"] = post_edit_dict["acc"].item()
        info_dict[f"{key}/{split}_log_prob"] = post_edit_dict["log_prob"].item()
        info_dict[f"{key}/{split}_prob"] = post_edit_dict["prob"].item()
        return post_edit_dict["nll"]

    def _get_post_edit_dict(self, edited_model: EditModelType, key_batch):
        post_edit_outputs = edited_model(key_batch)
        post_edit_logits = self._get_logits_from_model_outputs(post_edit_outputs)
        # Labels
        if isinstance(post_edit_outputs, torch.Tensor):
            post_batch_labels = key_batch["labels"]
        else:
            post_batch_labels = post_edit_outputs.get("labels", key_batch["labels"])
        # Edit loss
        return self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels, multimodal=True)

    def compute_locality_loss_and_accuracy(
        self,
        edited_model: EditModelType,
        batch: dict,
        base_logits,
        key: str,
        split: str,
        info_dict: dict[str, float],
        topk: int,
        log: bool,
    ) -> Tuple[torch.Tensor, float]:
        """Compute the locality loss (KL divergence) & accuracy for the given key."""
        if "pre_edit_labels" in batch[key]:
            pre_edit_labels = batch[key].pop("pre_edit_labels")
        else:
            pre_edit_labels = None

        post_base_outputs = edited_model(batch[key])
        post_base_logits = self._get_logits_from_model_outputs(post_base_outputs)
        # Get the KL mask
        kl_mask = self._get_kl_mask(post_base_outputs, batch[key])
        # Get the KL divergence loss
        loss_locality = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)
        info_dict[f"loss/{split}_{key}"] = loss_locality.item()
        # Get the accuracy
        if pre_edit_labels is not None:
            batch[key]["pre_edit_labels"] = pre_edit_labels
        info_dict[f"{key}/{split}_acc"] = self.compute_locality_accuracy(
            edited_model, base_logits, post_base_logits, batch, key, split, topk, log
        )
        return loss_locality, info_dict[f"{key}/{split}_acc"]

    @torch.no_grad()
    def compute_reliability_metrics(
        self, edited_model: EditModelType, batch: dict, split: str, log: str, info_dict: dict[str, float]
    ) -> float:
        """Compute the reliability accuracy for the given key."""
        if split == "train" or not log:
            inner_edit_outputs = edited_model(batch["edit_reliability"])
            inner_edit_logits = self._get_logits_from_model_outputs(inner_edit_outputs)
            if isinstance(inner_edit_outputs, torch.Tensor):
                inner_batch_labels = batch["edit_reliability"]["labels"]
            else:
                inner_batch_labels = inner_edit_outputs.get("labels", batch["edit_reliability"]["labels"])
            inner_edit_dict = self.model.edit_loss_fn(
                self.config, inner_edit_logits, inner_batch_labels, multimodal=True
            )
            accuracy = inner_edit_dict["acc"].item()
        else:
            # Log the reliability
            accuracy = self.metrics["reliability"](
                model=edited_model, batch=batch, tokenizer=self._get_tokenizer(), log=log
            )

        info_dict[f"reliability/{split}_acc"] = accuracy
        return accuracy

    @torch.no_grad()
    def compute_generality_metrics(
        self,
        edited_model: EditModelType,
        batch: dict,
        split: str,
        log: str,
        info_dict: dict[str, float],
        metric_key: str,
    ) -> float:
        """Compute the generality accuracy for the given key."""
        if split == "train" or not log:
            return info_dict[f"{metric_key}/{split}_acc"]
        # Log the generality
        accuracy = self.metrics[metric_key](model=edited_model, batch=batch, tokenizer=self._get_tokenizer(), log=log)

        info_dict[f"{metric_key}/{split}_acc"] = accuracy
        return accuracy

    @torch.no_grad()
    def compute_locality_accuracy(
        self,
        edited_model: EditModelType,
        pre_edit_loc_logits: torch.Tensor,
        post_edit_loc_logits: torch.Tensor,
        batch: dict,
        key: str,
        split: str,
        topk: int,
        log: bool = False,
    ) -> float:
        if split == "train" or not log:
            # Text locality accuracy - top 1
            locality_accuracy = compute_locality_accuracy(
                pre_edit_loc_logits, post_edit_loc_logits, labels=batch[key]["labels"], topk=topk
            )
        else:
            locality_accuracy = self.metrics[key](
                model=edited_model, batch=batch, tokenizer=self._get_tokenizer(), log=log
            )
        return locality_accuracy

    @torch.no_grad()
    def compute_portability_metrics(
        self, edited_model: EditModelType, batch: dict, split: str, log: str, info_dict: dict[str, float]
    ) -> None:
        # Post-edit portability loss
        self.compute_edit_loss(edited_model, batch, key="portability", split=split, info_dict=info_dict)
        self.compute_edit_loss(edited_model, batch, key="portability_image", split=split, info_dict=info_dict)
        background_edit_acc = None
        if split != "train":
            tokenizer = self._get_tokenizer()
            for metric_key, batch_key in [
                ("text_portability", "portability"),
                ("image_portability", "portability_image"),
            ]:
                if isinstance(batch[batch_key], list):
                    num_samples = len(batch[batch_key])
                    info_dict[f"{metric_key}/{split}_acc"] = 0
                    info_dict[f"{metric_key}_consistency/{split}_acc"] = 0
                    for idx in range(num_samples):
                        batch_i = {
                            batch_key: dict_to(batch[batch_key][idx], self.config.device),
                            "sample_ids": [
                                batch["sample_ids"][idx] for _ in range(batch[batch_key][idx]["input_ids"].shape[0])
                            ],
                        }
                        acc = self.metrics[metric_key](
                            model=edited_model, batch=batch_i, tokenizer=tokenizer, log=log
                        ).item()
                        info_dict[f"{metric_key}/{split}_acc"] += acc / num_samples
                        acc = self.metrics[f"{metric_key}_consistency"](
                            model=edited_model, batch=batch_i, tokenizer=tokenizer, log=log
                        ).item()
                        info_dict[f"{metric_key}_consistency/{split}_acc"] += acc / num_samples
                else:
                    info_dict[f"{metric_key}/{split}_acc"] = self.metrics[metric_key](
                        model=edited_model, batch=batch, tokenizer=tokenizer, log=log
                    ).item()
                    info_dict[f"{metric_key}_consistency/{split}_acc"] = self.metrics[f"{metric_key}_consistency"](
                        model=edited_model, batch=batch, tokenizer=tokenizer, log=log
                    ).item()

            # Post-edit background metric
            if log:
                if isinstance(batch["background"], list):
                    num_samples = len(batch["background"])
                    background_edit_acc = 0
                    for idx in range(num_samples):
                        batch_i = {
                            "background": dict_to(batch["background"][idx], self.config.device),
                            "sample_ids": [
                                batch["sample_ids"][idx] for _ in range(batch["background"][idx]["input_ids"].shape[0])
                            ],
                        }
                        acc = self.metrics["background"](
                            model=edited_model, batch=batch_i, tokenizer=tokenizer, log=log
                        ).item()
                        background_edit_acc += acc / num_samples
                else:
                    background_edit_acc = self.metrics["background"](
                        model=edited_model, batch=batch, tokenizer=tokenizer, log=log
                    ).item()
        if background_edit_acc is not None:
            info_dict[f"background/{split}_acc"] = background_edit_acc

    def _get_kl_mask(self, model_output: Union[ModelOutput, torch.Tensor], batch: dict) -> torch.Tensor:
        if not isinstance(model_output, torch.Tensor) and model_output.get("attention_mask") is not None:
            return model_output.attention_mask
        if batch.get("attention_mask") is not None:
            return batch["attention_mask"]
        return torch.ones(batch["input_ids"].shape[0], batch["input_ids"].shape[1]).to(batch["input_ids"].device)

    def _inline_validation_log(self, step: int, stats: dict[str, float], start_time: float, steps: int) -> None:
        log_info = f"Step {step+1}/{steps}".ljust(20)
        elapsed = (time.time() - start_time) / (step + 1)
        log_info += f"Elapsed: {elapsed:<12.5f} "
        aggregate_stats = {}
        for metric in self.metrics:
            acc = stats.get(f"{metric}/eval_acc")
            if acc is not None:
                aggregate_stats[f"aggregate/eval_{metric}_acc"] = acc
                log_info += f"{metric}: {acc:<12.5f} "
        wandb.log(aggregate_stats)
        LOG.info(log_info)

    def _get_logits_from_model_outputs(self, outputs: Union[Type[ModelOutput], torch.Tensor]) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        return outputs.logits

    def _is_loss_nan(self, loss_name: str, loss: torch.Tensor, batch) -> bool:
        is_nan = loss.isnan()
        if is_nan:
            LOG.warning(f"{loss_name} is nan")
        return is_nan

    def _get_tokenizer(self) -> Any:
        if self.train_loader is not None:
            return self.train_loader.dataset.collater.processor.tokenizer
        return self.val_loader.dataset.collater.processor.tokenizer

    def _get_processor(self) -> Any:
        if self.train_loader is not None:
            return self.train_loader.dataset.collater.processor.processor
        return self.val_loader.dataset.collater.processor.processor
