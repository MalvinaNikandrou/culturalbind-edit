import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

LOG = logging.getLogger(__name__)


@dataclass
class MultimodalOutput:
    accuracy: float
    token_ids: torch.Tensor | np.ndarray
    masked_logits: torch.Tensor | np.ndarray


class GenerationPipeline:
    generation_kwargs = {
        "max_new_tokens": 20,
        "do_sample": False,
        "top_k": None,
        "temperature": None,
        "top_p": 1.0,
    }

    def generate(self, model, tokenizer, batch, labels, pad_token_id: int | None = None) -> tuple[list[str], list[str]]:
        """Generate the output text for a sample.

        Args:
            model: The model to generate the output.
            tokenizer: The tokenizer to decode the tokens.
            batch: The batch of inputs.
            labels: The labels of the batch.
            pad_token_id: The id of the padding token.

        Returns:
            The input text and the predicted text.
        """
        if pad_token_id is None:
            pad_token_id = self._get_pad_token_id(tokenizer)
        sample_batch = self.prepare_inputs_for_generation(batch, labels, pad_token_id)
        with torch.inference_mode():
            # Generate the output tokens
            output_tokens = model.generate(
                **sample_batch,
                **self.generation_kwargs,
                pad_token_id=pad_token_id,
            )
        inputs = tokenizer.batch_decode(sample_batch["input_ids"], skip_special_tokens=True)
        output_tokens = output_tokens[:, sample_batch["input_ids"].shape[1] :]

        predicted = tokenizer.batch_decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return inputs, predicted

    def prepare_inputs_for_generation(self, batch, labels, pad_token_id) -> dict:
        """Prepare the inputs for generation by removing the labels and unsqueezing the batch dimension."""
        sample_batch = {key: value for key, value in batch.items() if "labels" not in key}
        # shift right according to the number of labels
        num_label_tokens = (labels != -100).sum(dim=1)
        new_input_ids = torch.full_like(sample_batch["input_ids"], pad_token_id)
        new_attention_mask = torch.zeros_like(sample_batch["attention_mask"])
        for index in range(sample_batch["input_ids"].shape[0]):
            num_label_tokens = (labels[index] != -100).sum()
            input_length = sample_batch["input_ids"].shape[1] - num_label_tokens

            new_input_ids[index, -input_length:] = sample_batch["input_ids"][index, :input_length]
            new_attention_mask[index, -input_length:] = sample_batch["attention_mask"][index, :input_length]

        sample_batch["input_ids"] = new_input_ids
        sample_batch["attention_mask"] = new_attention_mask
        return sample_batch

    def _get_pad_token_id(self, tokenizer: PreTrainedTokenizer) -> int:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        return pad_token_id


class Metric:
    """Abstract class for metrics."""

    def __init__(self, metric: str, output_dir: str, batch_key: str | None = None):
        self.metric = metric
        self.batch_key = batch_key if batch_key else metric
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = output_dir.joinpath(f"{self.metric}_samples_scores.json")
        if self.output_file.exists():
            LOG.error(f"Output file {self.output_file} already exists.")
            # raise FileExistsError

        samples = []
        with open(self.output_file, "w") as f:
            json.dump(samples, f)

    def update_output_file(self, sample: dict, log: bool = False):
        if log:
            samples = json.load(open(self.output_file, "r"))
            with open(self.output_file, "w") as f:
                samples.append(sample)
                json.dump(samples, f)

    def get_label_text(self, tokenizer: PreTrainedTokenizer, labels: torch.Tensor, sample_index: int) -> str:
        """Use the pre-edit label for the locality metric."""
        label_tokens = labels[sample_index]
        return tokenizer.decode(label_tokens[label_tokens != -100], skip_special_tokens=True)

    def _compute_metric(self, label: str, predicted: str) -> bool:
        label = self._normalize_text(label)
        predicted = self._normalize_text(predicted)
        return label == predicted

    def _normalize_text(self, text: str) -> str:
        # Lowercase the text
        text = text.lower()
        # Remove punctuation
        text = re.sub(r"\s+", " ", text)
        # Remove extra spaces
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    def __call__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch: dict, log: bool) -> float:
        correct = torch.tensor(0.0)
        labels = batch[self.batch_key]["labels"]
        num_samples = labels.shape[0]
        inputs, predicted = self._get_model_output(model, tokenizer, batch[self.batch_key], labels)
        for index in range(num_samples):
            label = self.get_label_text(tokenizer, labels, index)
            sample = {
                "input": inputs[index],
                "label": label,
                "prediction": predicted[index],
                "correct": self._compute_metric(label, predicted[index]),
                "sample_id": batch["sample_ids"][index],
            }
            self.update_output_file(sample, log)
            if sample["correct"]:
                correct += 1
        return correct / num_samples


class GenerationMetric(Metric, GenerationPipeline):
    def _get_model_output(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch: dict, labels
    ) -> tuple[list[str], list[str]]:
        return self.generate(model, tokenizer, batch, labels)


class EntityOverlapMetric(GenerationMetric):
    def _compute_metric(self, entity: str, text: str) -> bool:
        entity = self._normalize_text(entity)
        text = self._normalize_text(text)
        # Escape the entity to handle any special regex characters
        escaped_entity = re.escape(entity)
        # Use word boundaries to ensure the entity is not a subword
        pattern = rf"\b{escaped_entity}\b"
        return re.search(pattern, text) is not None


class EditSuccessMetric(GenerationMetric):
    def log(self, tokenizer: PreTrainedTokenizer, accuracy: float, batch: dict, log: bool) -> None:
        """
        Log the success metric.
        """
        if log:
            input = tokenizer.decode(batch[self.batch_key]["input_ids"][0], skip_special_tokens=False).strip()
            label_ids = batch[self.batch_key]["labels"][0]
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            label = tokenizer.decode(label_ids, skip_special_tokens=True).strip().lower()
            sample = {
                "input": input,
                "label": label,
                "prediction": batch[self.batch_key]["prediction"][0],
                "correct": accuracy,
                "sample_id": batch["sample_ids"][0],
            }
            self.update_output_file(sample, log)


class LocalityMetric(EditSuccessMetric):
    def __call__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch: dict, log: bool) -> float:
        """
        Compute the entity overlap metric.

        Args:
            tokenizer: The tokenizer used to convert the logits to tokens.
            logits: The logits of the model.
            label_tokens: The tokens of the label.

        Returns:
            The entity overlap metric.
        """
        correct = torch.tensor(0.0)
        labels = batch[self.batch_key]["labels"]
        num_samples = labels.shape[0]
        pre_edit_labels = batch[self.batch_key].pop("pre_edit_labels")
        if not torch.equal(labels, pre_edit_labels):
            raise ValueError("Labels and pre-edit labels must be the same for locality metric.")

        inputs, predicted = self.generate(model, tokenizer, batch[self.batch_key], labels)
        for index in range(num_samples):
            label = self.get_label_text(tokenizer, pre_edit_labels, index)
            sample = {
                "input": inputs[index],
                "label": label,
                "prediction": predicted[index],
                "correct": self._compute_metric(label, predicted[index]),
                "sample_id": batch["sample_ids"][index],
            }
            self.update_output_file(sample, log)
            if sample["correct"]:
                correct += 1
        return correct / num_samples


class TFMetric(Metric):
    def _get_model_output(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch: dict, labels
    ) -> tuple[list[str], list[str]]:
        return self.get_masked_logits(model, tokenizer, batch, labels)

    def get_masked_logits(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch: dict, labels
    ) -> tuple[list[str], torch.Tensor]:
        labels = labels.to(model.device)
        batch = batch.to(model.device)
        if hasattr(model, "requires_labels_for_inference") and model.requires_labels_for_inference:
            batch["labels"] = labels
        else:
            batch.pop("labels", None)
        with torch.inference_mode():
            outputs = model(**batch)
        # Get logits
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.logits

        inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        # Get the token ids and logits for the non-masked positions
        if logits.dim() == 3:
            logits = logits[:, :-1, :]
            labels = labels[:, 1:]

        mask = labels != -100
        token_ids = logits.argmax(-1).masked_fill(~mask, 0)
        token_ids = token_ids.masked_select(token_ids != 0).view(1, -1)
        masked_logits = logits.masked_select(mask.unsqueeze(-1)).view(mask.shape[0], -1, logits.shape[-1])
        predicted = tokenizer.batch_decode(
            masked_logits.argmax(dim=-1), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return inputs, predicted


def compute_locality_accuracy(
    pre_edit_logits: torch.Tensor,
    post_edit_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    topk: int = 1,
    tokenizer: PreTrainedTokenizer | None = None,
) -> float:
    pre_edit_logits = pre_edit_logits.to(torch.float32)
    post_edit_logits = post_edit_logits.to(torch.float32)
    # Get topk logit predictions post edit
    post_edit_logits_softmax_top_k = torch.topk(
        torch.nn.functional.softmax(post_edit_logits, dim=-1), k=topk, dim=-1
    ).indices
    post_edit_logits_softmax_top_k = post_edit_logits_softmax_top_k.view(-1, post_edit_logits.shape[1] * topk)
    # Get topk logit predictions pre edit
    pre_edit_logits_softmax_top_k = torch.topk(
        torch.nn.functional.softmax(pre_edit_logits, dim=-1), k=topk, dim=-1
    ).indices
    pre_edit_logits_softmax_top_k = pre_edit_logits_softmax_top_k.view(-1, pre_edit_logits.shape[1] * topk)

    # We want to ignore tokens where the label is -100
    if labels is not None:
        post_edit_logits_softmax_top_k[labels == -100] = 0
        pre_edit_logits_softmax_top_k[labels == -100] = 0

    # Compute locality accuracy
    locality_accuracy = (post_edit_logits_softmax_top_k == pre_edit_logits_softmax_top_k).all(-1).float().mean()
    return locality_accuracy.item()
