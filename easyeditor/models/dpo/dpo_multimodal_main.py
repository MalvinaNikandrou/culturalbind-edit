import logging
from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...trainer.losses import multiclass_acc_and_log_probs
from ...util import get_parameter, multimodal_tokenize, reset_model_weights
from .dpo_hparams import DPOHyperParams
from .dpo_multimodal_hparams import DPOMultimodalHyperParams

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

LOG = logging.getLogger(__name__)


def apply_dpo_to_multimodal_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: list[dict],
    hparams: DPOMultimodalHyperParams,
    copy=False,
    return_orig_weights=False,
    original_weights: dict[str, torch.Tensor] = None,
) -> tuple[AutoModelForCausalLM, dict[str, Any]]:
    """
    Returns a model with the desired changes.
    """
    if copy:
        model = deepcopy(model)

    device = torch.device(hparams.device)
    # Build inputs for positive samples
    batch_positive = multimodal_tokenize(requests, processor=tok, device=device, target_key="target")
    # Build inputs for negative samples
    batch_negative = multimodal_tokenize(requests, processor=tok, device=device, target_key="target_old")
    # Execute the DPO algorithm
    dpo_trainer = DPOTrainer(hparams, original_weights)
    deltas = dpo_trainer(model, batch_positive, batch_negative)
    del dpo_trainer

    # Apply the computed deltas to the model
    if original_weights is None:
        original_weights = {}
    with torch.inference_mode():
        for w_name, upd_matrix in deltas.items():
            w = get_parameter(model, w_name)
            if return_orig_weights and w_name not in original_weights:
                original_weights[w_name] = w.detach().clone()

            w[...] += upd_matrix

    # Return the edited model and (optionally) the original weights
    return model, original_weights


class DPOTrainer:
    def __init__(self, hparams: DPOHyperParams, original_weights: dict[str, torch.Tensor] = None):
        self.hparams = hparams
        self.original_weights = original_weights

    def setup_optimization(
        self,
        model: AutoModelForCausalLM,
    ) -> tuple[dict[str, torch.Tensor], torch.optim.Optimizer]:
        """
        Sets up the optimization process for DPO.
        """
        # Retrieve weights that user desires to change
        weights = {
            n: p
            for n, p in model.named_parameters()
            for layer in self.hparams.layers
            if self.hparams.rewrite_module_tmp.format(layer) in n
        }
        print(f"Weights to be updated: {list(weights.keys())}")

        # Store reference weights
        if self.original_weights is None:
            self.original_weights = {k: v.detach().clone() for k, v in weights.items()}

        # Set requires_grad appropriately
        for name, w in model.named_parameters():
            w.requires_grad = name in weights

        # Configure optimizer / gradients
        optimizer = torch.optim.Adam(
            list(weights.values()),
            lr=self.hparams.edit_lr,
            weight_decay=self.hparams.weight_decay,
        )
        return weights, optimizer

    def should_early_stop(self, accuracy: float) -> bool:
        """Should we early stop based on accuracy?"""
        if not self.hparams.early_stop:
            return False
        if accuracy == 1.0:
            self.patience -= 1
        else:
            self.patience = self.hparams.patience
        return self.patience <= 0

    def get_batch_logps(self, logits, labels):
        """
        Compute the sequence log-probabilities for a batch.

        Args:
            logits: Tensor of shape (batch_size, seq_len, vocab_size) with model predictions.
            labels: Tensor of shape (batch_size, seq_len).
        """
        labels = labels[:, 1:].clone()
        labels_mask = labels != -100
        labels[labels == -100] = 0  # Prevent gather errors

        # (batch_size, seq_len, vocab_size)
        logits = logits[:, :-1, :]
        # (batch_size, seq_len)
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        # Apply mask (batch_size)
        return (per_token_logps * labels_mask).sum(-1)

    def compute_dpo_loss(
        self,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
        ref_positive_logits: torch.Tensor,
        ref_negative_logits: torch.Tensor,
    ):
        """Compute the DPO loss given the logits and tokens.

        *_labels: (batch_size, seq_len)
        *_logits: (batch_size, seq_len, vocab_size)
        """
        # Current model log probs
        log_probs_pos = self.get_batch_logps(positive_logits, positive_labels)
        log_probs_neg = self.get_batch_logps(negative_logits, negative_labels)
        # Reference model log probs
        ref_log_probs_pos = self.get_batch_logps(ref_positive_logits, positive_labels)
        ref_log_probs_neg = self.get_batch_logps(ref_negative_logits, negative_labels)
        # Advantage
        dpo_advantage = self.hparams.beta * ((log_probs_pos - ref_log_probs_pos) - (log_probs_neg - ref_log_probs_neg))
        dpo_loss = -torch.mean(F.logsigmoid(dpo_advantage))
        return dpo_loss

    def __call__(
        self,
        model: AutoModelForCausalLM,
        batch_positive: dict[str, torch.Tensor],
        batch_negative: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Executes the DPO algorithm for the specified updates.
        """
        weights, optimizer = self.setup_optimization(model)

        self.patience = self.hparams.patience
        max_steps = self.hparams.num_steps
        positive_labels = batch_positive.pop("labels")
        negative_labels = batch_negative.pop("labels")
        # Compute logits with reference model
        model.eval()
        with torch.inference_mode():
            # Get full param dict, then override with reference weights
            # ref_params = dict(model.named_parameters())
            # ref_params = {k: (self.original_weights[k] if k in self.original_weights else v)
            #             for k, v in ref_params.items()}

            # ref_outputs_pos = functional_call(model, ref_params, kwargs=batch_positive)
            # ref_outputs_neg = functional_call(model, ref_params, kwargs=batch_negative)
            ref_positive_logits = model(**batch_positive).logits
            ref_negative_logits = model(**batch_negative).logits
        model.train()

        for edit_step in range(max_steps):
            model.zero_grad()
            # Compute outputs with current model
            outputs_pos = model(**batch_positive)

            # 1. LM loss on positive samples
            metric_dict = multiclass_acc_and_log_probs(
                pred=outputs_pos.logits,
                targ=positive_labels,
                shift=True,
            )
            lm_loss, acc = metric_dict["nll"], metric_dict["acc"]

            if self.should_early_stop(acc):
                break

            outputs_neg = model(**batch_negative)

            # 2. DPO loss
            dpo_loss = self.compute_dpo_loss(
                positive_labels=positive_labels,
                negative_labels=negative_labels,
                positive_logits=outputs_pos.logits,
                negative_logits=outputs_neg.logits,
                ref_positive_logits=ref_positive_logits,
                ref_negative_logits=ref_negative_logits,
            )

            # 3. Total loss
            loss = self.hparams.alpha * lm_loss + (1 - self.hparams.alpha) * dpo_loss
            print(
                f"Step {edit_step}: Total Loss={loss.item():.4f}, LM Loss={lm_loss.item():.4f}, DPO Loss={dpo_loss.item():.4f}, Acc={acc.item():.4f}",
                end="\r",
            )

            # Update weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(weights.values()),
                self.hparams.grad_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()

        model.eval()
        if edit_step == self.hparams.num_steps - 1 and acc != 1.0:
            LOG.warning(
                f"Edit may not have converged within {self.hparams.num_steps} steps. Final acc: {acc.item():.2f}"
            )
        else:
            LOG.info(f"Edit converged in {edit_step + 1} steps with final acc: {acc.item():.2f}")

        batch_positive["labels"] = positive_labels
        batch_negative["labels"] = negative_labels
        deltas = {k: (weights[k] - self.original_weights[k]).detach() for k in weights}
        # Restore state of original model
        reset_model_weights(model, self.original_weights)
        print(f"Deltas successfully computed for {list(weights.keys())}")

        return deltas
