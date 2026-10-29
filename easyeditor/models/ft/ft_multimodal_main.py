import logging
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...trainer.losses import multiclass_acc_and_log_probs
from ...util import multimodal_tokenize, nethook, reset_model_weights
from .ft_multimodal_hparams import FTMultimodalHyperParams

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

LOG = logging.getLogger(__name__)


def apply_ft_to_multimodal_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTMultimodalHyperParams,
    copy=False,
    return_orig_weights=False,
    original_weights: dict[str, torch.Tensor] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    """
    if copy:
        model = deepcopy(model)

    device = torch.device(hparams.device)
    # Build inputs for positive samples
    batch = multimodal_tokenize(requests, processor=tok, device=device, target_key="target")
    # Execute the FT algorithm
    deltas = execute_multimodal_ft(model, batch, hparams)

    # Apply the computed deltas to the model
    if original_weights is None:
        original_weights = {}
    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in original_weights:
                original_weights[w_name] = w.detach().clone()

            w[...] += upd_matrix

    # Return the edited model and (optionally) the original weights
    return model, original_weights


def execute_multimodal_ft(
    model: AutoModelForCausalLM,
    batch: Dict[str, torch.Tensor],
    hparams: FTMultimodalHyperParams,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    Executes the FT algorithm for the specified updates.
    """
    model.train()
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }

    # Save old weights for future restoration
    original_weights = {k: v.clone().detach() for k, v in weights.items()}
    logging.info(f"Weights to be updated: {list(weights.keys())}")

    # Configure optimizer / gradients
    for name, w in model.named_parameters():
        w.requires_grad = name in weights
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in model_parameters])

    logging.info(f"Model has {params} trainable parameters")
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.edit_lr,
        weight_decay=hparams.weight_decay,
    )

    labels = batch.pop("labels")
    for edit_step in range(hparams.num_steps):
        opt.zero_grad()
        # Compute outputs with current model
        outputs_pos = model(**batch)
        # 1. LM loss on positive samples
        metric_dict = multiclass_acc_and_log_probs(
            pred=outputs_pos.logits,
            targ=labels,
            shift=True,
        )
        loss, acc = metric_dict["nll"], metric_dict["acc"]
        if acc == 1.0:
            break
        print(f"Step {edit_step}: Total Loss={loss.item():.4f}, Acc={acc.item():.4f}")
        if loss.item() >= 1e-2:
            # Update weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in weights.items()],
                hparams.grad_clip,
                error_if_nonfinite=True,
            )
            opt.step()
        if type(hparams.norm_constraint) is float:
            eps = hparams.norm_constraint
            with torch.no_grad():
                for k, v in weights.items():
                    v[...] = torch.clamp(v, min=original_weights[k] - eps, max=original_weights[k] + eps)

    if edit_step == hparams.num_steps - 1 and acc != 1.0:
        LOG.warning(f"Edit may not have converged within {hparams.num_steps} steps. Final acc: {acc.item():.2f}")
    else:
        LOG.info(f"Edit converged in {edit_step} steps with final acc: {acc.item():.2f}")

    deltas = {k: (weights[k] - original_weights[k]).detach() for k in weights}
    # Restore state of original model
    reset_model_weights(model, original_weights)
    model.eval()  # Switch to evaluation mode

    return deltas
