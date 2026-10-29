from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict
from transformers import AutoProcessor

from ..trainer.losses import multiclass_acc_and_log_probs
from ..util import HyperParams
from .metrics import compute_locality_accuracy


@dataclass
class MultimodalOutput:
    accuracy: float
    token_ids: torch.Tensor | np.ndarray
    masked_logits: torch.Tensor | np.ndarray


class EvaluationResults(BaseModel):
    reliability_accuracy: float
    text_generality_accuracy: float
    image_generality_accuracy: float
    text_portability_accuracy: float
    image_portability_accuracy: float


class PreEditEvaluationResults(EvaluationResults):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    text_locality_output: torch.Tensor | np.ndarray
    image_locality_output: torch.Tensor | np.ndarray


class PostEditEvaluationResults(EvaluationResults):
    text_locality_accuracy: float
    image_locality_accuracy: float


def validate_results(results: dict, pre_edit: bool) -> None:
    """Validate the results dictionary to ensure correct types."""
    if pre_edit:
        PreEditEvaluationResults(**results)
    else:
        PostEditEvaluationResults(**results)


def prepare_multimodal_hf_edit(
    processor: AutoProcessor,
    targets: str | list[str],
    prompts: str | list[str],
    images: list[Image.Image] | None,
    device: torch.device = torch.device("cuda"),
    separate_labels: bool = False,
):
    """Prepare inputs for multimodal hf edit evaluation."""
    if isinstance(targets, str):
        targets = [targets]
    if isinstance(prompts, str):
        prompts = [prompts]
    if images is None:
        images = [None] * len(prompts)
    elif not isinstance(images, list):
        images = [images]

    multimodal_inputs = processor.prepare_inputs(prompts, targets, images)
    processor.prepare_labels(multimodal_inputs, prompts, targets, images)
    multimodal_inputs.to(device)
    if separate_labels:
        labels = multimodal_inputs.pop("labels")
        ret = {
            "multimodal_inputs": multimodal_inputs,
            "labels": labels,
        }
    else:
        ret = multimodal_inputs
    return ret


def get_multimodal_hf_output(model, batch, exact_match=True) -> MultimodalOutput:
    """Compute the outputs of the multimodal model given the batch."""
    batch["multimodal_inputs"] = batch["multimodal_inputs"].to(model.device)
    targ = batch["labels"].to(model.device)
    model.eval()
    if hasattr(model, "requires_labels_for_inference") and model.requires_labels_for_inference:
        batch["multimodal_inputs"]["labels"] = targ
    with torch.inference_mode():
        outputs = model(**batch["multimodal_inputs"])
    # Get logits
    if isinstance(outputs, torch.Tensor):
        logits = outputs
    else:
        logits = outputs.logits
    # Compute loss and accuracy
    metrics = multiclass_acc_and_log_probs(pred=logits, targ=targ, exact_match=exact_match)

    # Get the token ids and logits for the non-masked positions
    if logits.dim() == 3:
        logits = logits[:, :-1, :]
        targ = targ[:, 1:]

    mask = targ != -100
    targ[~mask] = 0
    token_ids = logits.argmax(-1).masked_fill(~mask, 0)
    token_ids = token_ids.masked_select(token_ids != 0).view(1, -1)

    masked_logits = logits.masked_select(mask.unsqueeze(-1)).view(mask.shape[0], -1, logits.shape[-1])
    output = MultimodalOutput(
        accuracy=metrics["acc"].item(),
        token_ids=token_ids.cpu(),
        masked_logits=masked_logits.cpu(),
    )
    return output


def evaluate_with_icl_examples(
    model,
    hparams: HyperParams,
    icl_examples,
    target,
    prompt,
    image: Image.Image | list[Image.Image] | None,
    processor=None,
) -> MultimodalOutput:
    # Prepare input batch
    samples = prepare_multimodal_hf_edit(processor, target, ["".join(icl_examples) + prompt], image)
    # Get the output
    return get_multimodal_hf_output(model, samples, exact_match=hparams.exact_match)


def compute_icl_multimodal_edit_metrics(
    model,
    processor,
    metrics: dict,
    record: dict,
    icl_examples: list[str],
    add_new_fact: bool,
) -> dict:
    """
    Compute IKE metrics for a multimodal model.
    """
    model.eval()
    # First, unpack rewrite evaluation record.
    image = record["image"]
    prompt = record["prompt"]
    target = record["target"]
    sample_ids = [record["cond"]]

    new_fact = f"New Fact: {prompt} {target}\nPrompt: {prompt}"
    results = {}
    # Reliability evaluation
    if add_new_fact:
        metric_prompt = new_fact
    else:
        metric_prompt = prompt
    metric_prompt = ["".join(icl_examples) + metric_prompt]
    reliability_batch = prepare_multimodal_hf_edit(processor, target, metric_prompt, image)
    results["reliability_accuracy"] = metrics["reliability"](
        model=model,
        batch=dict(tf_edit_reliability=reliability_batch, sample_ids=sample_ids),
        tokenizer=processor.processor.tokenizer,
        log=True,
    ).item()
    # Generality evaluation
    rephrase = record["rephrase_prompt"] if "rephrase_prompt" in record.keys() else None
    if rephrase is not None:
        if add_new_fact:
            metric_prompt = f"New Fact: {prompt} {target}\nPrompt: {rephrase}"
        else:
            metric_prompt = rephrase
        metric_prompt = ["".join(icl_examples) + metric_prompt]
        rephrase_batch = prepare_multimodal_hf_edit(processor, targets=target, prompts=metric_prompt, images=image)
        results["text_generality_accuracy"] = metrics["edit_generality"](
            model=model,
            batch=dict(tf_edit_generality=rephrase_batch, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()
    # Image generality evaluation
    rephrase_image = record["image_generality"] if "image_generality" in record.keys() else None
    if "image_generality" in record.keys():
        if add_new_fact:
            metric_prompt = new_fact
        else:
            metric_prompt = prompt
        metric_prompt = ["".join(icl_examples) + metric_prompt]
        edit_image_outer = prepare_multimodal_hf_edit(
            processor, targets=target, prompts=metric_prompt, images=rephrase_image
        )
        results["image_generality_accuracy"] = metrics["edit_generality_image"](
            model=model,
            batch=dict(tf_edit_generality_image=edit_image_outer, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()
    # Locality evaluation
    if "locality_prompt" in record.keys():
        loc_q = record["locality_prompt"]
        if add_new_fact:
            loc_prompt = f"New Fact: {prompt} {target}\nPrompt: {loc_q}"
        else:
            loc_prompt = loc_q
        metric_prompt = ["".join(icl_examples) + loc_prompt]
        locality_ground_truth = record["locality_pre_edit_answer"]
        locality_batch = prepare_multimodal_hf_edit(
            processor, targets=locality_ground_truth, prompts=metric_prompt, images=None
        )
        results["text_locality_accuracy"] = metrics["loc"](
            model=model,
            batch=dict(tf_loc=locality_batch, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()
    # Multimodal locality evaluation
    if "multimodal_locality_image" in record.keys():
        m_loc_image = record["multimodal_locality_image"]
        m_loc_q = record["multimodal_locality_prompt"]
        if add_new_fact:
            loc_prompt = f"New Fact: {prompt} {target}\nPrompt: {m_loc_q}"
        else:
            loc_prompt = m_loc_q
        metric_prompt = ["".join(icl_examples) + loc_prompt]
        locality_ground_truth = record["multimodal_locality_pre_edit_answer"]
        locality_batch = prepare_multimodal_hf_edit(
            processor,
            targets=locality_ground_truth,
            prompts=metric_prompt,
            images=m_loc_image,
        )
        results["image_locality_accuracy"] = metrics["loc_image"](
            model=model,
            batch=dict(tf_loc_image=locality_batch, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()
    # Portability evaluation
    if "text_portability_prompt" in record.keys():
        if isinstance(record["text_portability_prompt"], str):
            record["text_portability_prompt"] = [record["text_portability_prompt"]]
        if isinstance(record["text_portability_ground_truth"], str):
            record["text_portability_ground_truth"] = [record["text_portability_ground_truth"]]
        portability_accs = []
        for port_prompt, port_target in zip(
            record["text_portability_prompt"], record["text_portability_ground_truth"], strict=True
        ):
            if add_new_fact:
                port_prompt = f"New Fact: {prompt} {target}\nPrompt: {port_prompt}"
            port_prompt = "".join(icl_examples) + port_prompt
            port_edit = prepare_multimodal_hf_edit(processor, port_target, port_prompt, images=None)
            portability_acc = metrics["text_portability"](
                model=model,
                batch=dict(text_portability=port_edit, sample_ids=sample_ids),
                tokenizer=processor.processor.tokenizer,
                log=True,
            ).item()
            portability_accs.append(portability_acc)
        results["text_portability_accuracy"] = np.mean(portability_accs)
    # Multimodal portability evaluation
    if "multimodal_portability_prompt" in record.keys():
        if isinstance(record["multimodal_portability_prompt"], str):
            record["multimodal_portability_prompt"] = [record["multimodal_portability_prompt"]]
        if isinstance(record["multimodal_portability_ground_truth"], str):
            record["multimodal_portability_ground_truth"] = [record["multimodal_portability_ground_truth"]]
        portability_accs = []
        for port_prompt, port_target in zip(
            record["multimodal_portability_prompt"], record["multimodal_portability_ground_truth"], strict=True
        ):
            if add_new_fact:
                port_prompt = f"New Fact: {prompt} {target}\nPrompt: {port_prompt}"
            port_prompt = "".join(icl_examples) + port_prompt
            port_edit = prepare_multimodal_hf_edit(processor, port_target, port_prompt, image)
            portability_acc = metrics["image_portability"](
                model=model,
                batch=dict(image_portability=port_edit, sample_ids=sample_ids),
                tokenizer=processor.processor.tokenizer,
                log=True,
            ).item()
            portability_accs.append(portability_acc)
        results["image_portability_accuracy"] = np.mean(portability_accs)

    validate_results(results, pre_edit=False)
    return results


def compute_multimodal_edit_metrics(
    model,
    processor: AutoProcessor,
    metrics: dict,
    record: dict,
) -> dict:
    model.eval()
    results = {}
    # First, unpack reliability evaluation record.
    image = record["image"]
    prompt = record["prompt"]
    target = record["target"]
    sample_ids = [record["cond"]]

    # Reliability evaluation
    reliability_batch = prepare_multimodal_hf_edit(processor, targets=target, prompts=prompt, images=image)
    results["reliability_accuracy"] = metrics["reliability"](
        model=model,
        batch=dict(tf_edit_reliability=reliability_batch, sample_ids=sample_ids),
        tokenizer=processor.processor.tokenizer,
        log=True,
    ).item()
    # Generality evaluation
    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        rephrase_batch = prepare_multimodal_hf_edit(processor, targets=target, prompts=rephrase_prompts, images=image)
        results["text_generality_accuracy"] = metrics["edit_generality"](
            model=model,
            batch=dict(tf_edit_generality=rephrase_batch, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()
    # Image generality evaluation
    if "image_generality" in record.keys():
        rephrase_image = record["image_generality"]
        edit_image_outer = prepare_multimodal_hf_edit(processor, targets=target, prompts=prompt, images=rephrase_image)
        results["image_generality_accuracy"] = metrics["edit_generality_image"](
            model=model,
            batch=dict(tf_edit_generality_image=edit_image_outer, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()
    # Locality evaluation
    if "locality_prompt" in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_pre_edit_answer"]
        locality_batch = prepare_multimodal_hf_edit(
            processor, targets=locality_ground_truth, prompts=locality_prompt, images=None
        )
        results["text_locality_accuracy"] = metrics["loc"](
            model=model,
            batch=dict(tf_loc=locality_batch, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()
    # Multimodal locality evaluation
    if "multimodal_locality_prompt" in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_pre_edit_answer"]
        m_loc_image = record["multimodal_locality_image"]
        m_locality_batch = prepare_multimodal_hf_edit(
            processor, targets=m_loc_ground_truth, prompts=m_loc_prompt, images=m_loc_image
        )
        results["image_locality_accuracy"] = metrics["loc_image"](
            model=model,
            batch=dict(tf_loc_image=m_locality_batch, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()
    # Portability evaluation
    if "text_portability_prompt" in record.keys():
        if isinstance(record["text_portability_prompt"], list):
            num_samples = len(record["text_portability_prompt"])
            results["text_portability_accuracy"] = 0
            for idx in range(num_samples):
                port_q = record["text_portability_prompt"][idx]
                port_a = record["text_portability_ground_truth"][idx]
                # Portability evaluation
                port_batch = prepare_multimodal_hf_edit(processor, targets=port_a, prompts=port_q, images=None)
                processor.prepare_labels(port_batch, [port_q], [port_a], [None])
                port_batch.to(model.device)
                acc = metrics["text_portability"](
                    model=model,
                    batch=dict(text_portability=port_batch, sample_ids=sample_ids),
                    tokenizer=processor.processor.tokenizer,
                    log=True,
                ).item()
                results["text_portability_accuracy"] += acc / num_samples

    if "multimodal_portability_ground_truth" in record.keys():
        if isinstance(record["multimodal_portability_ground_truth"], list):
            num_samples = len(record["multimodal_portability_ground_truth"])
            results["image_portability_accuracy"] = 0
            for idx in range(num_samples):
                port_image = image
                port_q = record["multimodal_portability_prompt"][idx]
                port_a = record["multimodal_portability_ground_truth"][idx]
                # Portability evaluation
                port_batch = prepare_multimodal_hf_edit(processor, targets=port_a, prompts=port_q, images=port_image)
                processor.prepare_labels(port_batch, [port_q], [port_a], [port_image])
                port_batch.to(model.device)
                acc = metrics["image_portability"](
                    model=model,
                    batch=dict(image_portability=port_batch, sample_ids=sample_ids),
                    tokenizer=processor.processor.tokenizer,
                    log=True,
                ).item()
                results["image_portability_accuracy"] += acc / num_samples
    # verify results type
    validate_results(results, pre_edit=False)
    return results


def compute_icl_multimodal_edit_metrics_with_generation(
    model,
    processor,
    metrics,
    icl_examples,
    record: dict,
    add_new_fact: bool,
) -> dict:
    info_dict = {}
    # First, unpack reliability evaluation record.
    target = record["target"]
    prompt = record["prompt"]
    image = record["image"]
    new_fact = f"New Fact: {prompt} {target}\nPrompt: {prompt}"
    if add_new_fact:
        text = "".join(icl_examples) + new_fact
    else:
        text = "".join(icl_examples) + prompt
    sample_ids = [record["cond"]]
    reliability_batch = prepare_multimodal_hf_edit(processor, targets=target, prompts=text, images=image)
    info_dict["reliability_accuracy"] = metrics["reliability"](
        model=model,
        batch=dict(edit_reliability=reliability_batch, sample_ids=sample_ids),
        tokenizer=processor.processor.tokenizer,
        log=True,
    ).item()

    rephrase = record["rephrase_prompt"] if "rephrase_prompt" in record.keys() else None
    rephrase_image = record["image_generality"] if "image_generality" in record.keys() else None

    if rephrase is not None:
        rephrase_text = "".join(icl_examples) + f"New Fact: {prompt} {target}\nPrompt: {rephrase}"
        rephrase_batch = prepare_multimodal_hf_edit(processor, targets=target, prompts=rephrase_text, images=image)
        info_dict["text_generality_accuracy"] = metrics["edit_generality"](
            model=model,
            batch=dict(edit_generality=rephrase_batch, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()

    if rephrase_image is not None:
        rephrase_image_text = "".join(icl_examples) + new_fact
        rephrase_image_batch = prepare_multimodal_hf_edit(
            processor, targets=target, prompts=rephrase_image_text, images=rephrase_image
        )
        info_dict["image_generality_accuracy"] = metrics["edit_generality_image"](
            model=model,
            batch=dict(edit_generality_image=rephrase_image_batch, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()

    if "locality_prompt" in record.keys():
        loc_q = record["locality_prompt"]
        loc_a = record["locality_pre_edit_answer"]
        locality_text = "".join(icl_examples) + f"New Fact: {prompt} {target}\nPrompt: {loc_q}"
        locality_batch = prepare_multimodal_hf_edit(processor, targets=loc_a, prompts=locality_text, images=None)
        processor.prepare_labels(locality_batch, [locality_text], [loc_a], [None], new_key="pre_edit_labels")
        locality_batch.to(model.device)

        info_dict["text_locality_accuracy"] = metrics["loc"](
            model=model,
            batch=dict(loc=locality_batch, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()

    if "multimodal_locality_image" in record.keys():
        m_loc_image = record["multimodal_locality_image"]
        m_loc_q = record["multimodal_locality_prompt"]
        m_loc_a = record["multimodal_locality_pre_edit_answer"]
        m_loc_q_with_fact = f"New Fact: {prompt} {target}\nPrompt: {m_loc_q}"
        m_locality_text = "".join(icl_examples) + m_loc_q_with_fact
        m_locality_batch = prepare_multimodal_hf_edit(
            processor, targets=m_loc_a, prompts=m_locality_text, images=m_loc_image
        )
        processor.prepare_labels(
            m_locality_batch, [m_locality_text], [m_loc_a], [m_loc_image], new_key="pre_edit_labels"
        )
        m_locality_batch.to(model.device)
        info_dict["image_locality_accuracy"] = metrics["loc_image"](
            model=model,
            batch=dict(loc_image=m_locality_batch, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()

    if "text_portability_prompt" in record.keys():
        if isinstance(record["text_portability_prompt"], list):
            num_samples = len(record["text_portability_prompt"])
            info_dict["text_portability_accuracy"] = 0
            info_dict["text_portability_consistency_accuracy"] = 0
            for idx in range(num_samples):
                port_q = record["text_portability_prompt"][idx]
                port_a = record["text_portability_ground_truth"][idx]
                port_consistency_a = record["text_portability_pre_edit_answer"][idx]
                port_q_with_fact = f"New Fact: {prompt} {target}\nPrompt: {port_q}"
                port_text = "".join(icl_examples) + port_q_with_fact
                # Portability evaluation
                port_batch = prepare_multimodal_hf_edit(processor, targets=port_a, prompts=port_text, images=None)
                processor.prepare_labels(port_batch, [port_text], [port_a], [None])
                port_batch.to(model.device)
                acc = metrics["text_portability"](
                    model=model,
                    batch=dict(text_portability=port_batch, sample_ids=sample_ids),
                    tokenizer=processor.processor.tokenizer,
                    log=True,
                ).item()
                info_dict["text_portability_accuracy"] += acc / num_samples
                # Portability consistency evaluation
                port_batch = prepare_multimodal_hf_edit(
                    processor, targets=port_consistency_a, prompts=port_text, images=None
                )
                processor.prepare_labels(
                    port_batch, [port_text], [port_consistency_a], [None], new_key="pre_edit_labels"
                )
                port_batch.to(model.device)
                acc = metrics["text_portability_consistency"](
                    model=model,
                    batch=dict(text_portability=port_batch, sample_ids=sample_ids),
                    tokenizer=processor.processor.tokenizer,
                    log=True,
                ).item()
                info_dict["text_portability_consistency_accuracy"] += acc / num_samples

    if "multimodal_portability_prompt" in record.keys():
        if isinstance(record["multimodal_portability_prompt"], list):
            num_samples = len(record["multimodal_portability_prompt"])
            info_dict["multimodal_portability_accuracy"] = 0
            info_dict["multimodal_portability_consistency_accuracy"] = 0
            for idx in range(num_samples):
                port_image = image
                port_q = record["multimodal_portability_prompt"][idx]
                port_a = record["multimodal_portability_ground_truth"][idx]
                port_consistency_a = record["multimodal_portability_pre_edit_answer"][idx]
                port_q_with_fact = f"New Fact: {prompt} {target}\nPrompt: {port_q}"
                port_text = "".join(icl_examples) + port_q_with_fact
                # Portability evaluation
                port_batch = prepare_multimodal_hf_edit(processor, targets=port_a, prompts=port_text, images=port_image)
                processor.prepare_labels(port_batch, [port_text], [port_a], [port_image])
                port_batch.to(model.device)
                acc = metrics["image_portability"](
                    model=model,
                    batch=dict(image_portability=port_batch, sample_ids=sample_ids),
                    tokenizer=processor.processor.tokenizer,
                    log=True,
                ).item()
                info_dict["image_portability_accuracy"] += acc / num_samples
                # Portability consistency evaluation
                port_batch = prepare_multimodal_hf_edit(
                    processor, targets=port_consistency_a, prompts=port_text, images=port_image
                )
                processor.prepare_labels(
                    port_batch, [port_text], [port_consistency_a], [port_image], new_key="pre_edit_labels"
                )
                port_batch.to(model.device)
                acc = metrics["image_portability_consistency"](
                    model=model,
                    batch=dict(image_portability=port_batch, sample_ids=sample_ids),
                    tokenizer=processor.processor.tokenizer,
                    log=True,
                ).item()
                info_dict["image_portability_consistency_accuracy"] += acc / num_samples

    return info_dict


def compute_multimodal_edit_metrics_with_generation(
    model,
    processor: AutoProcessor,
    metrics: dict,
    record: dict,
) -> dict:
    model.eval()
    results = {}
    # First, unpack reliability evaluation record.
    image = record["image"]
    prompt = record["prompt"]
    target = record["target"]
    sample_ids = [record["cond"]]

    # Reliability evaluation
    reliability_batch = prepare_multimodal_hf_edit(processor, targets=target, prompts=prompt, images=image)
    results["reliability_accuracy"] = metrics["reliability"](
        model=model,
        batch=dict(edit_reliability=reliability_batch, sample_ids=sample_ids),
        tokenizer=processor.processor.tokenizer,
        log=True,
    ).item()
    # Generality evaluation
    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        rephrase_batch = prepare_multimodal_hf_edit(processor, targets=target, prompts=rephrase_prompts, images=image)
        results["text_generality_accuracy"] = metrics["edit_generality"](
            model=model,
            batch=dict(edit_generality=rephrase_batch, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()
    # Image generality evaluation
    if "image_generality" in record.keys():
        rephrase_image = record["image_generality"]
        rephrase_image_batch = prepare_multimodal_hf_edit(
            processor, targets=target, prompts=prompt, images=rephrase_image
        )
        results["image_generality_accuracy"] = metrics["edit_generality_image"](
            model=model,
            batch=dict(edit_generality_image=rephrase_image_batch, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()
    # Locality evaluation
    if "locality_prompt" in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_pre_edit_answer"]
        locality_batch = prepare_multimodal_hf_edit(
            processor, targets=locality_ground_truth, prompts=locality_prompt, images=None
        )
        locality_batch.to(model.device)

        results["text_locality_accuracy"] = metrics["loc"](
            model=model,
            batch=dict(loc=locality_batch, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()
    # Multimodal locality evaluation
    if "multimodal_locality_image" in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_pre_edit_answer"]
        m_loc_image = record["multimodal_locality_image"]
        m_locality_batch = prepare_multimodal_hf_edit(
            processor, targets=m_loc_ground_truth, prompts=m_loc_prompt, images=m_loc_image
        )
        m_locality_batch.to(model.device)
        results["image_locality_accuracy"] = metrics["loc_image"](
            model=model,
            batch=dict(loc_image=m_locality_batch, sample_ids=sample_ids),
            tokenizer=processor.processor.tokenizer,
            log=True,
        ).item()

    if "text_portability_prompt" in record.keys():
        if isinstance(record["text_portability_prompt"], list):
            num_samples = len(record["text_portability_prompt"])
            results["text_portability_accuracy"] = 0
            results["text_portability_consistency_accuracy"] = 0
            for idx in range(num_samples):
                port_q = record["text_portability_prompt"][idx]
                port_a = record["text_portability_ground_truth"][idx]
                port_consistency_a = record["text_portability_pre_edit_answer"][idx]
                # Portability evaluation
                port_batch = prepare_multimodal_hf_edit(processor, targets=port_a, prompts=port_q, images=None)
                processor.prepare_labels(port_batch, [port_q], [port_a], [None])
                port_batch.to(model.device)
                acc = metrics["text_portability"](
                    model=model,
                    batch=dict(text_portability=port_batch, sample_ids=sample_ids),
                    tokenizer=processor.processor.tokenizer,
                    log=True,
                ).item()
                results["text_portability_accuracy"] += acc / num_samples
                # Portability consistency evaluation
                port_batch = prepare_multimodal_hf_edit(
                    processor, targets=port_consistency_a, prompts=port_q, images=None
                )
                processor.prepare_labels(port_batch, [port_q], [port_consistency_a], [None], new_key="pre_edit_labels")
                port_batch.to(model.device)
                acc = metrics["text_portability_consistency"](
                    model=model,
                    batch=dict(text_portability=port_batch, sample_ids=sample_ids),
                    tokenizer=processor.processor.tokenizer,
                    log=True,
                ).item()
                results["text_portability_consistency_accuracy"] += acc / num_samples

    if "multimodal_portability_ground_truth" in record.keys():
        if isinstance(record["multimodal_portability_ground_truth"], list):
            num_samples = len(record["multimodal_portability_ground_truth"])
            results["image_portability_accuracy"] = 0
            results["image_portability_consistency_accuracy"] = 0
            for idx in range(num_samples):
                port_image = image
                port_q = record["multimodal_portability_prompt"][idx]
                port_a = record["multimodal_portability_ground_truth"][idx]
                port_consistency_a = record["multimodal_portability_pre_edit_answer"][idx]
                # Portability evaluation
                port_batch = prepare_multimodal_hf_edit(processor, targets=port_a, prompts=port_q, images=port_image)
                processor.prepare_labels(port_batch, [port_q], [port_a], [port_image])
                port_batch.to(model.device)
                acc = metrics["image_portability"](
                    model=model,
                    batch=dict(image_portability=port_batch, sample_ids=sample_ids),
                    tokenizer=processor.processor.tokenizer,
                    log=True,
                ).item()
                results["image_portability_accuracy"] += acc / num_samples
                # Portability consistency evaluation
                port_batch = prepare_multimodal_hf_edit(
                    processor, targets=port_consistency_a, prompts=port_q, images=port_image
                )
                processor.prepare_labels(
                    port_batch, [port_q], [port_consistency_a], [port_image], new_key="pre_edit_labels"
                )
                port_batch.to(model.device)
                acc = metrics["image_portability_consistency"](
                    model=model,
                    batch=dict(image_portability=port_batch, sample_ids=sample_ids),
                    tokenizer=processor.processor.tokenizer,
                    log=True,
                ).item()
                results["image_portability_consistency_accuracy"] += acc / num_samples

    # verify results type
    validate_results(results, pre_edit=False)
    return results


EVALUATION_UTILS_MAPPING = {
    "IKE_with_teacher_forcing": compute_icl_multimodal_edit_metrics,
    "IKE_with_generation": compute_icl_multimodal_edit_metrics_with_generation,
    "multimodal_with_teacher_forcing": compute_multimodal_edit_metrics,
    "multimodal_with_generation": compute_multimodal_edit_metrics_with_generation,
}
