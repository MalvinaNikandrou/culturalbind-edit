import contextlib
from copy import deepcopy

import higher
import torch
import torch.nn as nn

from ..losses import masked_log_probs
from ..utils import _logits, shift_targets
from .patch import monkeypatch as make_functional


class EditableModel(nn.Module):
    def __init__(self, model, config, model_constructor):
        super().__init__()

        self.model = model
        self.config = deepcopy(config)
        self.model_constructor = model_constructor
        self.weights_copy = None

        def _edit_loss_fn(config, pred, targ, **kwargs):
            if self._is_multimodal():
                kwargs["multimodal"] = True
                return masked_log_probs(config, pred, targ, exact_match=self.config.exact_match, shift=True, **kwargs)
            elif "qwen2-vl" in config.model_name.lower() or "llava" in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=False, **kwargs)
            elif "t5" in config.model_class.lower():
                return masked_log_probs(
                    config,
                    pred,
                    targ,
                )
            elif "gpt" in config.model_class.lower():
                return masked_log_probs(config, pred, targ, shift=True, **kwargs)
            elif "llama" in config.model_class.lower():
                return masked_log_probs(config, pred, targ, shift=True, **kwargs)
            elif "internlm" in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            elif "chatglm" in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            elif "qwen" in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            elif "mistral" in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            else:
                return masked_log_probs(
                    config,
                    pred,
                    targ,
                )

        self.edit_loss_fn = _edit_loss_fn
        self.loc_loss_fn = masked_log_probs

    def set_processor(self, processor):
        self.processor = processor

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare_edit_batch(self, batch, key):
        return batch[key]

    def edit(self, batch, condition=None, detach_history=False):
        raise NotImplementedError

    def forward(self, batch, **kwargs):
        return _logits(self.model(**batch, **kwargs))

    def generate(self, *inputs, **kwargs):
        return self.model.generate(*inputs, **kwargs)

    def outer_parameters(self):
        return self.parameters()

    def base_loss(self, input_ids, attention_masks, label_ids):
        pass

    def get_edit_model(self):
        edited_model = self.model
        if not isinstance(edited_model, higher.patch._MonkeyPatchBase):
            edited_model = make_functional(edited_model, track_higher_grads=False, in_place=True)
        return edited_model

    @torch.no_grad()
    def restore_weights(self, edit_model):
        if self.weights_copy is None:
            return
        # Restore state of original model
        for name, v in edit_model.named_parameters():
            if name not in self.weights_copy:
                continue
            v[...] = self.weights_copy[name]

    def maybe_autocast(self, device):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast("cuda")
        else:
            return contextlib.nullcontext()

    def _is_minigpt4(self):
        return "minigpt4" in self.config.model_name.lower()

    def _is_blip2(self):
        return "blip2" in self.config.model_name.lower()

    def _is_llava_model(self):
        return "llava-hf" in self.config.model_name.lower()

    def _is_qwen3_vl_model(self):
        return "qwen3-vl" in self.config.model_name.lower()

    def _is_gemma3_model(self):
        return "gemma-3" in self.config.model_name.lower()

    def _is_multimodal(self):
        return (
            self._is_minigpt4()
            or self._is_blip2()
            or self._is_llava_model()
            or self._is_qwen3_vl_model()
            or self._is_gemma3_model()
        )

    @property
    def device(self):
        return self.model.device
