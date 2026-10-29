import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


def kl_loc_loss(pre, post, mask=None):
    pre = pre.to(torch.float32)
    post = post.to(torch.float32)

    sequence = pre.dim() == 3
    pre_ = pre.contiguous().view(-1, pre.shape[-1])
    post_ = post.contiguous().view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
    else:  # We have sequences of predictions; masking needed
        if pre_.shape[-1] > 1:
            assert mask is not None
            mask_ = mask.view(pre_.shape[0])
            kl = (pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))).sum(-1)
            return (kl * mask_).sum() / mask_.sum()

    raise NotImplementedError


def binary_log_probs(pred, targ):
    neg_mask = torch.ones_like(pred)
    neg_mask[targ == 0] *= -1
    pred = pred * neg_mask
    log_probs = F.logsigmoid(pred)
    acc = (log_probs.exp() > 0.5).float().mean()
    return {
        "acc": acc,
        "log_prob": log_probs.mean(),
        "prob": log_probs.exp().mean(),
        "nll": -log_probs.mean(),
        "n_tokens": log_probs.shape[0],
    }


def masked_mean(values, mask):
    assert mask.dtype == torch.bool
    assert values.shape == mask.shape
    return (values * mask.float()).sum() / mask.sum().float()


def mask_hf_labels(labels, null_token=0):
    valid_mask = labels != -100
    valid_labels = labels.masked_fill(~valid_mask, null_token)
    return valid_mask, valid_labels


def multiclass_accuracy(pred, targ, mask, null_token=0, exact_match=False):
    pred_ids = pred.argmax(-1).masked_fill(~mask, null_token)
    correct = pred_ids == targ
    if exact_match:
        if pred.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        correct = correct & mask
        num_non_padding = mask.sum().float().item()

        acc = correct.sum() / num_non_padding
    return acc


def multiclass_log_probs(pred, targ, mask, eps=torch.finfo(torch.float32).eps, **kwargs):
    unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)

    if "inner_sent" in kwargs or "inner_per" in kwargs:
        same_sent_mask = kwargs["same_mask"]
        good_mask = mask * same_sent_mask.unsqueeze(-1)
        bad_mask = mask * (~same_sent_mask.unsqueeze(-1))

        good_log_prob = masked_mean(unmasked_log_probs, good_mask)
        bad_log_prob = masked_mean((1 - unmasked_log_probs.exp() + eps).log(), bad_mask)

        n_tokens = good_mask.float().sum()
        log_prob = good_log_prob
        prob = log_prob.exp()

        if kwargs["unlikelihood"]:
            nll = -good_log_prob - bad_log_prob
        else:
            nll = -good_log_prob
    else:
        n_tokens = mask.float().sum()
        log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
        prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens

        nll = -log_prob

    output_dict = {
        "log_prob": log_prob,
        "prob": prob,
        "n_tokens": n_tokens,
        "nll": nll,
    }
    return output_dict


def masked_cross_entropy_loss(inputs, logits, label_mask=None, **kwargs):
    """Compute the cross-entropy loss with optional masking.

    Args:
        inputs: Tensor of shape (batch_size, seq_len) with target token IDs.
        logits: Tensor of shape (batch_size, seq_len, vocab_size) with model predictions.
        label_mask: Optional tensor of shape (batch_size, seq_len) indicating valid positions (1 for valid, 0 for ignore).
    """
    batch_size = inputs.size(0)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(batch_size, -1)
    if label_mask is not None:
        loss = (loss * label_mask[:, 1:]).sum(1) / label_mask[:, 1:].sum(1)
    return loss.mean()


def multiclass_acc_and_log_probs(
    pred, targ, shift=True, eps=torch.finfo(torch.float32).eps, exact_match=False, **kwargs
):
    NULL_TOKEN = 0  # a placeholder used for masked target locations
    pred = pred.to(torch.float32)
    pred = pred.clone()
    targ = targ.clone()
    if shift and pred.dim() == 3:  # Dealing with sequences
        pred = pred[:, :-1]  # Remove last prediction in sequence
        targ = targ[:, 1:]

    mask = targ != -100
    targ[~mask] = NULL_TOKEN  # Can be any valid token, since we'll throw them out
    acc = multiclass_accuracy(pred, targ, mask=mask, exact_match=exact_match, null_token=NULL_TOKEN)
    output_dict = multiclass_log_probs(pred=pred, targ=targ, mask=mask, eps=eps, **kwargs)
    output_dict.update({"acc": acc})

    return output_dict


def masked_log_probs(config, pred, targ, shift=True, exact_match=False, **kwargs):
    pred = pred.to(torch.float32)

    if not (pred.dim() == 2 or pred.dim() == 3):
        raise RuntimeError(f"Expected pred to have 2 or 3 dimensions, got {pred.shape}")

    if pred.shape[-1] == 1:
        return binary_log_probs(pred, targ)
    else:
        return multiclass_acc_and_log_probs(pred, targ, shift=shift, exact_match=exact_match, **kwargs)


def es(pre_logits, post_logits, targ, same_per_mask, q_mask, NULL_TOKEN=0):
    with torch.no_grad():

        mask = targ != -100
        targ[~mask] = NULL_TOKEN

        pos_mask = same_per_mask.unsqueeze(-1) * q_mask
        neg_mask = ~same_per_mask.unsqueeze(-1) * q_mask

        # Compute log likelihoods of pos/neg samples

        pre_edit_token_log_probs = pre_logits.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
        post_edit_token_log_probs = post_logits.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)

        mean_pos_pre = masked_mean(pre_edit_token_log_probs, pos_mask)
        mean_pos_post = masked_mean(post_edit_token_log_probs, pos_mask)
        mean_neg_post = masked_mean(post_edit_token_log_probs, neg_mask)

        z_per = (mean_pos_post - mean_neg_post).sigmoid()
        z_topic_raw = (mean_pos_post - mean_pos_pre).exp()
        z_topic = min(1, z_topic_raw)

        es_per = z_per * z_topic
        return {
            "acc_per": es_per,
            "z_per": z_per,
            "z_topic": z_topic,
            "z_topic_raw": z_topic_raw,
            "correct_probs": mean_pos_post,
            "wrong_probs": mean_neg_post,
        }
