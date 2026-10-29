def get_logits(x):
    return x.logits if hasattr(x, "logits") else x


def multimodal_tokenize(batch, processor, device, target_key="target"):
    prompts = [item["prompt"] for item in batch]
    input_images = [item["image"] for item in batch]
    targets = [item[target_key] for item in batch]

    multimodal_inputs = processor.prepare_inputs(prompts, targets, input_images)
    processor.prepare_labels(multimodal_inputs, prompts, targets, input_images)

    multimodal_inputs.to(device)
    return multimodal_inputs
