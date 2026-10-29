import re

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor


class MultimodalProcessor:
    _processor_config = {}

    def __init__(
        self,
        config,
        max_length: int = 8192,
        return_tensors: str = "pt",
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(
            config.tokenizer_name,
            do_image_splitting=False,
            trust_remote_code=True,
            **self._processor_config,
        )
        try:
            self.tokenizer = self.processor.tokenizer
        except AttributeError:
            self.tokenizer = self.processor._tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.prompt = "{question}\nAnswer with a single word or phrase."

    def prepare_question(self, question: str) -> str:
        return self._get_question(question)

    def prepare_answer(self, answer: str, capitalize: bool = True) -> str:
        answer = answer.strip()
        # Make sure the first letter is capitalized
        if capitalize:
            answer = answer[0].upper() + answer[1:]
        # Remove any trailing punctuation
        answer = re.sub(r"[^\w\s]+$", "", answer)
        return answer

    def prepare_messages_for_generation(self, question, image):
        return self.prepare_messages(question=question, label=None, image=image)

    def apply_chat_template(self, messages, add_generation_prompt: bool = False):
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def prepare_input_text(self, text: str, label: str, image) -> str:
        messages = self.prepare_messages(text, label, image)
        return self.remove_end_token(self.apply_chat_template(messages, add_generation_prompt=False))

    def prepare_generation_input_text(self, text: str, image) -> str:
        messages = self.prepare_messages_for_generation(text, image)
        return self.apply_chat_template(messages, add_generation_prompt=True)

    def prepare_inputs(self, texts: list[str], labels: list[str], images: list):
        messages = [self.prepare_messages(text, label, img) for text, label, img in zip(texts, labels, images)]
        texts = [self.prepare_input_text(text, label, img) for text, label, img in zip(texts, labels, images)]
        images = self._prepare_images(messages, images)
        return self._tokenize(texts, images)

    def prepare_inputs_for_generation(self, texts: list[str], images: list):
        messages = [self.prepare_messages_for_generation(text, img) for text, img in zip(texts, images)]
        texts = [self.prepare_generation_input_text(text, img) for text, img in zip(texts, images)]
        images = self._prepare_images(messages, images)
        return self._tokenize(texts, images)

    def prepare_labels(self, inputs, texts, labels, images, new_key="labels"):
        messages = [self.prepare_input_text(text, label, img) for text, label, img in zip(texts, labels, images)]
        parts = []
        for message in messages:
            sub_parts = message.split(self._labels_split_str)
            assert len(sub_parts) >= 2
            suffix = sub_parts[-1]
            prefix = self._labels_split_str.join(sub_parts[:-1]) if len(sub_parts) > 2 else sub_parts[0]
            parts.append([prefix, suffix])

        assert all(len(parts) == 2 for parts in parts)
        label_tokens = inputs["input_ids"].clone()
        for sample_index, sample_parts in enumerate(parts):
            labels_len = self._get_labels_len(sample_parts[-1])
            if self.tokenizer.padding_side == "right":
                # get the number of padding tokens at the end of the tensor
                num_padding_tokens = self._get_right_padding_tokens_count(label_tokens[sample_index])
                labels_len += num_padding_tokens

            label_tokens[sample_index, :-labels_len] = -100

        label_tokens[label_tokens == self.tokenizer.pad_token_id] = -100
        inputs[new_key] = label_tokens

    def _get_right_padding_tokens_count(self, tensor: torch.Tensor):
        # Reverse the tensor and count padding tokens until a non-padding token is encountered
        padding_count = 0
        for value in torch.flip(tensor, dims=[0]):
            if value == self.tokenizer.pad_token_id:
                padding_count += 1
            else:
                break
        return padding_count

    def _get_question(self, question: str) -> str:
        # Make sure the question ends with a question mark
        question = question.split("\nAnswer with")[0]
        if not question.endswith(("?", "？", ".")):
            question += "?"
        question = question.strip().strip("\n")
        # Make sure the first letter is capitalized
        question = question[0].upper() + question[1:]
        return question

    def _tokenize(self, texts, images):
        inputs = self.processor(
            text=texts,
            images=images,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            add_special_tokens=self._add_special_tokens,
        )
        # print(self.tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=False))
        return inputs

    def _prepare_images(self, messages, images):
        if images is None or all(img is None for img in images):
            images = None
        else:
            images = [img for img in images if img is not None]
        return images


class LLAVAMultimodalProcessor(MultimodalProcessor):
    _processor_config = {"image_grid_pinpoints": [[336, 336]], "padding_side": "left"}

    _labels_split_str = "[/INST] "

    def prepare_messages(self, question, label, image):
        question = self.prompt.format(question=question)
        if image is not None:
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]
        else:
            content = [{"type": "text", "text": question}]
        message = [
            {
                "role": "user",
                "content": content,
            }
        ]
        if label is not None:
            message.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": label}],
                }
            )
        return message

    def remove_end_token(self, text: str, token: str = "</s> ") -> str:
        if text.endswith(token):
            return text[: -len(token)]
        return text

    def _prepare_images(self, messages, images):
        images = [img for img in images if img is not None]
        return images

    def _get_labels_len(self, label_text):
        labels_len = self.tokenizer(label_text, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[1]
        # Remove bos token
        labels_len -= 1
        return labels_len


class Qwen2MultimodalProcessor(MultimodalProcessor):

    _processor_config = {"max_pixels": 336 * 336, "padding_side": "left"}
    _labels_split_str = "<|im_start|>assistant\n"

    def prepare_messages(self, question, label, image):
        question = self.prompt.format(question=question)
        if image is not None:
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]
        else:
            content = [{"type": "text", "text": question}]
        message = [
            {
                "role": "user",
                "content": content,
            }
        ]
        if label is not None:
            message.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": label}],
                }
            )
        return message

    def remove_end_token(self, text: str, token: str = "\n") -> str:
        if text.endswith(token):
            return text[: -len(token)]
        return text

    def _prepare_images(self, messages, images):
        if images is None or all(img is None for img in images):
            images = None
        else:
            images, _ = process_vision_info(messages)
        return images

    def _get_labels_len(self, label_text):
        labels_len = self.tokenizer(label_text, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[1]
        return labels_len


class Qwen3MultimodalProcessor(Qwen2MultimodalProcessor):

    _processor_config = {
        "size": {"longest_edge": 512 * 512, "shortest_edge": 4096},
        "do_resize": True,
        "padding_side": "left",
        "add_eos_token": True,
    }
    _add_special_tokens = True

    def _prepare_images(self, messages, images):
        if images is None or all(img is None for img in images):
            images = None
        else:
            images = [img for img in images if img is not None]
        return images


class Gemma3MultimodalProcessor(MultimodalProcessor):
    # fixes for gemma
    # 1. images need to be nested per sample in batch
    # 2. need to remove assertion of number of texts equal to images in processor
    # 3. decoder layers should not be sliced for mend to  work modeling_gemma3.py (line: 566)

    # apply_chat_template adds <bos> but not <eos>
    # then the tokenizer adds another <bos> and an <eos>

    _labels_split_str = "<start_of_turn>model\n"
    _add_special_tokens = False
    _processor_config = {"add_eos_token": False}

    def prepare_messages(self, question, label, image):
        question = self.prompt.format(question=question)
        if image is not None:
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]
        else:
            content = [{"type": "text", "text": question}]
        message = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": content,
            },
        ]
        if label is not None:
            message.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": label}],
                }
            )
        return message

    def apply_chat_template(self, messages, add_generation_prompt: bool = False):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        
        if not add_generation_prompt and not text.endswith(self.tokenizer.eos_token):
            text = self.remove_end_token(text)
            text += self.tokenizer.eos_token
        return text

    def remove_end_token(self, text: str, token: str = "\n") -> str:
        if text.endswith(token):
            return text[: -len(token)]
        return text

    def _get_labels_len(self, label_text):
        return self.tokenizer(label_text, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]

    def _prepare_images(self, messages, images):
        if images is None or all(img is None for img in images):
            images = None
        else:
            images = [[img] for img in images if img is not None]
        return images
