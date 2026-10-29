from ...trainer.utils import dict_to
from ..processor.multimodal_processor import (
    Gemma3MultimodalProcessor,
    LLAVAMultimodalProcessor,
    MultimodalProcessor,
    Qwen2MultimodalProcessor,
    Qwen3MultimodalProcessor,
)


class TrainDataCollatorForVLM:
    def __init__(
        self,
        config,
        max_length: int = 8192,
        return_tensors: str = "pt",
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.verbose = verbose
        if "Qwen2" in config.model_name:
            self.processor = Qwen2MultimodalProcessor(config, max_length=max_length, return_tensors=return_tensors)
        elif "Qwen3" in config.model_name:
            self.processor = Qwen3MultimodalProcessor(config, max_length=max_length, return_tensors=return_tensors)
        elif "gemma-3" in config.model_name:
            self.processor = Gemma3MultimodalProcessor(config, max_length=max_length, return_tensors=return_tensors)
        elif "llava" in config.model_name:
            self.processor = LLAVAMultimodalProcessor(config, max_length=max_length, return_tensors=return_tensors)
        else:
            self.processor = MultimodalProcessor(config, max_length=max_length, return_tensors=return_tensors)

    def prepare_inputs(self, text, label, images):
        inputs = self.processor.prepare_inputs(text, label, images)
        self.processor.prepare_labels(inputs, text, label, images)
        return inputs

    def prepare_locality_inputs(self, features, key):
        loc_q = [feature[f"{key}_prompt"] for feature in features]
        loc_a = [feature[f"{key}_pre_edit_answer"] for feature in features]
        if "multimodal" in key:
            loc_image = [feature[f"{key}_image"] for feature in features]
        else:
            loc_image = [None] * len(loc_q)
        return self.prepare_inputs(loc_q, loc_a, loc_image)

    def prepare_text_qa_inputs(self, features, question_key, answer_key):
        questions = [feature[question_key] for feature in features]
        answers = [feature[answer_key] for feature in features]
        if questions[0] is None:
            return None
        return self.prepare_inputs(questions, answers, [None] * len(questions))

    def prepare_multimodal_portability_inputs(self, features, images):
        text = [feature["multimodal_portability_prompt"] for feature in features]
        label = [feature["multimodal_portability_ground_truth"] for feature in features]
        return self.prepare_inputs(text, label, images)

    def __call__(self, features, return_tensors=None):
        # edit_reliability: Edit Sample: question - image - answer
        src = [feature["prompt"] for feature in features]
        trg = [feature["target"] for feature in features]

        image = [feature["image"] for feature in features]
        edit_reliability = self.prepare_inputs(src, trg, image)

        # edit_generality: Generality Sample: rephrase - image - answer
        rephrase = [feature["rephrase_prompt"] for feature in features]
        edit_generality = self.prepare_inputs(rephrase, trg, image)

        # edit_generality_image: Generality Sample: question - rephrase image - answer
        image_generality = [feature["image_generality"] for feature in features]
        edit_generality_image = self.prepare_inputs(src, trg, image_generality)

        # loc: Locality Sample: locality question - locality answer
        loc = self.prepare_locality_inputs(features=features, key="locality")

        # m_loc: Multimodal Locality Sample: multimodal locality question - multimodal locality image - multimodal locality answer
        loc_image = self.prepare_locality_inputs(features=features, key="multimodal_locality")

        # text portability: Text Portability Sample: portability question - portability answer
        portability = self.prepare_text_qa_inputs(features, "text_portability_prompt", "text_portability_ground_truth")

        # multimodal portability: Portability Sample: portability question - portability answer
        portability_image = self.prepare_multimodal_portability_inputs(features, image)

        # background: Background Sample: background question - background answer
        background = self.prepare_text_qa_inputs(features, "background_prompt", "multimodal_portability_ground_truth")

        # cond
        cond = [feature["cond"] for feature in features]
        cond = self.processor.tokenizer(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )["input_ids"]

        batch = {
            "edit_reliability": edit_reliability,
            "edit_generality": edit_generality,
            "edit_generality_image": edit_generality_image,
            "portability": portability,
            "portability_image": portability_image,
            "background": background,
            "loc": loc,
            "loc_image": loc_image,
            "cond": cond,
        }
        if self.verbose:
            for metric in batch:
                if metric == "cond" or batch[metric] is None:
                    continue
                print("\n", metric)
                if not isinstance(batch[metric], list):
                    metric_batch = [batch[metric]]
                else:
                    metric_batch = batch[metric]
                for batch_ in metric_batch:
                    input_ids = batch_["input_ids"]
                    labels = batch_["labels"].clone()
                    labels[labels == -100] = self.processor.tokenizer.pad_token_id
                    text = self.processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    output = self.processor.tokenizer.batch_decode(labels, skip_special_tokens=False)
                    print("input", text)
                    print("output", output)

        return dict_to(batch, self.config.device)


class EvalDataCollatorForVLM(TrainDataCollatorForVLM):
    def prepare_locality_inputs(self, features, key):
        loc_q = [feature[f"{key}_prompt"] for feature in features]
        pre_edit_label = [feature[f"{key}_pre_edit_answer"] for feature in features]
        if "multimodal" in key:
            loc_image = [feature[f"{key}_image"] for feature in features]
        else:
            loc_image = [None] * len(loc_q)
        loc_inputs = self.prepare_inputs(loc_q, pre_edit_label, loc_image)
        self.processor.prepare_labels(loc_inputs, loc_q, pre_edit_label, loc_image, new_key="pre_edit_labels")
        return loc_inputs

    def prepare_text_qa_inputs(self, features, question_key, answer_key):
        questions = [feature[question_key] for feature in features]
        answers = [feature[answer_key] for feature in features]
        if questions[0] is None:
            return None
        if isinstance(questions[0], list):
            inputs = [self.prepare_inputs(q, a, [None] * len(q)) for q, a in zip(questions, answers)]
            for inp, t, l in zip(inputs, questions, answers, strict=True):
                self.processor.prepare_labels(inp, t, l, [None] * len(t), new_key="pre_edit_labels")
        else:
            images = [None] * len(questions)
            inputs = self.prepare_inputs(questions, answers, images)
            self.processor.prepare_labels(inputs, questions, answers, images, new_key="pre_edit_labels")
        return inputs

    def prepare_multimodal_portability_inputs(self, features, images):
        text = [feature["multimodal_portability_prompt"] for feature in features]
        label = [feature["multimodal_portability_ground_truth"] for feature in features]
        if isinstance(text[0], list):
            inputs = [self.prepare_inputs(t, l, [i] * len(t)) for t, l, i in zip(text, label, images)]
            for inp, t, l, i in zip(inputs, text, label, images, strict=True):
                self.processor.prepare_labels(inp, t, l, [i] * len(t), new_key="pre_edit_labels")
        else:
            inputs = self.prepare_inputs(text, label, images)
            self.processor.prepare_labels(inputs, text, label, images, new_key="pre_edit_labels")
        return inputs
