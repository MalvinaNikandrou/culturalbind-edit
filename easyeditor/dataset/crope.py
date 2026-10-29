import json
import logging
import typing
from pathlib import Path

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from easyeditor.evaluate.metrics import GenerationPipeline
from easyeditor.evaluate.multimodal_evaluation_utils import prepare_multimodal_hf_edit

from .collator.multimodal_collator import EvalDataCollatorForVLM, TrainDataCollatorForVLM
from .processor.base_dataset import BaseDataset

LOG = logging.getLogger(__name__)


class SequentialSampler:
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples

    def sample(self, annotations):
        concepts = set()
        sample_indices = []
        for index in range(len(annotations)):

            concept = annotations["answer"][index]
            if concept in concepts:
                continue

            sample_indices.append(index)
            concepts.add(concept)
            if len(concepts) >= self.num_samples:
                break
        return annotations.select(sample_indices)


class CropeDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        split: typing.Literal["train", "valid", "test"],
        size: typing.Optional[int] = None,
        config=None,
        *args,
        **kwargs,
    ):
        if config is not None and not hasattr(config, "tokenizer_name"):
            raise ValueError("config must have a tokenizer_name attribute")

        self.max_length = 8129
        if split == "train":
            self.add_portability_data = self.add_portability_data_train
            self.collater = TrainDataCollatorForVLM(config, max_length=self.max_length)
        else:
            self.add_portability_data = self.add_portability_data_eval
            self.collater = EvalDataCollatorForVLM(config, max_length=self.max_length)

        self.split = split
        self.config = config
        # Load the dataset
        hf_split = "validation" if split == "valid" else split
        if Path(data_dir).is_dir():
            self.annotation = load_from_disk(data_dir)[hf_split]
        else:
            self.annotation = load_dataset(data_dir, split=hf_split)

        self.dataset_name = data_dir.replace("/", "_")
        if self.config.sequential_edit:
            self.annotation = self.annotation.shuffle(seed=config.seed)
            self.annotation = SequentialSampler(num_samples=size if size is not None else 100).sample(self.annotation)
        # self.annotation = self.annotation.select(range(25))

        if size is not None:
            max_size = min(size, len(self.annotation))
            self.annotation = self.annotation.shuffle(config.seed)
            self.annotation = self.annotation.select(range(max_size))
        print(f"Loaded {len(self.annotation)} samples for {split} split")

        # self.prompt = "Question: {} Short answer:"
        self.prepare_data()

    @property
    def pre_edit_cache(self) -> Path:
        pre_edit_cache = Path(f"storage/pre_edit_cache/{self.dataset_name}_{self.split}.json")
        pre_edit_cache.parent.mkdir(parents=True, exist_ok=True)
        return pre_edit_cache

    def add_data(self, dataset):
        self._data.extend(dataset._data)

    def prepare_data(self):
        """Prepare a dataset into the required format."""
        data = []
        for i, sample in enumerate(self.annotation):
            image = sample["image"].convert("RGB")
            rephrase_image = sample["generality_image"].convert("RGB")
            locality_image = sample["image_locality_image"].convert("RGB")

            item = {
                "prompt": self.process_question(sample["question"]),
                "pred": sample["pre_edit_answer"],
                "target": self.process_answer(sample["answer"], capitalize=True),
                "target_old": self.process_answer(sample["pre_edit_answer"], capitalize=True),
                "rephrase_prompt": self.process_question(sample["generality_question"]),
                "image": image,
                "image_generality": rephrase_image,
                # Text Locality
                "locality_prompt": self.process_question(sample["text_locality_question"]),
                "locality_ground_truth": self.process_answer(sample["text_locality_answer"], capitalize=True),
                # Multimodal Locality
                "multimodal_locality_image": locality_image,
                "multimodal_locality_prompt": self.process_question(sample["image_locality_question"]),
                "multimodal_locality_ground_truth": self.process_answer(
                    sample["image_locality_answer"], capitalize=True
                ),
                # Metadata
                "cond": "{}---{} >> {} || {}".format(
                    sample["edit_id"], sample["pre_edit_answer"], sample["answer"], sample["question"]
                ),
            }
            assert len(item["cond"].split("---")) == 2

            self.add_portability_data(item, sample)

            data.append(item)
        assert len(data) == len(set([item["cond"] for item in data])), "Duplicate samples found in the dataset!"
        self._data = data

    def update_pre_edit_answers(self, model):
        """Update the pre-edit answers in the dataset using the current model.

        Answers are cached per sample, so runs on different subsets share the cache and only generate missing answers.
        """
        sample_id_to_pre_edit = {}
        if self.pre_edit_cache.exists():
            LOG.info(f"Loading pre-edit answers from cache: {self.pre_edit_cache}")
            with open(self.pre_edit_cache, "r") as f:
                sample_id_to_pre_edit = json.load(f)
        missing = [sample for sample in self._data if sample["cond"] not in sample_id_to_pre_edit]
        if missing:
            generation_pipeline = GenerationPipeline()
            for sample in tqdm(
                missing, desc=f"Generating pre-edit answers for {self.dataset_name}", total=len(missing)
            ):
                pre_edit_data = {}
                for prefix in ["locality", "multimodal_locality"]:
                    prompt = sample[f"{prefix}_prompt"]
                    answer = sample[f"{prefix}_ground_truth"]
                    # answer = sample[f"{prefix}_pre_edit_answer"]
                    image = sample[f"{prefix}_image"] if f"{prefix}_image" in sample else None
                    locality_batch = prepare_multimodal_hf_edit(
                        self.processor, targets=answer, prompts=prompt, images=image, separate_labels=False
                    )
                    self.processor.prepare_labels(
                        locality_batch, [prompt], [answer], [image], new_key="pre_edit_labels"
                    )
                    locality_batch.to(model.device)
                    _, generated_text = generation_pipeline.generate(
                        model=model,
                        tokenizer=self.tokenizer,
                        batch=locality_batch,
                        labels=locality_batch["labels"],
                    )

                    pre_edit_data[f"{prefix}_pre_edit_answer"] = generated_text[0].strip()

                for key in ["background", "text_portability", "multimodal_portability"]:
                    if isinstance(sample[f"{key}_prompt"], str):
                        questions = [sample[f"{key}_prompt"]]
                        answers = [sample[f"{key}_ground_truth"]]
                    else:
                        questions = sample[f"{key}_prompt"][:3]
                        answers = sample[f"{key}_ground_truth"][:3]
                    if "multimodal" in key:
                        image = sample["image"]
                    else:
                        image = None
                    pre_edit_data[f"{key}_pre_edit_answer"] = []
                    for prompt, ground_truth in zip(questions, answers, strict=True):
                        batch = prepare_multimodal_hf_edit(
                            self.processor,
                            targets=ground_truth,
                            prompts=prompt,
                            images=image,
                            separate_labels=False,
                        )
                        self.processor.prepare_labels(batch, [prompt], [ground_truth], [image], new_key="pre_edit_labels")
                        batch.to(model.device)
                        _, generated_text = generation_pipeline.generate(
                            model=model,
                            tokenizer=self.tokenizer,
                            batch=batch,
                            labels=batch["labels"],
                        )
                        pre_edit_data[f"{key}_pre_edit_answer"].append(generated_text[0].strip())

                sample_id_to_pre_edit[sample["cond"]] = pre_edit_data

            with open(self.pre_edit_cache, "w") as f:
                json.dump(sample_id_to_pre_edit, f)

        for sample in tqdm(self._data, desc="Adding pre-edit answers", total=len(self._data)):
            pre_edit_data = sample_id_to_pre_edit[sample["cond"]]
            # update the sample with the new pre-edit answers
            sample.update(pre_edit_data)

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        return self.collater(batch)

    def process_question(self, question):
        if isinstance(question, list):
            return [self.process_question(q) for q in question]
        return self.collater.processor.prepare_question(question)

    def process_answer(self, answer, capitalize: bool = True):
        if isinstance(answer, list):
            return [self.process_answer(a, capitalize=capitalize) for a in answer]
        return self.collater.processor.prepare_answer(answer, capitalize=capitalize)

    def add_portability_data_train(self, edit_item, sample):
        edit_item["text_portability_prompt"] = self.process_question(sample["visual_attribute_qas"][0]["question"])
        edit_item["text_portability_ground_truth"] = self.process_answer(
            sample["visual_attribute_qas"][0]["answer"], capitalize=True
        )
        edit_item["multimodal_portability_prompt"] = self.process_question(
            sample["vg_encyclopedic_qas"][0]["question"]
        )
        edit_item["multimodal_portability_ground_truth"] = self.process_answer(
            sample["vg_encyclopedic_qas"][0]["answer"], capitalize=True
        )
        edit_item["background_prompt"] = self.process_question(sample["encyclopedic_qas"][0]["question"])
        edit_item["background_ground_truth"] = edit_item["multimodal_portability_ground_truth"]

    def add_portability_data_eval(self, edit_item, sample):
        multimodal_portability_qas = sample["vg_encyclopedic_qas"]
        background_qas = sample["encyclopedic_qas"]
        assert len(multimodal_portability_qas) == len(background_qas)
        text_portability_qas = sample["visual_attribute_qas"]
        multimodal_portability_questions = [qa["question"] for qa in multimodal_portability_qas]
        multimodal_portability_answers = [qa["answer"] for qa in multimodal_portability_qas]
        background_questions = [qa["question"] for qa in multimodal_portability_qas]
        background_answers = [qa["answer"] for qa in multimodal_portability_qas]
        text_portability_questions = [qa["question"] for qa in text_portability_qas]
        text_portability_answers = [qa["answer"] for qa in text_portability_qas]

        if self.split != "test":
            multimodal_portability_questions = multimodal_portability_questions[0]
            multimodal_portability_answers = multimodal_portability_answers[0]
            background_questions = background_questions[0]
            background_answers = background_answers[0]
            text_portability_questions = text_portability_questions[0]
            text_portability_answers = text_portability_answers[0]

        edit_item["multimodal_portability_prompt"] = self.process_question(multimodal_portability_questions)
        edit_item["multimodal_portability_ground_truth"] = self.process_answer(
            multimodal_portability_answers, capitalize=True
        )

        assert multimodal_portability_answers == background_answers
        edit_item["background_prompt"] = self.process_question(background_questions)
        edit_item["background_ground_truth"] = self.process_answer(background_answers, capitalize=True)

        edit_item["text_portability_prompt"] = self.process_question(text_portability_questions)
        edit_item["text_portability_ground_truth"] = self.process_answer(text_portability_answers, capitalize=True)

    @property
    def processor(self):
        return self.collater.processor

    @property
    def tokenizer(self):
        return self.collater.processor.tokenizer
