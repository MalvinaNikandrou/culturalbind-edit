import re
from collections import Counter
from itertools import islice

from tqdm import tqdm

from .prediction_sample import PredictionSample, PredictionScore
from .evaluator import LLMJudgeEvaluator


def normalize_answer(answer: str) -> str:
    """Normalize a VQA answer."""
    if answer is None or not isinstance(answer, str):
        answer = "no prediction"
    answer = answer.replace("\n", " ")
    answer = answer.replace("\t", " ")
    answer = answer.lower().strip()
    return answer


class ClassificationSample(PredictionSample):

    max_score: float = 2.0

    @property
    def normalized_references(self) -> list[str]:
        return [normalize_answer(ref) for ref in self.references]

    @property
    def normalized_prediction(self) -> str:
        return normalize_answer(self.prediction)

    def is_exact_match(self) -> bool:
        refs = Counter(self.normalized_references)
        most_common_ref = refs.most_common(1)[0][0]
        return self.normalized_prediction == most_common_ref

    def matches_concept(self) -> bool:
        concept = normalize_answer(self.metadata.get("concept", ""))
        return self.normalized_prediction == concept


class LAVE(LLMJudgeEvaluator):
    _max_score: float = 2.0

    def generate_score(self, sample: PredictionSample) -> PredictionScore:
        if sample.is_exact_match():
            return PredictionScore(score=self._max_score, explanation="Exact match")
        messages = self._get_message(sample)
        if self.pipeline is None:
            generated_text = self._generate_openai_score(messages)
        else:
            generated_text = self._generate_hf_score(messages)
        return self._get_score(generated_text)

    def generate_scores(self, samples: list[PredictionSample]) -> list[ClassificationSample]:
        """
        Generate scores for a list of ClassificationSample objects in batch mode.

        Args:
            samples (list[PredictionSample]): List of PredictionSample objects.
            batch_size (int): Number of samples to process in each batch.

        Returns:
            list[ClassificationSample]: List of scored samples.
        """
        exact_matches = [s.is_exact_match() for s in samples]
        if all(exact_matches):
            scores = [PredictionScore(score=self._max_score, explanation="Exact match")] * len(samples)
            scored_batch = [
                ClassificationSample(
                    **sample.model_copy(update={"score": score, "max_score": self._max_score}).model_dump()
                )
                for sample, score in zip(samples, scores, strict=True)
            ]
            return scored_batch

        # Helper function to create batches
        def batch_iterable(iterable, size):
            it = iter(iterable)
            while batch := list(islice(it, size)):
                yield batch

        # Separate exact matches and non-exact matches
        num_exact_matches = sum(exact_matches)
        non_exact_matches = [
            sample for sample, is_exact_match in zip(samples, exact_matches, strict=True) if not is_exact_match
        ]
        messages = [self._get_message(sample) for sample in non_exact_matches]
        if self.pipeline is None:
            raw_scores = self._generate_openai_scores(messages)
            scores = [self._get_score(text) for text in raw_scores]
        else:
            num_batches = (len(non_exact_matches) + self.batch_size - 1) // self.batch_size
            scores = []
            pbar = tqdm(
                batch_iterable(non_exact_matches, self.batch_size),
                total=num_batches,
                desc="Generating scores",
            )
            for batch in pbar:
                batch_messages = [self._get_message(sample) for sample in batch]
                generated_texts = self._generate_hf_scores(batch_messages)
                scores.extend([self._get_score(text) for text in generated_texts])

                # Calculate and display running accuracy
                correct = sum(s.normalized_score(self._max_score) for s in scores) + num_exact_matches
                total = len(scores) + num_exact_matches
                running_accuracy = correct / total
                pbar.set_postfix({"accuracy": f"{running_accuracy:.4f}"})

        # Combine exact matches and model scores
        final_scores = []
        for is_exact in exact_matches:
            if is_exact:
                prediction_score = PredictionScore(score=self._max_score, explanation="Exact match")
            else:
                prediction_score = scores.pop(0)

            final_scores.append(prediction_score)
        scored_batch = [
            ClassificationSample(
                **sample.model_copy(update={"score": score, "max_score": self._max_score}).model_dump()
            )
            for sample, score in zip(samples, final_scores, strict=True)
        ]
        return scored_batch


class PostEditBackgroundLAVE(LAVE):
    _min_score: float = 0.0

    def generate_score(self, sample: PredictionSample) -> PredictionScore:
        if sample.is_exact_match():
            return PredictionScore(score=self._max_score, explanation="Exact match")
        if sample.matches_concept():
            return PredictionScore(score=self._min_score, explanation="Concept match")
        messages = self._get_message(sample)
        if self.pipeline is None:
            generated_text = self._generate_openai_score(messages)
        else:
            generated_text = self._generate_hf_score(messages)
        return self._get_score(generated_text)

    def generate_scores(self, samples: list[PredictionSample]) -> list[ClassificationSample]:
        """
        Generate scores for a list of ClassificationSample objects in batch mode.

        Args:
            samples (list[PredictionSample]): List of PredictionSample objects.
            batch_size (int): Number of samples to process in each batch.

        Returns:
            list[ClassificationSample]: List of scored samples.
        """
        string_matches = []
        # Separate exact matches and non-exact matches
        non_string_matches = []
        for sample in samples:
            if sample.is_exact_match():
                string_matches.append(True)
            elif sample.matches_concept():
                string_matches.append(True)
            else:
                string_matches.append(False)
                non_string_matches.append(sample)

        if all(string_matches):
            scored_batch = []
            for sample in samples:
                if sample.is_exact_match():
                    score = PredictionScore(score=self._max_score, explanation="Exact match")
                else:
                    score = PredictionScore(score=self._min_score, explanation="Concept match")
                scored_batch.append(
                    ClassificationSample(
                        **sample.model_copy(update={"score": score, "max_score": self._max_score}).model_dump()
                    )
                )
            return scored_batch

        # Helper function to create batches
        def batch_iterable(iterable, size):
            it = iter(iterable)
            while batch := list(islice(it, size)):
                yield batch

        num_string_matches = sum(string_matches)
        messages = [self._get_message(sample) for sample in non_string_matches]
        if self.pipeline is None:
            raw_scores = self._generate_openai_scores(messages)
            scores = [self._get_score(text) for text in raw_scores]
        else:
            num_batches = (len(non_string_matches) + self.batch_size - 1) // self.batch_size
            scores = []
            pbar = tqdm(
                batch_iterable(non_string_matches, self.batch_size),
                total=num_batches,
                desc="Generating scores",
            )
            for batch in pbar:
                batch_messages = [self._get_message(sample) for sample in batch]
                generated_texts = self._generate_hf_scores(batch_messages)
                scores.extend([self._get_score(text) for text in generated_texts])

                # Calculate and display running accuracy
                correct = sum(s.normalized_score(self._max_score) for s in scores) + num_string_matches
                total = len(scores) + num_string_matches
                running_accuracy = correct / total
                pbar.set_postfix({"accuracy": f"{running_accuracy:.4f}"})

        # Combine string matches and model scores
        final_scores = []
        for sample, is_string in zip(samples, string_matches, strict=True):
            if is_string:
                if sample.is_exact_match():
                    prediction_score = PredictionScore(score=self._max_score, explanation="Exact match")
                else:
                    prediction_score = PredictionScore(score=self._min_score, explanation="String match")
            else:
                prediction_score = scores.pop(0)

            final_scores.append(prediction_score)
        scored_batch = [
            ClassificationSample(
                **sample.model_copy(update={"score": score, "max_score": self._max_score}).model_dump()
            )
            for sample, score in zip(samples, final_scores, strict=True)
        ]
        return scored_batch
