import ast
import json
from pathlib import Path

import torch

from .evaluator import LLMJudgeAccuracy
from .lave_accuracy import LAVE, ClassificationSample, normalize_answer

RECOGNITION_PROMPT = """You are given a question, a set of gold-standard reference answers written by experts, and a candidate answer. You must rate the accuracy of the candidate answer considering both the question and the reference answers.

## Scoring Rubric (0-2)

Use a scale of 0-2: 

0 - Incorrect: The candidate answer contradicts the references, gives an irrelevant answer, or only restates information from the question.

1 - Partially correct: The candidate answer is consistent with the reference answer BUT it is incomplete or lacks the specificity of the reference answers. The answer must not be inferred from the question alone or contradict the references.

2 - Correct: The candidate answer is fully consistent with the reference answers. It provides an equivalent term or correctly identifies the object at the appropriate level of specificity according to the question.

## EXAMPLES

### Example 1
Question: What traditional clothing is the woman wearing?
Reference answers: hanbok
Candidate answer: kimono
Output: {{
    "explanation": "Hanbok is Korean traditional clothing, while kimono is Japanese.",
    "score": 0
}}

### Example 2
Question: What is this dish?
Reference answers: jiaozi, dumplings, gyoza
Candidate answer: gyōza
Output: {{
    "explanation": "'Gyōza' is listed among the reference answers.",
    "score": 2
}}

### Example 3
Question: What is the rice dish in the image called?
Reference answers: pilaf
Candidate answer: fried rice
Output: {{
    "explanation": "'Pilaf' and 'fried rice' are different dishes.",
    "score": 0
}}

### Example 4
Question: What is this flower called?
Reference answers: chamomile
Candidate answer: daisy
Output: {{
    "explanation": "While chamomile belongs to the daisy family, the candidate is less specific than the reference.",
    "score": 1
}}

### Example 5
Question: What is this instrument called?
Reference answers: hulusi, cucurbit flute
Candidate answer: flute
Output: {{
    "explanation": "'Flute' is the broader category, the candidate answer lacks the specificity of the references.",
    "score": 1
}}

### Example 6
Question: What type of fruit is in the picture?
Reference answers: apple
Candidate answer: fruit
Output: {{
    "explanation": "The candidate answer only restates information from the question.",
    "score": 0
}}

### Example 7
Question: What type of plant is this?
Reference answers: maize
Candidate answer: corn
Output: {{
    "explanation": "'Maize' and 'corn' refer to the same plant.",
    "score": 2
}}

### Example 8
Question: What food is shown?
Reference answers: sushi
Candidate answer: Japanese food
Output: {{
    "explanation": "The candidate answer correctly identifies the cuisine but is less specific than 'sushi'.",
    "score": 1
}}

### Example 9
Question: What is this formation?
Reference answers: terrace
Candidate answer: terraced field
Output: {{
    "explanation": "'Terraced field' appropriately matches the reference in this context.",
    "score": 2
}}

## Output Format

Respond only with the following JSON structure and nothing else:

{{
  "explanation": "<short explanation>",
  "score": <0 | 1 | 2>
}}

## Example to Score

Question: {question}
Reference answers: {references}
Candidate answer: {prediction}
Output:
"""


class RecognitionAccuracy(LLMJudgeAccuracy):
    _metric_name = "lave_recognition_accuracy"

    def __init__(
        self,
        model_type: str,
        run_name: str | None,
        wandb_project: str | None,
        predictions_path: Path | str,
        judge_model: str,
        load_in_8bit: bool = False,
        prompt: str = RECOGNITION_PROMPT,
        max_new_tokens: int = 256,
        batch_size: int = 8,
        torch_dtype: torch.dtype | None = None,
    ):
        if model_type == "pre-edit":
            self.get_question = self.get_pre_edit_question
            self.get_prediction = self.get_pre_edit_prediction
            self.get_label = self.get_pre_edit_label
            self._add_metadata = self._add_pre_edit_metadata
        elif model_type == "post-edit":
            self.get_question = self.get_post_edit_question
            self.get_prediction = self.get_post_edit_prediction
            self.get_label = self.get_post_edit_label
            self._add_metadata = lambda result: result
        super().__init__(
            run_name=run_name,
            wandb_project=wandb_project,
            predictions_path=predictions_path,
            judge_model=judge_model,
            evaluator=LAVE,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch_dtype,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )

    def prepare_ground_truths(self, ground_truths):
        if isinstance(ground_truths, str):
            try:
                ground_truths = ast.literal_eval(ground_truths)
            except (ValueError, SyntaxError):
                return normalize_answer(ground_truths)
        return [normalize_answer(gt) for gt in ground_truths]

    def get_pre_edit_question(self, result) -> str:
        return result["question"]

    def get_pre_edit_prediction(self, result) -> str:
        prediction = result["prediction"]
        return normalize_answer(prediction)

    def get_pre_edit_label(self, result) -> list[str]:
        return self.prepare_ground_truths(result["references"])

    def get_post_edit_question(self, result) -> str:
        return result["input"].split("user\n")[-1].split("\nAnswer")[0].strip()

    def get_post_edit_prediction(self, result) -> str:
        return normalize_answer(result["prediction"].strip())

    def get_post_edit_label(self, result) -> list[str]:
        return self.prepare_ground_truths([result["label"].strip()])

    def load_prediction_samples(self, predictions_path: str | Path) -> list[ClassificationSample]:
        if isinstance(predictions_path, str):
            predictions_path = Path(predictions_path)
        with predictions_path.open("r") as fp:
            results = json.load(fp)
        prediction_samples = []
        self.sample_to_metadata = {}
        for result in results:
            question = self.get_question(result)
            prediction = self.get_prediction(result)
            label = self.get_label(result)
            sample_id = result.get("sample_id", str(result.get("id", None)))
            metadata = {
                "concept": result.get("concept", ""),
                "image_id": result.get("image_id", ""),
                "chapter": result.get("chapter", ""),
            }
            prediction_samples.append(
                ClassificationSample(
                    question=question,
                    references=label,
                    prediction=prediction,
                    sample_id=sample_id,
                    metadata=metadata,
                )
            )
            self.sample_to_metadata[sample_id] = result
        self.predictions = prediction_samples

    def _add_pre_edit_metadata(self, result):
        metadata = self.sample_to_metadata[result["sample_id"]]
        result["chapter"] = metadata["chapter"]
        result["image_id"] = metadata["image_id"]
