import json
from pathlib import Path

import torch
from datasets import load_dataset

from .constants import Constants
from .evaluator import LLMJudgeAccuracy
from .lave_accuracy import LAVE, ClassificationSample, PostEditBackgroundLAVE

BACKGROUND_PROMPT = """You are given a factual question, a ground-truth reference answer written by experts, and a candidate answer. You must rate the accuracy of the candidate answer considering both the question and the reference answer.

## Scoring Rubric (0-2)

Use a scale of 0-2: 

0 - Incorrect: The candidate answer is incoherent or contradicts the reference, is an unrelated fact, or only restates information from the question.

1 - Partially correct: The candidate answer is consistent with the reference answer BUT it is incomplete, under-specific, or only provides one instance of a broader category. The answer must add meaningful information beyond what can be directly inferred from the question and it must not contradict the reference.

2 - Correct: The candidate answer is fully consistent with the reference answer. It expresses the same fact or an equivalent paraphrase. No essential information is missing.

## EXAMPLES

### Example 1
Question: What is the country of origin for Almdudler?
Reference answers: Austria
Candidate answer: germany
Output: {{
  "explanation": "Clear contradiction. The correct country is Austria, not Germany.",
  "score": 0
}}

### Example 2
Question: What type of meat is typically used in chanakhi?
Reference answers: Lamb Stew
Candidate answer: lamb
Output: {{
  "explanation": "In the context of this question, 'lamb' is equivalent to 'lamb stew'",
  "score": 2
}}

### Example 3
Question: What are the types of events commonly held at the Zappeion?
Reference answers: Official and private ceremonies
Candidate answer: concert
Output: {{
  "explanation": "Incorrect because concerts are not official or private ceremonies.",
  "score": 0
}}

### Example 4
Question: Which global regions have been influenced by the spread of chapatis through Indian immigrants?
Reference answers: Central Asia, Southeast Asia, East Africa, and the Caribbean
Candidate answer: africa
Output: {{
  "explanation": "'Africa' overlaps with 'East Africa' but is under-specific and incomplete.",
  "score": 1
}}

### Example 5
Question: In which country did the mbira originate? 
Reference answers: Shona
Candidate answer: zimbabwe
Output: {{
  "explanation": "The question asks for a country, and Zimbabwe is the country where the Shona people (referenced in the answer) primarily live.",
  "score": 2
}}

### Example 6
Question: At which type of events is rendang traditionally served?
Reference answers: ceremonial occasions
Candidate answer: wedding
Output: {{
  "explanation": "Incomplete candidate answer because 'wedding' is only one type of ceremonial occasions.",
  "score": 1
}}

### Example 7
Question: What kind of musical ensembles in China commonly feature the sanxian?
Reference answers: Jiangnan sizhu
Candidate answer: Traditional Chinese ensembles
Output: {{
  "explanation": "'Traditional Chinese ensembles' is a broader category that includes 'Jiangnan sizhu', but it is not as specific. ",
  "score": 1
}}

### Example 8
Question: What specific type of bamboo is commonly used to make a suling?
Reference answers: Tamiang
Candidate answer: Bamboo
Output: {{
  "explanation": "The candidate answer provides no meaningful information beyond what is already in by the question.",
  "score": 0
}}

### Example 9
Question: What type of flour is used to make chapati?
Reference answers: Whole-wheat flour (atta)
Candidate answer: Wheat
Output: {{
  "explanation": "The candidate answer is correct because 'Wheat' is equivalent to 'Whole-wheat flour (atta)' in this context.",
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


class BackgroundAccuracy(LLMJudgeAccuracy):
    _metric_name = "lave_background_accuracy"

    def __init__(
        self,
        model_type: str,
        run_name: str | None,
        wandb_project: str | None,
        predictions_path: Path | str,
        judge_model: str,
        load_in_8bit: bool = False,
        torch_dtype: torch.dtype | None = None,
        prompt: str = BACKGROUND_PROMPT,
        max_new_tokens: int = 64,
        batch_size: int = 8,
    ):
        if model_type == "pre-edit":
            self.get_question = self.get_pre_edit_question
            self.get_prediction = self.get_pre_edit_prediction
            self.get_label = self.get_pre_edit_label
            self.load_prediction_samples = self.load_pre_edit_prediction_samples
        elif model_type == "post-edit":
            self.get_question = self.get_post_edit_question
            self.get_prediction = self.get_post_edit_prediction
            self.get_label = self.get_post_edit_label
            self.load_prediction_samples = self.load_post_edit_prediction_samples
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        if model_type == "post-edit":
            evaluator = PostEditBackgroundLAVE
        else:
            evaluator = LAVE
        super().__init__(
            run_name=run_name,
            wandb_project=wandb_project,
            predictions_path=predictions_path,
            judge_model=judge_model,
            evaluator=evaluator,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch_dtype,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )

    def get_pre_edit_question(self, result) -> str:
        return result["question"]

    def get_pre_edit_prediction(self, result) -> str:
        result["prediction"] = result["prediction"].strip()
        return result["prediction"]

    def get_post_edit_question(self, result) -> str:
        question = result["input"].split("user\n")[-1].split("\n\n\n\n")[-1].split("\nAnswer")[0]
        # IKE inputs are "<in-context examples>New Fact: <edit>\nPrompt: <question>"
        question = question.split("Prompt: ")[-1]
        return question.strip()

    def get_post_edit_prediction(self, result) -> str:
        return result["prediction"].strip()

    def get_pre_edit_label(self, result) -> str:
        if isinstance(result["references"], str):
            return result["references"].strip()
        return result["references"][0]

    def get_post_edit_label(self, result) -> str:
        return result["label"].strip()

    def load_pre_edit_prediction_samples(self, predictions_path: Path) -> list[ClassificationSample]:
        with predictions_path.open("r") as fp:
            results = json.load(fp)

        dataset = load_dataset(Constants.merged_open_ended_dataset, split="test")
        concepts = set(item["answer"].lower() for item in dataset)

        predictions = []
        for result in results:
            concept = result.get("concept", "").lower()
            if concept not in concepts:
                continue
            question = self.get_question(result)
            prediction = self.get_prediction(result)
            label = self.get_label(result)
            sample_id = str(result.get("sample_id", str(result.get("id", None))))
            predictions.append(
                ClassificationSample(
                    question=question,
                    references=label,
                    prediction=prediction,
                    sample_id=sample_id,
                    metadata={"concept": result.get("concept", "")},
                )
            )
        self.predictions = predictions

    def load_post_edit_prediction_samples(self, predictions_path: Path) -> list[ClassificationSample]:
        with predictions_path.open("r") as fp:
            results = json.load(fp)

        predictions = []
        for result in results:
            question = self.get_question(result)
            prediction = self.get_prediction(result)
            label = self.get_label(result)
            sample_id = str(result.get("sample_id", str(result.get("id", None))))
            concept = sample_id.split("||")[0].split(">>")[-1].strip()

            predictions.append(
                ClassificationSample(
                    question=question,
                    references=label,
                    prediction=prediction,
                    sample_id=sample_id,
                    metadata={"concept": concept},
                )
            )
        self.predictions = predictions

    def reset_predictions(self, predictions_path: str | Path) -> None:
        if isinstance(predictions_path, str):
            predictions_path = Path(predictions_path)
        self.load_prediction_samples(predictions_path)