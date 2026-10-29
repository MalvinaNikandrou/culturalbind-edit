import json
from pathlib import Path

import openai
import torch
import transformers
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

from .constants import Constants
from .logging_utils import WandbLogger, logger
from .prediction_sample import PredictionSample, PredictionScore


class Accuracy:
    _metric_name = "accuracy"

    def __init__(
        self,
        run_name: str | None,
        wandb_project: str | None,
        predictions_path: Path | str,
        **kwargs,
    ):
        self.wandb_callback = WandbLogger(project=wandb_project, run_name=run_name)
        self.scored_predictions_path = Constants.model_predictions_path.joinpath(self.wandb_callback.run_name)
        self.scored_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(predictions_path, str):
            predictions_path = Path(predictions_path)
        self.load_prediction_samples(predictions_path)

    def load_prediction_samples(self, predictions_path: str | Path) -> list[PredictionSample]:
        raise NotImplementedError

    def log_results(self, scored_predictions: list[PredictionSample]) -> None:
        combined_results = []
        for prediction in scored_predictions:
            combined_results.append(prediction.to_result())

        self.wandb_callback.log_table(combined_results, table_name="scored_predictions")
        with self.scored_predictions_path.open("w") as fp:
            json.dump(combined_results, fp, indent=2)
        logger.info(f"Results saved to {self.scored_predictions_path}")

    def compute(self) -> float:
        scored_predictions = self.evaluator.generate_scores(self.predictions)
        self.log_results(scored_predictions)
        score = self.compute_average_score(scored_predictions)
        logger.info(f"Average Score: {score * 100: .2f}")
        self.wandb_callback.log_metrics({self._metric_name: score})
        return score

    def compute_average_score(self, scored_predictions: list[PredictionSample]) -> float:
        scores = [pred.score.normalized_score(self.evaluator._max_score) for pred in scored_predictions]
        return sum(scores) / len(scores)


class LLMJudgeEvaluator:
    def __init__(
        self,
        judge_model: str,
        load_in_8bit: bool,
        prompt: str,
        torch_dtype: torch.dtype | None,
        max_new_tokens: int = 256,
        batch_size: int = 16,
    ):
        self.judge_model = judge_model
        self.load_in_8bit = load_in_8bit
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.batch_size = batch_size
        if "gpt-4" in judge_model or "gpt-3.5-turbo" in judge_model:
            openai.api_key = Constants.openai_api_key
            self.pipeline = None
        else:
            self.load_hf_model(judge_model, load_in_8bit, torch_dtype)

    def load_hf_model(self, judge_model: str, load_in_8bit: bool, torch_dtype: torch.dtype | None):
        model_kwargs = {}
        if torch_dtype is not None:
            model_kwargs["dtype"] = torch_dtype

        if self.load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=load_in_8bit)
        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=judge_model,
            model_kwargs=model_kwargs,
            device_map="cuda:0",
        )
        self.pipeline.tokenizer.padding_side = "left"
        if self.pipeline.tokenizer.pad_token_id is None:
            self.pipeline.tokenizer.pad_token = self.pipeline.tokenizer.eos_token

    def __call__(self, sample: PredictionSample) -> PredictionScore:
        return self.generate_score(sample)

    def _get_message(self, sample: PredictionSample) -> list[dict]:
        messages = [
            {"role": "system", "content": "You are a knowledgeable and fair judge."},
            {
                "role": "user",
                "content": self.prompt.format(
                    question=sample.question,
                    references=", ".join(sample.references),
                    prediction=sample.prediction,
                ),
            },
        ]
        return messages

    def _generate_hf_score(self, messages) -> str:
        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=1.0,
            top_k=1,
            tokenizer_encode_kwargs={"enable_thinking": False},
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
        )
        return outputs[0]["generated_text"][-1]["content"]

    def _generate_hf_scores(self, batch_messages) -> list[str]:
        # Process non-exact matches using the pipeline
        outputs = self.pipeline(
            batch_messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=1.0,
            top_k=1,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            tokenizer_encode_kwargs={"enable_thinking": False},
            batch_size=len(batch_messages),
        )
        return [output[0]["generated_text"][-1]["content"] for output in outputs]

    def _generate_openai_score(self, messages) -> str:
        client = openai.OpenAI(api_key=Constants.openai_api_key)
        prediction = client.chat.completions.create(
            model=Constants.open_ai_model,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        )
        return prediction.choices[0].message.content

    def _generate_openai_scores(self, batch_messages) -> list[str]:
        client = openai.OpenAI(api_key=Constants.openai_api_key)
        predictions = []
        for messages in tqdm(batch_messages, desc="Generating OpenAI scores"):
            prediction = client.chat.completions.create(
                model=Constants.open_ai_model,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            predictions.append(prediction.choices[0].message.content)
        return predictions

    def _get_score(self, generated_text: str) -> PredictionScore:
        generated_text = generated_text.lower().strip()
        try:
            parsed_score = PredictionScore.model_validate_json(generated_text)
        except ValueError:
            parsed_score = PredictionScore(
                score=0.0, explanation=f"Could not parse the model output.\n{generated_text}"
            )
        return parsed_score


class LLMJudgeAccuracy(Accuracy):
    def __init__(
        self,
        run_name: str | None,
        wandb_project: str | None,
        predictions_path: Path | str,
        judge_model: str,
        prompt: str,
        evaluator: LLMJudgeEvaluator,
        load_in_8bit: bool = False,
        max_new_tokens: int = 256,
        batch_size: int = 8,
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__(
            run_name=run_name,
            wandb_project=wandb_project,
            predictions_path=predictions_path,
        )
        self.evaluator = evaluator(
            judge_model=judge_model,
            load_in_8bit=load_in_8bit,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
        )
