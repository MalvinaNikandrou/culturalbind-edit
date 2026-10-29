import logging
import os
from pathlib import Path

from .judge_metrics import BackgroundAccuracy, RecognitionAccuracy

LOG = logging.getLogger(__name__)


class PostEvalMetric:
    """Abstract class for metrics."""

    def __init__(self, metric: str, predictions_file: str, output_dir: str):
        self.metric = metric
        self.predictions_file = predictions_file
        self.output_file = os.path.join(output_dir, f"{self.metric}_scored.json")
        if os.path.exists(self.output_file):
            LOG.error(f"Output file {self.output_file} already exists.")
            # raise FileExistsError

    def compute(self, log_to_wandb: bool = True) -> float:
        LOG.info(f"Scoring post-edit {self.metric}")

        lave_accuracy_recognition = self.metric_func(
            model_type="post-edit",
            run_name=self.metric,
            wandb_project="easyeditor-eval",
            judge_model="Qwen/Qwen3-32B",
            load_in_8bit=True,
            predictions_path=Path(self.predictions_file),
        )
        accuracy = lave_accuracy_recognition.compute()
        LOG.info(f"Accuracy: {100*accuracy:.2f}%")
        del lave_accuracy_recognition
        return accuracy


class PostEvalRecognitionMetric(PostEvalMetric):
    metric_func = RecognitionAccuracy


class PostEvalBackgroundMetric(PostEvalMetric):
    metric_func = BackgroundAccuracy
