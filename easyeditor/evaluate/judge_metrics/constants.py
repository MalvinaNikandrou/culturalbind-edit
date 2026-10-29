import os
from pathlib import Path


class Constants:
    openai_api_key: str | None = os.environ.get("OPENAI_API_KEY")
    open_ai_model: str = "gpt-5.2-2025-12-11"
    merged_open_ended_dataset = "Malvinan/Merged_OpenEnded"
    model_predictions_path: Path = Path("data/model_predictions")
