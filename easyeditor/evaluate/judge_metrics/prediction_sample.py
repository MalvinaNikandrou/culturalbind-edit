from pydantic import BaseModel, field_validator


class PredictionScore(BaseModel):
    score: float
    explanation: str = ""

    def normalized_score(self, max_score: float) -> float:
        """Normalize the score to be between 0 and 1."""
        if self.score < 0:
            return 0.0
        elif self.score > max_score:
            return 1.0
        return self.score / max_score


class PredictionSample(BaseModel):
    sample_id: str
    question: str
    prediction: str
    references: list[str]
    metadata: dict = {}
    score: PredictionScore | None = None
    max_score: float = 1.0

    @field_validator("references", mode="before")
    @classmethod
    def validate_references(cls, references: list[str] | str) -> list[str]:
        if isinstance(references, str):
            references = [references]
        return references

    @classmethod
    def from_result(cls, result: dict) -> "PredictionSample":
        score = result.pop("score", None)
        if score is not None:
            score = PredictionScore(
                score=score,
                explanation=result.pop("score_explanation", ""),
            )
        metadata = result.pop("metadata", {})
        metadata.update({k: v for k, v in result.items() if k not in cls.model_fields})
        return PredictionSample(
            **{k: v for k, v in result.items() if k in cls.model_fields},
            metadata=metadata,
            score=score,
        )

    def to_result(self) -> dict:
        result = self.model_dump(exclude={"score", "max_score", "metadata"})
        if isinstance(result["references"], list) and len(result["references"]) == 1:
            result["references"] = result["references"][0]
        if self.score is not None:
            result["score"] = self.score.normalized_score(self.max_score)
            result["score_explanation"] = self.score.explanation
        # Flatten metadata
        for key, value in self.metadata.items():
            result[key] = value
        return result
