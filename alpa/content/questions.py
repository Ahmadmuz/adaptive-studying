"""Question generation, metadata schema, validation, and deterministic grading
(Phase I). Generated items carry full structured metadata and are validated
before they may enter the bank. Malformed/ambiguous items are rejected with
reasons, never silently admitted.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

ITEM_TYPES = {"multiple_choice", "numerical", "short_answer", "structured"}
GRADING_METHODS = {"exact_choice", "numeric_tolerance", "normalized_match",
                   "rubric_external"}


@dataclass
class QuestionSpec:
    question: str
    answer: str
    explanation: str
    concept: str
    difficulty: float
    item_type: str
    prerequisites: list[str] = field(default_factory=list)
    grading_method: str = "normalized_match"
    choices: list[str] | None = None
    correct_index: int | None = None
    tolerance: float | None = None
    rubric: str | None = None
    provenance: str = "llm:template:v1"

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.question or len(self.question.strip()) < 8:
            errs.append("question too short")
        if not self.concept or not self.concept.strip():
            errs.append("missing concept")
        if not (0.0 <= self.difficulty <= 1.0):
            errs.append("difficulty outside [0,1]")
        if self.item_type not in ITEM_TYPES:
            errs.append(f"unknown item_type {self.item_type!r}")
        if self.grading_method not in GRADING_METHODS:
            errs.append(f"unknown grading_method {self.grading_method!r}")
        if self.item_type == "multiple_choice":
            errs += self._validate_mc()
        if self.item_type == "numerical":
            errs += self._validate_numerical()
        if self.item_type == "short_answer" and not self.answer.strip():
            errs.append("short_answer requires non-empty answer")
        if self.item_type == "structured" and not self.rubric:
            errs.append("structured items require a rubric")
        if self.grading_method == "rubric_external" and not self.rubric:
            errs.append("rubric_external grading requires a rubric")
        return errs

    def _validate_mc(self) -> list[str]:
        errs = []
        if not self.choices or len(self.choices) < 3:
            errs.append("multiple_choice needs >=3 choices")
            return errs
        if any(not c.strip() for c in self.choices):
            errs.append("empty choice")
        norm = [re.sub(r"\s+", " ", c.strip().lower()) for c in self.choices]
        if len(set(norm)) != len(norm):
            errs.append("duplicate/ambiguous choices")
        if self.correct_index is None or not (0 <= self.correct_index < len(self.choices)):
            errs.append("correct_index out of range")
        if self.correct_index is not None and 0 <= self.correct_index < len(self.choices):
            if self.answer and self.answer.strip() and \
                    self.answer.strip().lower() != self.choices[self.correct_index].strip().lower():
                errs.append("answer field inconsistent with correct_index")
        return errs

    def _validate_numerical(self) -> list[str]:
        errs = []
        try:
            float(self.answer)
        except (TypeError, ValueError):
            errs.append("numerical answer not parseable as float")
        if self.tolerance is None or self.tolerance < 0:
            errs.append("numerical items need tolerance >= 0")
        if self.grading_method != "numeric_tolerance":
            errs.append("numerical items must grade by numeric_tolerance")
        return errs

    def grade(self, response: str) -> tuple[bool, str]:
        """Deterministic grading for closed forms. Returns (correct, method) or
        (None, 'external') when the item needs a rubric/LLM grader — which stays
        OUTSIDE the core student model."""
        if self.item_type == "multiple_choice":
            try:
                idx = int(str(response).strip())
            except ValueError:
                return False, "exact_choice"
            return idx == self.correct_index, "exact_choice"
        if self.item_type == "numerical":
            try:
                val = float(str(response).strip())
            except ValueError:
                return False, "numeric_tolerance"
            return abs(val - float(self.answer)) <= (self.tolerance or 0), "numeric_tolerance"
        if self.item_type == "short_answer":
            a = re.sub(r"\s+", " ", response.strip().lower())
            b = re.sub(r"\s+", " ", self.answer.strip().lower())
            return a == b, "normalized_match"
        return None, "external"


def validate_batch(specs: list[QuestionSpec]) -> tuple[list[QuestionSpec], list[dict]]:
    ok, rejected = [], []
    for s in specs:
        errs = s.validate()
        if errs:
            rejected.append({"question": s.question[:80], "errors": errs})
        else:
            ok.append(s)
    return ok, rejected
