"""LLM provider abstraction (Phase H).

Contract: the adaptive engine decides WHAT and WHY (GenerationSpec); a provider
decides HOW to express it. Providers are interchangeable; the ML engine imports
nothing provider-specific. A deterministic TemplateProvider implements the full
interface offline; HTTPChatProvider implements it against any OpenAI-compatible
chat-completions endpoint (OpenAI, Azure, vLLM, llama.cpp, Ollama) via base_url.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ProviderOutputError(ValueError):
    """Provider returned content that could not be parsed into the contract."""


@dataclass(frozen=True)
class GenerationSpec:
    """The structured instruction the ML engine hands to content generation.
    The LLM never selects the intervention — it receives it."""
    concept: str
    mastery: float
    uncertainty: float
    difficulty: float
    intervention: str
    prerequisites: tuple[str, ...] = ()


class LLMProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def generate_explanation(self, spec: GenerationSpec) -> str: ...

    @abstractmethod
    def generate_example(self, spec: GenerationSpec) -> str: ...

    @abstractmethod
    def generate_hint(self, spec: GenerationSpec) -> str: ...

    @abstractmethod
    def generate_question(self, spec: GenerationSpec) -> dict:
        """Returns a QuestionSpec-shaped dict (see alpa.content.questions)."""

    @abstractmethod
    def generate_flashcard(self, spec: GenerationSpec) -> dict:
        """Returns {front: str, back: str, concept: str}."""

    @abstractmethod
    def generate_summary(self, spec: GenerationSpec) -> str:
        """Short summary of the concept for quick review."""

    @abstractmethod
    def generate_mini_lesson(self, spec: GenerationSpec) -> str:
        """A mini-lesson / micro-teaching content."""

    @abstractmethod
    def extract_concepts(self, text: str) -> list[dict]:
        """Concept PROPOSALS: [{name, description, confidence}],
        provenance-stamped by the caller."""

    @abstractmethod
    def extract_relationships(self, concepts: list[str], text: str) -> list[dict]:
        """Edge PROPOSALS: [{src, dst, kind, confidence}]."""


class TemplateProvider(LLMProvider):
    """Deterministic offline provider. Content is structurally valid but
    pedagogically minimal — it exists to exercise and test the full
    generate -> validate -> bank pipeline without external dependencies."""
    name = "template:v1"

    def generate_explanation(self, spec: GenerationSpec) -> str:
        prereq = (f" Build on: {', '.join(spec.prerequisites)}."
                  if spec.prerequisites else "")
        return (f"Explanation of '{spec.concept}' at difficulty "
                f"{spec.difficulty:.2f} (mastery {spec.mastery:.2f}).{prereq}")

    def generate_example(self, spec: GenerationSpec) -> str:
        return f"Worked example illustrating '{spec.concept}'."

    def generate_hint(self, spec: GenerationSpec) -> str:
        return f"Hint: revisit the definition of '{spec.concept}'."

    def generate_question(self, spec: GenerationSpec) -> dict:
        b = round((spec.difficulty - 0.5) * 3.0, 3)   # [0,1] -> logit prior
        return {
            "question": f"Which statement best characterizes {spec.concept}?",
            "choices": [
                f"A correct characterization of {spec.concept}",
                f"A common confusion involving {spec.concept}",
                "An unrelated statement",
                "An oversimplified statement",
            ],
            "correct_index": 0,
            "answer": "A correct characterization of " + spec.concept,
            "explanation": self.generate_explanation(spec),
            "concept": spec.concept,
            "difficulty": spec.difficulty,
            "item_type": "multiple_choice",
            "prerequisites": list(spec.prerequisites),
            "grading_method": "exact_choice",
        }

    def extract_concepts(self, text: str) -> list[dict]:
        seen, out = set(), []
        for line in text.splitlines():
            line = line.strip().lstrip("-*# ").strip()
            if 3 <= len(line) <= 60 and line[:1].isupper() and line not in seen:
                seen.add(line)
                out.append({"name": line, "description": "", "confidence": 0.5})
            if len(out) >= 20:
                break
        return out

    def extract_relationships(self, concepts: list[str], text: str) -> list[dict]:
        return []  # template provider proposes no edges: no fabricated structure

    def generate_flashcard(self, spec: GenerationSpec) -> dict:
        return {
            "front": f"What is {spec.concept}?",
            "back": f"{spec.concept}: key definition and properties.",
            "concept": spec.concept,
        }

    def generate_summary(self, spec: GenerationSpec) -> str:
        prereq = (f" Builds on: {', '.join(spec.prerequisites)}."
                  if spec.prerequisites else "")
        return (f"Summary of '{spec.concept}': core ideas at difficulty "
                f"{spec.difficulty:.2f}.{prereq}")

    def generate_mini_lesson(self, spec: GenerationSpec) -> str:
        prereq = (f" Prerequisites: {', '.join(spec.prerequisites)}."
                  if spec.prerequisites else "")
        return (f"Mini-lesson on '{spec.concept}': structured overview covering "
                f"key points at level {spec.difficulty:.2f}.{prereq}")


class ProviderRegistry:
    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> LLMProvider:
        if name not in self._providers:
            raise KeyError(f"unknown LLM provider '{name}'; "
                           f"registered: {sorted(self._providers)}")
        return self._providers[name]

    def names(self) -> list[str]:
        return sorted(self._providers)


REGISTRY = ProviderRegistry()
REGISTRY.register(TemplateProvider())


def register_from_env(env: dict | None = None) -> str | None:
    """Register an HTTPChatProvider from environment variables, if configured.

    Any OpenAI-compatible chat-completions endpoint works here — including
    free-tier options (Groq, Google AI Studio's OpenAI-compat endpoint,
    OpenRouter free models) and zero-cost local ones (Ollama, llama.cpp
    server). Reads:
        ALPA_LLM_PROVIDER  name to register under (default 'live:v1')
        ALPA_LLM_BASE_URL  e.g. https://api.groq.com/openai/v1
        ALPA_LLM_API_KEY   bearer token (Ollama/llama.cpp: any placeholder)
        ALPA_LLM_MODEL     model id, e.g. llama-3.1-8b-instant
    Returns the registered provider name, or None if base_url/model absent
    (silently a no-op — the deterministic TemplateProvider remains default).
    """
    import os
    env = env if env is not None else os.environ
    base_url, model = env.get("ALPA_LLM_BASE_URL"), env.get("ALPA_LLM_MODEL")
    if not base_url or not model:
        return None
    name = env.get("ALPA_LLM_PROVIDER", "live:v1")
    provider = HTTPChatProvider(
        base_url=base_url, api_key=env.get("ALPA_LLM_API_KEY", ""),
        model=model, name=name)
    REGISTRY.register(provider)
    return name


# ------------------------------------------------------------------ JSON ----

def parse_json_loose(raw: str):
    """Parse LLM JSON output defensively: bare JSON, fenced blocks, or JSON
    embedded in prose. Raises ProviderOutputError when unrecoverable."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        lo, hi = raw.find(opener), raw.rfind(closer)
        if lo != -1 and hi > lo:
            try:
                return json.loads(raw[lo:hi + 1])
            except json.JSONDecodeError:
                continue
    raise ProviderOutputError(f"unparseable provider output: {raw[:120]!r}")


# ------------------------------------------------------------------ HTTP ----

QUESTION_SCHEMA_PROMPT = """You generate ONE assessment item. Reply with a single
JSON object and nothing else, matching exactly:
{"question": string, "item_type": "multiple_choice"|"numerical"|"short_answer",
 "choices": [3-6 strings, multiple_choice only],
 "correct_index": integer, multiple_choice only,
 "answer": string (the correct answer; numeric answers as digits),
 "explanation": string,
 "difficulty": float in [0,1],
 "prerequisites": [strings]}
Rules: no ambiguous or duplicate choices; one unambiguously correct answer;
difficulty reflects the instruction from the adaptive engine."""

CONCEPTS_PROMPT = """Extract concept proposals from the learning material. Reply
with a single JSON object: {"concepts": [{"name": string, "description": string,
"confidence": float in [0,1]}]}. Propose at most 12 concepts; names must be
short noun phrases actually present in the material."""

RELATIONSHIPS_PROMPT = """Given the concept list and the source material, propose
learning relationships. Reply with a single JSON object:
{"relationships": [{"src": string, "dst": string,
"kind": "PREREQUISITE"|"RELATED"|"PART_OF"|"DEPENDS_ON",
"confidence": float in [0,1]}]}. src must be learned before dst for PREREQUISITE.
Propose only relationships supported by the material."""


class HTTPChatProvider(LLMProvider):
    """OpenAI-compatible chat-completions provider. The ML engine's
    GenerationSpec is embedded verbatim in the instruction; the provider is
    told explicitly not to alter concept/intervention — it decides HOW only."""

    def __init__(self, base_url: str, api_key: str, model: str, name: str | None = None,
                 timeout: float = 30.0, transport=None):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = name or f"http:{model}"
        self._client = httpx.Client(timeout=timeout, transport=transport)

    # ------------------------------------------------------------- transport
    def _chat(self, system: str, user: str, want_json: bool = False) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if want_json:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        r = self._client.post(self.base_url + "/chat/completions",
                              json=payload, headers=headers)
        if r.status_code == 400 and want_json:      # endpoint without json mode
            payload.pop("response_format", None)
            r = self._client.post(self.base_url + "/chat/completions",
                                  json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _spec_block(self, spec: GenerationSpec) -> str:
        return ("Instruction from the adaptive engine (do not change concept, "
                "intervention, or difficulty targets):\n" + json.dumps({
                    "concept": spec.concept, "mastery": spec.mastery,
                    "uncertainty": spec.uncertainty, "difficulty": spec.difficulty,
                    "intervention": spec.intervention,
                    "prerequisites": list(spec.prerequisites)}, indent=1))

    # ------------------------------------------------------------- content
    def generate_explanation(self, spec: GenerationSpec) -> str:
        return self._chat(
            "You are a precise tutor writing concise teaching content.",
            self._spec_block(spec) +
            f"\nWrite a short explanation of '{spec.concept}' suited to the "
            f"student state above. Plain prose, no JSON.").strip()

    def generate_example(self, spec: GenerationSpec) -> str:
        return self._chat(
            "You are a precise tutor writing worked examples.",
            self._spec_block(spec) +
            f"\nWrite one worked example illustrating '{spec.concept}'. "
            "Plain prose, no JSON.").strip()

    def generate_hint(self, spec: GenerationSpec) -> str:
        return self._chat(
            "You are a tutor giving one short hint. Never reveal the answer.",
            self._spec_block(spec) +
            f"\nGive one hint for a student working on '{spec.concept}'.").strip()

    def generate_question(self, spec: GenerationSpec) -> dict:
        raw = self._chat(QUESTION_SCHEMA_PROMPT, self._spec_block(spec), want_json=True)
        q = parse_json_loose(raw)
        if not isinstance(q, dict) or "question" not in q:
            raise ProviderOutputError("question payload missing 'question'")
        return q

    # ----------------------------------------------------------- extraction
    def extract_concepts(self, text: str) -> list[dict]:
        raw = self._chat(CONCEPTS_PROMPT, "Material:\n" + text[:6000], want_json=True)
        out = parse_json_loose(raw)
        concepts = out.get("concepts", []) if isinstance(out, dict) else []
        return [c for c in concepts
                if isinstance(c, dict) and str(c.get("name", "")).strip()]

    def extract_relationships(self, concepts: list[str], text: str) -> list[dict]:
        raw = self._chat(RELATIONSHIPS_PROMPT,
                         "Concepts: " + ", ".join(concepts) + "\nMaterial:\n" + text[:6000],
                         want_json=True)
        out = parse_json_loose(raw)
        rels = out.get("relationships", []) if isinstance(out, dict) else []
        valid = set(concepts)
        return [r for r in rels
                if isinstance(r, dict) and r.get("src") in valid and r.get("dst") in valid]

    # ----------------------------------------------------- flashcard/summary/lesson
    FLASHCARD_PROMPT = """Generate ONE flashcard. Reply with JSON only:
{"front": string (a question or prompt), "back": string (the answer/explanation),
 "concept": string}. The front should test recall of the concept."""

    SUMMARY_PROMPT = """Write a concise summary (3-5 sentences) of the concept.
Plain prose, no JSON."""

    LESSON_PROMPT = """Write a mini-lesson (structured overview, ~150 words) on the concept.
Include key points and relationships to prerequisites if relevant. Plain prose, no JSON."""

    def generate_flashcard(self, spec: GenerationSpec) -> dict:
        raw = self._chat(self.FLASHCARD_PROMPT, self._spec_block(spec) +
                         f"\n\nConcept: {spec.concept}", want_json=True)
        fc = parse_json_loose(raw)
        if not isinstance(fc, dict) or "front" not in fc or "back" not in fc:
            raise ProviderOutputError("flashcard payload missing front/back")
        fc.setdefault("concept", spec.concept)
        return fc

    def generate_summary(self, spec: GenerationSpec) -> str:
        return self._chat(
            "You are a tutor writing concise summaries.",
            self._spec_block(spec) +
            f"\n\nWrite a short summary of '{spec.concept}' for quick review.").strip()

    def generate_mini_lesson(self, spec: GenerationSpec) -> str:
        return self._chat(
            "You are a tutor writing structured mini-lessons.",
            self._spec_block(spec) +
            f"\n\nWrite a mini-lesson on '{spec.concept}' covering key points.").strip()
