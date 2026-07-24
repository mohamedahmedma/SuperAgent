from typing import Any, Optional
import importlib
import os
import sys
from uuid import uuid4

from langsmith import evaluate

# Add the project root to sys.path so the backend package can be imported
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.env import load_env

load_env()

chat_with_agent = importlib.import_module("backend.chat.service").chat_with_agent


def _extract_answer(outputs: Any) -> str:
    if isinstance(outputs, dict):
        # Prefer the field that holds the actual final response
        answer = outputs.get("response") or outputs.get("answer") or outputs.get("output")
        return str(answer or "").strip()
    if hasattr(outputs, "outputs") and isinstance(outputs.outputs, dict):
        answer = (
            outputs.outputs.get("response")
            or outputs.outputs.get("answer")
            or outputs.outputs.get("output")
        )
        return str(answer or "").strip()
    return ""


def _extract_reference(reference_outputs: Optional[dict]) -> str:
    if not isinstance(reference_outputs, dict):
        return ""
    for key in ("response", "answer", "output", "expected_answer"):
        value = reference_outputs.get(key)
        if value:
            return str(value).strip()
    return ""

# 1. Select your dataset
dataset_name = "RAG"

# 2. Define an evaluator (evaluates the final answer, not the retrieved chunks)
def custom_evaluator(run_outputs: dict, reference_outputs: dict) -> bool:
    answer = _extract_answer(run_outputs)
    if not answer:
        return False
    if "Retrieved Chunks:" in answer:
        return False

    reference = _extract_reference(reference_outputs)
    if not reference:
        return True

    # When a reference answer exists, at least ensure some semantic overlap
    # (using a lightweight character-set overlap ratio as a rough check)
    answer_chars = {ch for ch in answer if not ch.isspace()}
    ref_chars = {ch for ch in reference if not ch.isspace()}
    if not answer_chars or not ref_chars:
        return False

    overlap = len(answer_chars & ref_chars) / max(1, len(ref_chars))
    return overlap >= 0.2

# Directly invoke your existing full Agent flow as the evaluation target
def target_function(inputs: dict) -> dict:
    question = inputs["question"]
    # Use a separate session for each evaluation sample to avoid context bleed
    session_id = f"langsmith_eval_{uuid4().hex}"
    result = chat_with_agent(
        user_text=question,
        user_id="langsmith_eval_user",
        session_id=session_id,
    )

    response_text = ""
    rag_trace = {}
    if isinstance(result, dict):
        response_text = str(result.get("response", "") or "")
        rag_trace = result.get("rag_trace", {}) or {}
    else:
        response_text = str(result)

    return {
        "response": response_text,
        "rag_trace": rag_trace,
    }

# 3. Run an evaluation
# For more info on evaluators, see: https://docs.langchain.com/langsmith/evaluation-concepts
evaluate(
    target_function,
    data=dataset_name,
    evaluators=[custom_evaluator],
    experiment_prefix="RAG Pipeline Real Evaluation"
)
