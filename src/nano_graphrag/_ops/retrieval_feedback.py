from __future__ import annotations

import time
from dataclasses import dataclass, field

from .._utils import logger


@dataclass
class RetrievalFeedback:
    query: str = ""
    doc_ids_retrieved: list[str] = field(default_factory=list)
    doc_ids_relevant: list[str] = field(default_factory=list)
    answer_correct: bool | None = None
    recall: float = 0.0
    precision: float = 0.0
    deducible: bool = False


async def compute_retrieval_feedback(
    query: str,
    global_config: dict,
    context_text: str | None = None,
    relevant_doc_ids: list[str] | None = None,
    doc_ids_retrieved: list[str] | None = None,
) -> RetrievalFeedback:
    feedback = RetrievalFeedback(
        query=query,
        doc_ids_retrieved=doc_ids_retrieved or [],
    )
    if relevant_doc_ids:
        retrieved_set = set(doc_ids_retrieved or [])
        relevant_set = set(relevant_doc_ids)
        if relevant_set:
            common = retrieved_set & relevant_set
            feedback.recall = len(common) / len(relevant_set)
            if retrieved_set:
                feedback.precision = len(common) / len(retrieved_set)
        feedback.doc_ids_relevant = list(relevant_set)
    llm_func = global_config.get("cheap_model_func")
    if llm_func is not None and context_text:
        ded_prompt = (
            f"Question: {query}\n\n"
            f"Can the answer be fully deduced from the following context?\n\n"
            f"---Context---\n"
            f"{context_text[:3000]}\n\n"
            f'Answer only "yes" or "no".'
        )
        try:
            resp = await llm_func(ded_prompt)
            feedback.deducible = isinstance(resp, str) and resp.strip().lower().startswith("yes")
        except Exception as e:
            logger.debug("deducibility_judge_failed", error=str(e))
    if not relevant_doc_ids and context_text:
        feedback.recall = 1.0 if feedback.deducible else 0.0
    return feedback


async def log_retrieval_feedback(
    feedback: RetrievalFeedback,
    document_index,
    global_config: dict,
):
    feedback_key = f"retrieval_feedback_{int(time.time())}"
    try:
        await document_index.upsert(
            {
                feedback_key: {
                    "query": feedback.query,
                    "doc_ids_retrieved": feedback.doc_ids_retrieved,
                    "doc_ids_relevant": feedback.doc_ids_relevant,
                    "answer_correct": feedback.answer_correct,
                    "recall": feedback.recall,
                    "precision": feedback.precision,
                    "deducible": feedback.deducible,
                }
            }
        )
    except Exception as e:
        logger.debug("feedback_log_failed", error=str(e))

    stats_key = "feedback_stats"
    try:
        existing_stats_raw = await document_index.get_by_id(stats_key)
        existing_stats = dict(existing_stats_raw) if isinstance(existing_stats_raw, dict) else {}
    except Exception:
        existing_stats = {}

    doc_ids_seen = set(feedback.doc_ids_retrieved)
    doc_ids_seen.update(feedback.doc_ids_relevant)
    for doc_id in doc_ids_seen:
        if doc_id not in existing_stats:
            existing_stats[doc_id] = {
                "total_queries": 0,
                "total_recall": 0.0,
                "total_precision": 0.0,
                "total_deducible": 0,
            }
        stat = existing_stats[doc_id]
        stat["total_queries"] = stat.get("total_queries", 0) + 1
        stat["total_recall"] = stat.get("total_recall", 0.0) + feedback.recall
        stat["total_precision"] = stat.get("total_precision", 0.0) + feedback.precision
        if feedback.deducible:
            stat["total_deducible"] = stat.get("total_deducible", 0) + 1
        stat["avg_recall"] = stat["total_recall"] / stat["total_queries"]
        stat["avg_precision"] = stat["total_precision"] / stat["total_queries"]
    existing_stats["_updated_at"] = time.time()
    try:
        await document_index.upsert({stats_key: existing_stats})
    except Exception as e:
        logger.debug("feedback_stats_upsert_failed", error=str(e))
