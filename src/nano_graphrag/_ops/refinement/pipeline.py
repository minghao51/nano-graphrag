from __future__ import annotations

import json
import os
import time
from typing import Any

from ..._utils import get_all_nodes_safe, logger
from .enrich import _enrich_phase
from .infer import _infer_phase
from .merge import _merge_phase

_MAX_JOURNAL_ENTRIES = 100
_REJECTION_TTL_DAYS = {0: 3, 100: 5, 500: 7}


def _get_rejection_ttl(graph_size: int) -> float:
    for threshold in sorted(_REJECTION_TTL_DAYS.keys(), reverse=True):
        if graph_size >= threshold:
            return _REJECTION_TTL_DAYS[threshold] * 86400
    return 3 * 86400


class RefinementJournal:
    def __init__(self, path: str):
        self.path = path
        self.entries: list[dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.entries.append(json.loads(line))
            except (json.JSONDecodeError, OSError):
                self.entries = []

    def save(self):
        with open(self.path, "w") as f:
            for entry in self.entries[-_MAX_JOURNAL_ENTRIES:]:
                f.write(json.dumps(entry) + "\n")

    def add(self, phase: str, stats: dict[str, Any]):
        self.entries.append(
            {
                "timestamp": time.time(),
                "phase": phase,
                "stats": stats,
            }
        )
        if len(self.entries) > _MAX_JOURNAL_ENTRIES * 2:
            self.entries = self.entries[-_MAX_JOURNAL_ENTRIES:]


class RejectionCache:
    def __init__(self, path: str):
        self.path = path
        self._data: dict[str, float] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __setitem__(self, key: str, value: float):
        self._data[key] = value

    def prune(self, ttl: float):
        now = time.time()
        self._data = {k: v for k, v in self._data.items() if now - v < ttl}


async def arefine(
    knowledge_graph_inst,
    entity_vdb,
    text_chunks_kv,
    global_config: dict,
    phases: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    if not global_config.get("enable_refinement", False):
        logger.info("refinement_skipped", reason="enable_refinement=False")
        return {"skipped": {"reason": "enable_refinement is disabled"}}

    all_phases = ["merge", "enrich", "infer"]
    if phases is None:
        phases = all_phases
    phases = [p for p in phases if p in all_phases]

    if not phases:
        return {"skipped": {"reason": "no valid phases specified"}}

    working_dir = global_config.get("working_dir", "./nano_graphrag")
    journal = RefinementJournal(os.path.join(working_dir, "refinement_journal.jsonl"))
    rejection_cache = RejectionCache(os.path.join(working_dir, "refinement_rejections.json"))

    merge_threshold = global_config.get("refinement_merge_threshold", 0.93)
    enrich_min_chars = global_config.get("refinement_enrich_min_chars", 80)
    infer_confidence = global_config.get("refinement_infer_confidence", 0.80)
    infer_hub_cap = global_config.get("refinement_infer_hub_cap", 3)
    batch_size = global_config.get("refinement_batch_size", 50)

    all_nodes = await get_all_nodes_safe(knowledge_graph_inst)
    graph_size = len(all_nodes)
    ttl = _get_rejection_ttl(graph_size)
    rejection_cache.prune(ttl)

    results: dict[str, dict[str, Any]] = {}

    if "merge" in phases:
        logger.info("refinement_merge_start")
        stats = await _merge_phase(
            knowledge_graph_inst,
            entity_vdb,
            global_config,
            merge_threshold=merge_threshold,
            hub_cap=infer_hub_cap,
        )
        results["merge"] = stats
        journal.add("merge", stats)

    if "enrich" in phases:
        logger.info("refinement_enrich_start")
        stats = await _enrich_phase(
            knowledge_graph_inst,
            text_chunks_kv,
            global_config,
            min_chars=enrich_min_chars,
            batch_size=batch_size,
        )
        results["enrich"] = stats
        journal.add("enrich", stats)

    if "infer" in phases:
        logger.info("refinement_infer_start")
        stats = await _infer_phase(
            knowledge_graph_inst,
            text_chunks_kv,
            entity_vdb,
            global_config,
            min_confidence=infer_confidence,
            hub_cap=infer_hub_cap,
            batch_size=batch_size,
            rejection_cache=rejection_cache,
        )
        results["infer"] = stats
        journal.add("infer", stats)

    journal.save()
    rejection_cache.save()
    return results
