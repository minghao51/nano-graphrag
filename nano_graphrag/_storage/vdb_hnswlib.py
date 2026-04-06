import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import hnswlib
import numpy as np
import xxhash

try:
    import msgpack
    _USE_MSGPACK = True
except ImportError:
    _USE_MSGPACK = False

from .._utils import logger
from ..base import BaseVectorStorage


@dataclass
class HNSWVectorStorage(BaseVectorStorage):
    ef_construction: int = 100
    M: int = 16
    max_elements: int = 1000000
    ef_search: int = 50
    num_threads: int = -1
    _index: Any = field(init=False)
    _metadata: dict[str, dict] = field(default_factory=dict)
    _current_elements: int = 0

    def _create_fresh_index(self, max_elements: int) -> hnswlib.Index:
        """Create and initialize a new HNSW index."""
        idx = hnswlib.Index(space="cosine", dim=self.embedding_func.embedding_dim)
        idx.init_index(max_elements=max_elements, ef_construction=self.ef_construction, M=self.M)
        idx.set_ef(self.ef_search)
        return idx

    def __post_init__(self):
        self._index_file_name = os.path.join(
            self.global_config["working_dir"], f"{self.namespace}_hnsw.index"
        )
        self._metadata_file_name = os.path.join(
            self.global_config["working_dir"], f"{self.namespace}_hnsw_metadata.mpk"
        )
        self._embedding_batch_num = self.global_config.get(
            "embedding_batch_size",
            self.global_config.get("embedding_batch_num", 100),
        )

        hnsw_params = self.global_config.get("vector_db_storage_cls_kwargs", {})
        self.ef_construction = hnsw_params.get("ef_construction", self.ef_construction)
        self.M = hnsw_params.get("M", self.M)
        self.max_elements = hnsw_params.get("max_elements", self.max_elements)
        self.ef_search = hnsw_params.get("ef_search", self.ef_search)
        self.num_threads = hnsw_params.get("num_threads", self.num_threads)

        if os.path.exists(self._index_file_name) and os.path.exists(self._metadata_file_name):
            self._index = hnswlib.Index(space="cosine", dim=self.embedding_func.embedding_dim)
            self._index.load_index(self._index_file_name, max_elements=self.max_elements)
            with open(self._metadata_file_name, "rb") as f:
                if _USE_MSGPACK:
                    loaded = msgpack.load(f, raw=False)
                else:
                    import json
                    loaded = json.load(f)
                self._metadata = {int(k): v for k, v in loaded["metadata"].items()}
                self._current_elements = loaded["current_elements"]
            logger.info(
                f"Loaded existing index for {self.namespace} with {self._current_elements} elements"
            )
        else:
            self._index = self._create_fresh_index(self.max_elements)
            self._metadata = {}
            self._current_elements = 0
            logger.info(f"Created new index for {self.namespace}")

    async def upsert(self, data: dict[str, dict]) -> np.ndarray:
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not data:
            logger.warning("You insert an empty data to vector DB")
            return []

        if self._current_elements + len(data) > self.max_elements:
            new_max = max(self.max_elements * 2, self._current_elements + len(data))
            logger.info(f"Resizing HNSW index from {self.max_elements} to {new_max}")
            self._index.save_index(self._index_file_name)
            new_index = hnswlib.Index(space="cosine", dim=self.embedding_func.embedding_dim)
            new_index.load_index(self._index_file_name, max_elements=new_max)
            new_index.set_ef(self.ef_search)
            self._index = new_index
            self.max_elements = new_max

        list_data = [
            {
                "id": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batch_size = min(self._embedding_batch_num, len(contents))
        embeddings = np.concatenate(
            await asyncio.gather(
                *[
                    self.embedding_func(contents[i : i + batch_size])
                    for i in range(0, len(contents), batch_size)
                ]
            )
        )

        ids = np.fromiter(
            (xxhash.xxh32_intdigest(d["id"].encode()) for d in list_data),
            dtype=np.uint32,
            count=len(list_data),
        )
        self._metadata.update(
            {
                id_int: {k: v for k, v in d.items() if k in self.meta_fields or k == "id"}
                for id_int, d in zip(ids, list_data)
            }
        )
        self._index.add_items(data=embeddings, ids=ids, num_threads=self.num_threads)
        self._current_elements = self._index.get_current_count()
        return ids

    async def query(self, query: str, top_k: int = 5, better_than_threshold: Optional[float] = None) -> list[dict]:
        if self._current_elements == 0:
            return []

        actual_k = min(top_k, self._current_elements)
        if actual_k == 0:
            return []

        query_k = min(top_k * 3, self._current_elements)
        query_k = max(query_k, top_k)
        query_k = min(query_k, self._current_elements)

        if query_k >= self.ef_search:
            target_ef = query_k + 1
            logger.warning(
                f"Setting ef_search to {target_ef} because k={query_k} requires ef > k"
            )
            self._index.set_ef(target_ef)

        embedding = await self.embedding_func([query])
        try:
            labels, distances = self._index.knn_query(
                data=embedding[0], k=query_k, num_threads=self.num_threads
            )
        except RuntimeError:
            retry_k = actual_k if query_k > actual_k else max(1, query_k // 2)
            while True:
                try:
                    labels, distances = self._index.knn_query(
                        data=embedding[0], k=retry_k, num_threads=self.num_threads
                    )
                    break
                except RuntimeError:
                    if retry_k == 1:
                        raise
                    retry_k = max(1, retry_k // 2)

        results = []
        for label, distance in zip(labels[0], distances[0]):
            if label not in self._metadata:
                continue
            similarity = 1 - distance
            if better_than_threshold is not None and similarity < better_than_threshold:
                continue
            results.append({
                **self._metadata.get(label, {}),
                "distance": distance,
                "similarity": similarity,
            })
            if len(results) >= top_k:
                break

        return results

    async def delete(self, ids: list[str]):
        if not ids:
            return
        deleted_any = False
        for id_str in ids:
            id_int = xxhash.xxh32_intdigest(id_str.encode())
            if id_int in self._metadata:
                self._metadata.pop(id_int, None)
                if hasattr(self._index, "mark_deleted"):
                    self._index.mark_deleted(id_int)
                deleted_any = True

        if deleted_any:
            self._current_elements = self._index.get_current_count()
            if self._current_elements < self._index.get_max_elements() // 2:
                self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild the HNSW index with only active (non-deleted) vectors."""
        active_ids = list(self._metadata.keys())
        if not active_ids:
            self._index = self._create_fresh_index(max(self.max_elements, 1024))
            self._current_elements = 0
            return

        # Save vectors from old index before replacing
        old_vectors = self._index.get_items(active_ids)
        new_max = max(len(active_ids) * 2, self.max_elements)
        self._index = self._create_fresh_index(new_max)
        self._index.add_items(data=old_vectors, ids=np.array(active_ids), num_threads=self.num_threads)
        self.max_elements = new_max
        self._current_elements = len(active_ids)

    async def index_done_callback(self):
        self._index.save_index(self._index_file_name)
        data = {
            "metadata": {str(k): v for k, v in self._metadata.items()},
            "current_elements": self._current_elements,
        }
        if _USE_MSGPACK:
            with open(self._metadata_file_name, "wb") as f:
                msgpack.dump(data, f)
        else:
            import json
            with open(self._metadata_file_name, "w") as f:
                json.dump(data, f)
