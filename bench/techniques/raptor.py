"""RAPTOR hierarchical summarization for multi-granular retrieval.

Creates a recursive clustering and summarization tree over raw chunks,
enabling retrieval at the right granularity level.

Based on: RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
https://arxiv.org/abs/2401.18059

Example:
    retriever = RaptorRetriever(max_levels=3, cluster_model="gmm")
    await retriever.build_tree(chunks, graph_rag)
    context = await retriever("What is the main topic?", graph_rag)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from nano_graphrag import GraphRAG
from nano_graphrag.base import QueryParam


@dataclass
class RaptorNode:
    """A node in the RAPTOR tree."""

    content: str
    level: int
    children: List["RaptorNode"] | None = None
    parent: "RaptorNode" | None = None
    embedding: List[float] | None = None
    node_id: str = ""


class RaptorRetriever:
    """Recursive clustering and summarization for hierarchical retrieval.

    Builds a tree of summaries where:
    - Level 0: Original chunks
    - Level 1+: Cluster summaries from previous level

    Args:
        max_levels: Maximum tree depth.
        cluster_model: Clustering algorithm ("gmm" or "kmeans").
        summary_model: Model to use for summarization ("best" or "cheap").
        chunk_size: Target token size for summaries.
        top_k: Number of chunks to retrieve per level.

    Attributes:
        max_levels: Maximum tree depth.
        cluster_model: Clustering algorithm.
        summary_model: Model for summarization.
        chunk_size: Target token size.
        top_k: Number of chunks to retrieve.
        root: Root node of the tree.
    """

    def __init__(
        self,
        max_levels: int = 3,
        cluster_model: str = "gmm",
        summary_model: str = "cheap",
        chunk_size: int = 200,
        top_k: int = 10,
    ) -> None:
        if cluster_model not in ["gmm", "kmeans"]:
            raise ValueError(f"Unknown cluster model '{cluster_model}'. Use 'gmm' or 'kmeans'")

        self._max_levels = max_levels
        self._cluster_model = cluster_model
        self._summary_model = summary_model
        self._chunk_size = chunk_size
        self._top_k = top_k
        self._root: Optional[RaptorNode] = None

    async def build_tree(self, chunks: List[str], graph_rag: GraphRAG) -> None:
        """Build the RAPTOR tree from chunks.

        Args:
            chunks: List of text chunks.
            graph_rag: GraphRAG instance for embeddings and LLM calls.
        """
        # Create level 0 nodes (original chunks)
        level_0_nodes = []
        for i, chunk in enumerate(chunks):
            node = RaptorNode(content=chunk, level=0)
            node.node_id = f"chunk_{i}"
            level_0_nodes.append(node)

        # Build the tree recursively
        self._root = await self._build_level(level_0_nodes, 0, graph_rag)

    async def _build_level(
        self,
        nodes: List[RaptorNode],
        current_level: int,
        graph_rag: GraphRAG,
    ) -> RaptorNode:
        """Build a single level of the tree.

        Args:
            nodes: Nodes from the previous level.
            current_level: Current level number.
            graph_rag: GraphRAG instance.

        Returns:
            Root node for this subtree.
        """
        if current_level >= self._max_levels or len(nodes) <= 1:
            # Base case: single root node
            if len(nodes) == 1:
                return nodes[0]
            # Merge remaining nodes
            return await self._merge_nodes(nodes, graph_rag)

        # Cluster nodes by embedding similarity
        clusters = await self._cluster_nodes(nodes, graph_rag)

        # Create summary nodes for each cluster
        next_level_nodes = []
        for cluster_nodes in clusters:
            summary_node = await self._summarize_cluster(cluster_nodes, graph_rag)
            summary_node.level = current_level + 1
            summary_node.children = cluster_nodes
            for child in cluster_nodes:
                child.parent = summary_node
            next_level_nodes.append(summary_node)

        # Recursively build the next level
        return await self._build_level(next_level_nodes, current_level + 1, graph_rag)

    async def _cluster_nodes(
        self,
        nodes: List[RaptorNode],
        graph_rag: GraphRAG,
    ) -> List[List[RaptorNode]]:
        """Cluster nodes by embedding similarity.

        Args:
            nodes: Nodes to cluster.
            graph_rag: GraphRAG instance for embeddings.

        Returns:
            List of clusters (each cluster is a list of nodes).
        """
        if len(nodes) <= 2:
            return [nodes]

        # Get embeddings for all nodes
        contents = [node.content for node in nodes]
        embeddings = await graph_rag.embedding_func(contents)

        # Perform clustering
        if self._cluster_model == "gmm":
            return await self._gmm_cluster(nodes, embeddings)
        else:  # kmeans
            return await self._kmeans_cluster(nodes, embeddings)

    async def _gmm_cluster(
        self,
        nodes: List[RaptorNode],
        embeddings: List[List[float]],
    ) -> List[List[RaptorNode]]:
        """Cluster using Gaussian Mixture Model.

        Args:
            nodes: Nodes to cluster.
            embeddings: Node embeddings.

        Returns:
            List of clusters.
        """
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            raise ImportError(
                "scikit-learn is required for RAPTOR clustering. Install with: uv add scikit-learn"
            )

        import numpy as np

        X = np.array(embeddings)

        # Determine number of clusters (sqrt of n, max 5)
        n_clusters = min(int(np.sqrt(len(nodes))), 5)
        n_clusters = max(n_clusters, 2)

        # Fit GMM
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gmm.fit_predict(X)

        # Group nodes by cluster
        clusters: dict[int, List[RaptorNode]] = {}
        for node, label in zip(nodes, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node)

        return list(clusters.values())

    async def _kmeans_cluster(
        self,
        nodes: List[RaptorNode],
        embeddings: List[List[float]],
    ) -> List[List[RaptorNode]]:
        """Cluster using K-means.

        Args:
            nodes: Nodes to cluster.
            embeddings: Node embeddings.

        Returns:
            List of clusters.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required for RAPTOR clustering. Install with: uv add scikit-learn"
            )

        import numpy as np

        X = np.array(embeddings)

        # Determine number of clusters (sqrt of n, max 5)
        n_clusters = min(int(np.sqrt(len(nodes))), 5)
        n_clusters = max(n_clusters, 2)

        # Fit K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Group nodes by cluster
        clusters: dict[int, List[RaptorNode]] = {}
        for node, label in zip(nodes, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node)

        return list(clusters.values())

    async def _summarize_cluster(
        self,
        nodes: List[RaptorNode],
        graph_rag: GraphRAG,
    ) -> RaptorNode:
        """Summarize a cluster of nodes into a single node.

        Args:
            nodes: Nodes in the cluster.
            graph_rag: GraphRAG instance for LLM calls.

        Returns:
            Summary node.
        """
        # Combine node contents
        combined = "\n\n".join([node.content for node in nodes])

        # Truncate if too long
        if len(combined) > 4000:
            combined = combined[:4000]

        # Generate summary
        prompt = f"""Summarize the following text into a concise overview:

{combined}

Provide a clear, informative summary that captures the main points."""

        # Use the appropriate model
        if self._summary_model == "best" and graph_rag.best_model_func:
            summary = await graph_rag.best_model_func(prompt)
        elif graph_rag.cheap_model_func:
            summary = await graph_rag.cheap_model_func(prompt)
        else:
            summary = combined[:500]  # Fallback: truncate

        return RaptorNode(content=summary, level=0)

    async def _merge_nodes(
        self,
        nodes: List[RaptorNode],
        graph_rag: GraphRAG,
    ) -> RaptorNode:
        """Merge remaining nodes when max level is reached.

        Args:
            nodes: Nodes to merge.
            graph_rag: GraphRAG instance.

        Returns:
            Merged node.
        """
        return await self._summarize_cluster(nodes, graph_rag)

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        """Retrieve context using RAPTOR tree.

        Args:
            query: User question.
            graph_rag: GraphRAG instance.
            param: Query parameters (unused, kept for protocol compatibility).
            **kwargs: Additional parameters.

        Returns:
            Retrieved context as string.
        """
        if self._root is None:
            return ""

        # Search the tree for relevant chunks
        relevant_chunks = await self._search_tree(query, graph_rag)

        return "\n\n".join(relevant_chunks[: self._top_k])

    async def _search_tree(
        self,
        query: str,
        graph_rag: GraphRAG,
    ) -> List[str]:
        """Search the RAPTOR tree for relevant chunks.

        Args:
            query: User question.
            graph_rag: GraphRAG instance for embeddings.

        Returns:
            List of relevant chunk contents.
        """
        # Get query embedding
        query_emb = await graph_rag.embedding_func([query])[0]

        # Collect all nodes in the tree
        all_nodes = self._collect_nodes(self._root)

        # Calculate similarity scores
        scored_nodes = []
        for node in all_nodes:
            if node.embedding is None:
                # Get embedding on-demand
                node_emb = await graph_rag.embedding_func([node.content])[0]
                node.embedding = node_emb
            else:
                node_emb = node.embedding

            # Simple cosine similarity (dot product for normalized embeddings)
            score = sum(q * e for q, e in zip(query_emb, node_emb))
            scored_nodes.append((score, node.content))

        # Sort by score and return top contents
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in scored_nodes]

    def _collect_nodes(self, node: RaptorNode | None) -> List[RaptorNode]:
        """Collect all nodes in the tree.

        Args:
            node: Root node.

        Returns:
            List of all nodes.
        """
        if node is None:
            return []

        nodes = [node]
        if node.children:
            for child in node.children:
                nodes.extend(self._collect_nodes(child))
        return nodes

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RaptorRetriever":
        """Create retriever from configuration dict.

        Args:
            config: Configuration dict with keys:
                - max_levels (int): Maximum tree depth
                - cluster_model (str): Clustering algorithm
                - summary_model (str): Model for summarization
                - chunk_size (int): Target token size
                - top_k (int): Number of chunks to retrieve

        Returns:
            Configured RaptorRetriever instance.
        """
        return cls(
            max_levels=config.get("max_levels", 3),
            cluster_model=config.get("cluster_model", "gmm"),
            summary_model=config.get("summary_model", "cheap"),
            chunk_size=config.get("chunk_size", 200),
            top_k=config.get("top_k", 10),
        )
