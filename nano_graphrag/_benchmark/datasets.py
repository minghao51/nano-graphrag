"""Dataset loaders for multi-hop RAG benchmarks."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol


@dataclass
class QAPair:
    """A question-answer pair with optional supporting facts.

    Attributes:
        id: Unique identifier for the question
        question: The question text
        answer: The ground truth answer
        supporting_facts: List of supporting facts for context recall evaluation
        metadata: Additional metadata about the question
    """

    id: str
    question: str
    answer: str
    supporting_facts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Passage:
    """A document passage for indexing.

    Attributes:
        id: Unique identifier for the passage
        title: Optional title or heading
        text: The passage content
    """

    id: str
    title: str = ""
    text: str = ""


class BenchmarkDataset(Protocol):
    """Protocol for benchmark datasets.

    A dataset provides questions and a corpus for GraphRAG indexing.
    """

    name: str

    def questions(self, split: str = "test") -> Iterator[QAPair]:
        """Return questions with typed QAPair objects.

        Args:
            split: Dataset split (e.g., "test", "validation")

        Returns:
            Iterator of QAPair objects with question, answer, and optional
            supporting_facts and metadata
        """
        ...

    def corpus(self) -> Iterator[Passage]:
        """Return corpus documents as Passage objects.

        Returns:
            Iterator of Passage objects with id, title, and text
        """
        ...

    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None:
        """Download dataset from remote source.

        Args:
            cache_dir: Directory to cache downloaded files

        Raises:
            NotImplementedError: If download not supported for this dataset
        """
        ...


@dataclass
class MultiHopRAGDataset:
    """MultiHop-RAG dataset loader.

    Expects JSON files with the following format:

    Questions file:
    [
        {
            "id": "q1",  // optional, will be generated if missing
            "question": "What is the capital of France?",
            "answer": "Paris",
            "supporting_facts": [...],  // optional
            "metadata": {...}  // optional
        },
        ...
    ]

    Corpus file:
    [
        {
            "id": "doc1",  // optional, will be generated if missing
            "content": "Document text here...",
            "title": "Optional title"  // optional
        },
        ...
    ]
    """

    questions_path: str
    corpus_path: str
    max_samples: int = -1  # -1 means all samples
    name: str = "multihop-rag"

    def questions(self, split: str = "test") -> Iterator[QAPair]:
        """Load questions from JSON file."""
        with open(self.questions_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Apply max_samples limit
        if self.max_samples > 0:
            data = data[: self.max_samples]

        for idx, item in enumerate(data):
            # Generate ID if not provided
            qa_id = item.get("id", f"q_{idx}")

            yield QAPair(
                id=qa_id,
                question=item["question"],
                answer=item["answer"],
                supporting_facts=item.get("supporting_facts", []),
                metadata=item.get("metadata", {}),
            )

    def corpus(self) -> Iterator[Passage]:
        """Load corpus documents from JSON file."""
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, doc in enumerate(data):
            if isinstance(doc, dict):
                content = doc.get("content", "")
                title = doc.get("title", "")
                doc_id = doc.get("id", f"doc_{idx}")
            elif isinstance(doc, str):
                content = doc
                title = ""
                doc_id = f"doc_{idx}"
            else:
                continue

            if content.strip():
                yield Passage(id=doc_id, title=title, text=content.strip())

    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None:
        """Download MultiHop-RAG dataset.

        Raises:
            NotImplementedError: MultiHop-RAG dataset requires manual download
        """
        raise NotImplementedError(
            "MultiHop-RAG dataset requires manual download. "
            "Please download from the official source and provide paths."
        )


@dataclass
class HotpotQADataset:
    """HotpotQA dataset loader.

    Supports both "dev" and "test" splits.
    Expected format: standard HotpotQA JSON format.
    """

    data_path: str
    split: str = "dev"
    max_samples: int = -1
    name: str = "hotpotqa"

    def questions(self, split: str = "test") -> Iterator[QAPair]:
        """Load questions from HotpotQA dataset."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Apply max_samples limit
        if self.max_samples > 0:
            data = data[: self.max_samples]

        for idx, item in enumerate(data):
            # Extract supporting facts from context if available
            supporting_facts = []
            if "supporting_facts" in item:
                # HotpotQA includes supporting fact titles
                supporting_facts = item["supporting_facts"]

            yield QAPair(
                id=item.get("_id", f"hotpot_{idx}"),
                question=item["question"],
                answer=item["answer"],
                supporting_facts=supporting_facts,
                metadata={
                    "type": item.get("type", "unknown"),
                    "level": item.get("level", "unknown"),
                },
            )

    def corpus(self) -> Iterator[Passage]:
        """Extract context documents as corpus."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Collect all unique context documents
        seen = set()
        for item in data:
            for title, sentences in item.get("context", []):
                doc_id = compute_mdhash_id(title, prefix="hotpot_")
                if doc_id not in seen:
                    seen.add(doc_id)
                    content = f"{title}\n{' '.join(sentences)}"
                    yield Passage(id=doc_id, title=title, text=content.strip())

    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None:
        """Download HotpotQA dataset.

        Raises:
            NotImplementedError: HotpotQA requires manual download from official source
        """
        raise NotImplementedError(
            "HotpotQA requires manual download from http://hotpotqa.org. "
            "Please download and provide the data path."
        )


@dataclass
class MuSiQueDataset:
    """MuSiQue dataset loader (Multi-hop Question Answering).

    Expected format: MuSiQue JSON format with decompositions.
    """

    data_path: str
    split: str = "dev"
    max_samples: int = -1
    name: str = "musique"

    def questions(self, split: str = "test") -> Iterator[QAPair]:
        """Load questions from MuSiQue dataset."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Apply max_samples limit
        if self.max_samples > 0:
            data = data[: self.max_samples]

        for idx, item in enumerate(data):
            yield QAPair(
                id=item.get("id", f"musique_{idx}"),
                question=item["question"],
                answer=item["answer"],
                supporting_facts=[],  # MuSiQue doesn't provide explicit supporting facts
                metadata={
                    "decomposition": item.get("question_decomposition", []),
                },
            )

    def corpus(self) -> Iterator[Passage]:
        """Extract context documents as corpus."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        doc_idx = 0
        for item in data:
            for para in item.get("context", []):
                if isinstance(para, dict):
                    content = para.get("body", "")
                    title = para.get("title", "")
                elif isinstance(para, str):
                    content = para
                    title = ""
                else:
                    continue

                if content.strip():
                    yield Passage(id=f"musique_doc_{doc_idx}", title=title, text=content.strip())
                    doc_idx += 1

    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None:
        """Download MuSiQue dataset.

        Raises:
            NotImplementedError: MuSiQue requires manual download from official source
        """
        raise NotImplementedError(
            "MuSiQue requires manual download from the official repository. "
            "Please download and provide the data path."
        )


@dataclass
class TwoWikiMultiHopQADataset:
    """2WikiMultiHopQA dataset loader.

    Expected format: 2WikiMultiHopQA JSON format.
    """

    data_path: str
    split: str = "dev"
    max_samples: int = -1
    name: str = "2wikimultihopqa"

    def questions(self, split: str = "test") -> Iterator[QAPair]:
        """Load questions from 2WikiMultiHopQA dataset."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Apply max_samples limit
        if self.max_samples > 0:
            data = data[: self.max_samples]

        for idx, item in enumerate(data):
            yield QAPair(
                id=item.get("_id", f"2wiki_{idx}"),
                question=item["question"],
                answer=item["answer"],
                supporting_facts=[],  # 2Wiki doesn't provide explicit supporting facts
                metadata={
                    "type": item.get("type", "unknown"),
                },
            )

    def corpus(self) -> Iterator[Passage]:
        """Extract context documents as corpus."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        doc_idx = 0
        for item in data:
            for evidence in item.get("evidence", []):
                if isinstance(evidence, dict):
                    content = evidence.get("content", "")
                    title = evidence.get("title", "")
                elif isinstance(evidence, str):
                    content = evidence
                    title = ""
                else:
                    continue

                if content.strip():
                    yield Passage(id=f"2wiki_doc_{doc_idx}", title=title, text=content.strip())
                    doc_idx += 1

    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None:
        """Download 2WikiMultiHopQA dataset.

        Raises:
            NotImplementedError: 2WikiMultiHopQA requires manual download from official source
        """
        raise NotImplementedError(
            "2WikiMultiHopQA requires manual download from the official repository. "
            "Please download and provide the data path."
        )


# Helper function for content hashing
def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute MD5 hash of content for deduplication."""
    from hashlib import md5

    return prefix + md5(content.encode()).hexdigest()
