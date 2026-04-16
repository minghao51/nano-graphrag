"""Dataset loaders for multi-hop RAG benchmarks."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Protocol


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
    max_corpus_samples: int = -1  # -1 means all corpus documents
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

        # Apply max_corpus_samples limit
        if self.max_corpus_samples > 0:
            data = data[: self.max_corpus_samples]

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
        """Download MultiHop-RAG dataset from HuggingFace.

        Args:
            cache_dir: Directory to cache downloaded files
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library is required for auto-download. "
                "Install with: uv add --optional datasets"
            )

        cache_path = Path(cache_dir).expanduser()
        dataset_dir = cache_path / "multihoprag"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print("[Download] Loading MultiHopRAG from HuggingFace...")
        hf_dataset = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")

        questions_data = []
        corpus_by_title: Dict[str, Dict[str, str]] = {}  # title -> {id, title, content}

        for item in hf_dataset:
            qa_id = item.get("id", f"multihoprag_{len(questions_data)}")
            evidence_list = item.get("evidence_list", [])

            supporting_facts = []

            for ev in evidence_list:
                if isinstance(ev, dict):
                    fact = ev.get("fact", "")
                    title = ev.get("title", "")
                    if fact:
                        supporting_facts.append(fact)
                    if title and title not in corpus_by_title:
                        corpus_by_title[title] = {
                            "id": f"doc_{len(corpus_by_title)}",
                            "title": title,
                            "content": fact,
                        }
                    elif title and fact:
                        # Merge facts for same title
                        existing = corpus_by_title[title]["content"]
                        if fact not in existing:
                            corpus_by_title[title]["content"] = existing + "\n" + fact
                elif isinstance(ev, str):
                    supporting_facts.append(ev)

            questions_data.append(
                {
                    "id": qa_id,
                    "question": item["query"],
                    "answer": item["answer"],
                    "supporting_facts": supporting_facts,
                }
            )

        corpus_data = list(corpus_by_title.values())

        questions_path = dataset_dir / "questions.json"
        corpus_path = dataset_dir / "corpus.json"

        with open(questions_path, "w", encoding="utf-8") as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)

        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(corpus_data, f, indent=2, ensure_ascii=False)

        self.questions_path = str(questions_path)
        self.corpus_path = str(corpus_path)

        print(f"[Download] Questions saved to {questions_path}")
        print(f"[Download] Corpus saved to {corpus_path}")


@dataclass
class HotpotQADataset:
    """HotpotQA dataset loader.

    Supports both "dev" and "test" splits.
    Expected format: standard HotpotQA JSON format.
    """

    data_path: str
    split: str = "dev"
    max_samples: int = -1
    max_corpus_samples: int = -1
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

        # Apply max_samples limit to questions for corpus extraction
        if self.max_samples > 0:
            data = data[: self.max_samples]

        # Collect all unique context documents
        seen = set()
        corpus_count = 0
        for item in data:
            for title, sentences in item.get("context", []):
                doc_id = compute_mdhash_id(title, prefix="hotpot_")
                if doc_id not in seen:
                    seen.add(doc_id)
                    content = f"{title}\n{' '.join(sentences)}"
                    yield Passage(id=doc_id, title=title, text=content.strip())
                    corpus_count += 1
                    # Apply max_corpus_samples limit
                    if self.max_corpus_samples > 0 and corpus_count >= self.max_corpus_samples:
                        return

    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None:
        """Download HotpotQA dataset from HuggingFace.

        Args:
            cache_dir: Directory to cache downloaded files
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library is required for auto-download. "
                "Install with: uv add --optional datasets"
            )

        cache_path = Path(cache_dir).expanduser()
        dataset_dir = cache_path / "hotpotqa"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Download] Loading HotpotQA {self.split} split from HuggingFace...")
        hf_dataset = load_dataset("hotpotqa/hotpot_qa", self.split)

        questions_data = []
        corpus_docs = {}

        for item in hf_dataset:
            qa_id = item.get("id", f"hotpot_{len(questions_data)}")
            supporting_facts = []

            context = item.get("context", [])
            for title, sentences in context:
                if isinstance(sentences, list):
                    content = " ".join(sentences)
                else:
                    content = str(sentences)

                if title not in corpus_docs:
                    doc_id = compute_mdhash_id(title, prefix="hotpot_")
                    corpus_docs[title] = {"id": doc_id, "title": title, "content": content}

            sp = item.get("supporting_facts", {})
            if isinstance(sp, dict) and "title" in sp:
                for fact_title in sp["title"]:
                    if fact_title in corpus_docs:
                        supporting_facts.append(fact_title)

            questions_data.append(
                {
                    "id": qa_id,
                    "question": item["question"],
                    "answer": item["answer"],
                    "supporting_facts": supporting_facts,
                    "metadata": {
                        "type": item.get("type", "unknown"),
                        "level": item.get("level", "unknown"),
                    },
                }
            )

        output_path = dataset_dir / f"{self.split}.json"
        with open(output_path, "w", encoding="utf-8", errors="replace") as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)

        corpus_path = dataset_dir / "corpus.json"
        with open(corpus_path, "w", encoding="utf-8", errors="replace") as f:
            json.dump(list(corpus_docs.values()), f, indent=2, ensure_ascii=False)

        self.data_path = str(output_path)

        print(f"[Download] Questions saved to {output_path}")
        print(f"[Download] Corpus saved to {corpus_path}")


@dataclass
class MuSiQueDataset:
    """MuSiQue dataset loader (Multi-hop Question Answering).

    Expected format: MuSiQue JSON format with decompositions.
    """

    data_path: str
    split: str = "dev"
    max_samples: int = -1
    max_corpus_samples: int = -1
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

        # Apply max_samples limit to questions for corpus extraction
        if self.max_samples > 0:
            data = data[: self.max_samples]

        doc_idx = 0
        corpus_count = 0
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
                    corpus_count += 1
                    # Apply max_corpus_samples limit
                    if self.max_corpus_samples > 0 and corpus_count >= self.max_corpus_samples:
                        return

    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None:
        """Download MuSiQue dataset from HuggingFace.

        Args:
            cache_dir: Directory to cache downloaded files
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library is required for auto-download. "
                "Install with: uv add --optional datasets"
            )

        cache_path = Path(cache_dir).expanduser()
        dataset_dir = cache_path / "musique"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Download] Loading MuSiQue {self.split} split from HuggingFace...")
        hf_dataset = load_dataset("voidful/MuSiQue", "default", split=self.split)

        questions_data = []
        corpus_docs = {}

        for item in hf_dataset:
            qa_id = item.get("id", f"musique_{len(questions_data)}")
            supporting_facts = []

            paragraphs = item.get("paragraphs", [])
            for para in paragraphs:
                if isinstance(para, dict):
                    title = para.get("title", "")
                    content = para.get("paragraph_text", "")
                    para_idx = para.get("idx", "")

                    if content and title:
                        doc_key = f"{title}_{para_idx}"
                        if doc_key not in corpus_docs:
                            doc_id = f"musique_doc_{len(corpus_docs)}"
                            corpus_docs[doc_key] = {
                                "id": doc_id,
                                "title": title,
                                "content": content,
                            }

                    decomp = para.get("decomposition", [])
                    for d in decomp:
                        if isinstance(d, dict) and d.get("answer"):
                            supporting_facts.append(d["answer"])

            questions_data.append(
                {
                    "id": qa_id,
                    "question": item["question"],
                    "answer": item["answer"],
                    "supporting_facts": supporting_facts[:2] if supporting_facts else [],
                    "metadata": {
                        "decomposition": item.get("question_decomposition", []),
                    },
                }
            )

        output_path = dataset_dir / f"{self.split}.json"
        with open(output_path, "w", encoding="utf-8", errors="replace") as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)

        corpus_path = dataset_dir / "corpus.json"
        with open(corpus_path, "w", encoding="utf-8", errors="replace") as f:
            json.dump(list(corpus_docs.values()), f, indent=2, ensure_ascii=False)

        self.data_path = str(output_path)

        print(f"[Download] Questions saved to {output_path}")
        print(f"[Download] Corpus saved to {corpus_path}")


@dataclass
class TwoWikiMultiHopQADataset:
    """2WikiMultiHopQA dataset loader.

    Expected format: 2WikiMultiHopQA JSON format.
    """

    data_path: str = ""
    corpus_path: str = ""
    split: str = "dev"
    max_samples: int = -1
    max_corpus_samples: int = -1
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
                supporting_facts=item.get("supporting_facts", []),
                metadata={
                    "type": item.get("type", "unknown"),
                },
            )

    def corpus(self) -> Iterator[Passage]:
        """Load corpus documents from separate corpus file."""
        if not self.corpus_path:
            return

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for idx, doc in enumerate(data):
            if self.max_corpus_samples > 0 and count >= self.max_corpus_samples:
                return

            if isinstance(doc, dict):
                content = doc.get("content", "")
                title = doc.get("title", "")
                doc_id = doc.get("id", f"2wiki_doc_{idx}")
            else:
                continue

            if content.strip():
                yield Passage(id=doc_id, title=title, text=content.strip())
                count += 1

    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None:
        """Download 2WikiMultiHopQA dataset from HuggingFace.

        Args:
            cache_dir: Directory to cache downloaded files
        """
        try:
            import ast

            import pandas as pd
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub and pandas are required for auto-download. "
                "Install with: uv add huggingface_hub pandas"
            )

        cache_path = Path(cache_dir).expanduser()
        dataset_dir = cache_path / "2wikimultihopqa"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Download] Loading 2WikiMultiHopQA {self.split} split from HuggingFace...")

        parquet_file = hf_hub_download(
            repo_id="xanhho/2WikiMultihopQA",
            filename=f"{self.split}.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(parquet_file)

        questions_data = []
        corpus_docs = {}

        for _, row in df.iterrows():
            qa_id = row.get("_id", f"2wiki_{len(questions_data)}")
            supporting_facts = []

            context = row.get("context", [])
            if isinstance(context, str):
                context = ast.literal_eval(context)
            if isinstance(context, list):
                for title, sentences in context:
                    if isinstance(sentences, list):
                        content = " ".join(sentences)
                    else:
                        content = str(sentences)

                    if title and content:
                        doc_key = f"{title}_{len(corpus_docs)}"
                        if doc_key not in corpus_docs:
                            doc_id = f"2wiki_doc_{len(corpus_docs)}"
                            corpus_docs[doc_key] = {
                                "id": doc_id,
                                "title": title,
                                "content": content,
                            }

            sp = row.get("supporting_facts", [])
            if isinstance(sp, str):
                sp = ast.literal_eval(sp)
            if isinstance(sp, list):
                for entry in sp:
                    if isinstance(entry, list) and len(entry) > 0:
                        title = entry[0]
                        if title and title not in supporting_facts:
                            supporting_facts.append(title)

            questions_data.append(
                {
                    "id": qa_id,
                    "question": row["question"],
                    "answer": row["answer"],
                    "supporting_facts": supporting_facts,
                    "metadata": {
                        "type": row.get("type", "unknown"),
                    },
                }
            )

        output_path = dataset_dir / f"{self.split}.json"
        with open(output_path, "w", encoding="utf-8", errors="replace") as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)

        corpus_path = dataset_dir / "corpus.json"
        with open(corpus_path, "w", encoding="utf-8", errors="replace") as f:
            json.dump(list(corpus_docs.values()), f, indent=2, ensure_ascii=False)

        self.data_path = str(output_path)
        self.corpus_path = str(corpus_path)

        print(f"[Download] Questions saved to {output_path}")
        print(f"[Download] Corpus saved to {corpus_path}")


# Helper function for content hashing
def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute MD5 hash of content for deduplication."""
    from hashlib import md5

    return prefix + md5(content.encode()).hexdigest()
