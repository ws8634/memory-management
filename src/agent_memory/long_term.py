from datetime import datetime
from typing import Any, Optional
import hashlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent_memory.config import LongTermConfig
from agent_memory.models import MemoryItem, MemoryType
from agent_memory.token_utils import estimate_tokens


class LongTermVectorMemory:
    def __init__(self, config: Optional[LongTermConfig] = None):
        self.config = config or LongTermConfig()
        self.documents: list[MemoryItem] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix: Optional[np.ndarray] = None
        self._needs_refit: bool = True
        self._doc_ids: set[str] = set()

    def upsert(
        self,
        content: str,
        source: Optional[str] = None,
        tags: Optional[list[str]] = None,
        importance: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, bool]:
        doc_id = self._generate_doc_id(content, source)
        
        existing_idx = self._find_doc_by_id(doc_id)
        
        doc_item = MemoryItem(
            id=doc_id,
            memory_type=MemoryType.LONG_TERM,
            content=content,
            source=source,
            tags=tags or [],
            importance=importance,
            is_active=True,
            metadata=metadata or {},
        )
        
        is_update = False
        
        if existing_idx is not None:
            self.documents[existing_idx] = doc_item
            is_update = True
        else:
            if len(self.documents) >= self.config.max_documents:
                self._evict_least_important()
            self.documents.append(doc_item)
            self._doc_ids.add(doc_id)
        
        self._needs_refit = True
        return doc_id, is_update

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[tuple[MemoryItem, float]]:
        k = top_k or self.config.top_k
        sim_threshold = threshold or self.config.similarity_threshold
        
        if not self.documents or k <= 0:
            return []
        
        if self._needs_refit or self._vectorizer is None:
            self._fit_vectorizer()
        
        active_docs = [d for d in self.documents if d.is_active]
        if not active_docs:
            return []
        
        query_vector = self._vectorizer.transform([query])
        
        doc_vectors = []
        active_indices = []
        for i, doc in enumerate(self.documents):
            if doc.is_active:
                active_indices.append(i)
        
        if self._tfidf_matrix is not None and len(active_indices) == self._tfidf_matrix.shape[0]:
            doc_vectors = self._tfidf_matrix
        else:
            doc_contents = [d.content for d in active_docs]
            doc_vectors = self._vectorizer.transform(doc_contents)
        
        similarities = cosine_similarity(query_vector, doc_vectors)[0]
        
        scored_docs: list[tuple[MemoryItem, float, float]] = []
        for i, sim in enumerate(similarities):
            if i < len(active_docs):
                doc = active_docs[i]
                adjusted_score = sim * doc.importance
                scored_docs.append((doc, adjusted_score, sim))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results: list[tuple[MemoryItem, float]] = []
        for doc, adjusted_score, raw_sim in scored_docs[:k]:
            if raw_sim >= sim_threshold:
                results.append((doc, adjusted_score))
        
        return results

    def invalidate(
        self,
        pattern: str,
        mark_inactive: bool = True,
        reduce_importance_to: Optional[float] = None,
    ) -> int:
        invalidated_count = 0
        pattern_lower = pattern.lower()
        
        for doc in self.documents:
            if pattern_lower in doc.content.lower() and doc.is_active:
                if mark_inactive:
                    doc.is_active = False
                if reduce_importance_to is not None:
                    doc.importance = min(doc.importance, reduce_importance_to)
                invalidated_count += 1
        
        return invalidated_count

    def deduplicate(self, threshold: Optional[float] = None) -> int:
        dedup_threshold = threshold or self.config.deduplication_threshold
        if len(self.documents) < 2:
            return 0
        
        if self._needs_refit or self._vectorizer is None:
            self._fit_vectorizer()
        
        active_docs = [d for d in self.documents if d.is_active]
        if len(active_docs) < 2:
            return 0
        
        contents = [d.content for d in active_docs]
        vectors = self._vectorizer.transform(contents)
        sim_matrix = cosine_similarity(vectors)
        
        to_merge: set[int] = set()
        merge_count = 0
        
        for i in range(len(active_docs)):
            if i in to_merge:
                continue
            
            for j in range(i + 1, len(active_docs)):
                if j in to_merge:
                    continue
                
                sim = sim_matrix[i][j]
                if sim >= dedup_threshold:
                    self._merge_documents(active_docs[i], active_docs[j])
                    to_merge.add(j)
                    merge_count += 1
        
        for idx in sorted(to_merge, reverse=True):
            doc = active_docs[idx]
            doc.is_active = False
        
        return merge_count

    def get_document_count(self, include_inactive: bool = False) -> int:
        if include_inactive:
            return len(self.documents)
        return len([d for d in self.documents if d.is_active])

    def format_for_context(
        self,
        results: list[tuple[MemoryItem, float]],
        include_score: bool = False,
        separator: str = "\n",
    ) -> str:
        if not results:
            return ""
        
        parts: list[str] = []
        for i, (doc, score) in enumerate(results, 1):
            source_info = f" [来源: {doc.source}]" if doc.source else ""
            score_info = f" (相关度: {score:.2f})" if include_score else ""
            parts.append(f"{i}. {doc.content}{source_info}{score_info}")
        
        return separator.join(parts)

    def clear(self) -> None:
        self.documents.clear()
        self._doc_ids.clear()
        self._vectorizer = None
        self._tfidf_matrix = None
        self._needs_refit = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": [
                {
                    "id": d.id,
                    "memory_type": d.memory_type.value,
                    "content": d.content,
                    "source": d.source,
                    "tags": d.tags,
                    "timestamp": d.timestamp.isoformat(),
                    "importance": d.importance,
                    "is_active": d.is_active,
                    "metadata": d.metadata,
                }
                for d in self.documents
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: Optional[LongTermConfig] = None) -> "LongTermVectorMemory":
        ltm = cls(config)
        from agent_memory.models import MemoryType
        
        for doc_data in data.get("documents", []):
            doc = MemoryItem(
                id=doc_data["id"],
                memory_type=MemoryType(doc_data["memory_type"]),
                content=doc_data["content"],
                source=doc_data.get("source"),
                tags=doc_data.get("tags", []),
                timestamp=datetime.fromisoformat(doc_data["timestamp"]) if "timestamp" in doc_data else datetime.now(),
                importance=doc_data.get("importance", 1.0),
                is_active=doc_data.get("is_active", True),
                metadata=doc_data.get("metadata", {}),
            )
            ltm.documents.append(doc)
            ltm._doc_ids.add(doc.id)
        ltm._needs_refit = True
        return ltm

    def _generate_doc_id(self, content: str, source: Optional[str]) -> str:
        key = f"{content}:{source or ''}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]

    def _find_doc_by_id(self, doc_id: str) -> Optional[int]:
        for i, doc in enumerate(self.documents):
            if doc.id == doc_id:
                return i
        return None

    def _fit_vectorizer(self) -> None:
        if not self.documents:
            self._vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words="english",
            )
            return
        
        contents = [d.content for d in self.documents if d.is_active]
        
        self._vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        
        if contents:
            self._tfidf_matrix = self._vectorizer.fit_transform(contents)
        self._needs_refit = False

    def _evict_least_important(self) -> None:
        if not self.documents:
            return
        
        sorted_docs = sorted(
            self.documents,
            key=lambda d: (d.importance, d.timestamp)
        )
        
        to_remove = sorted_docs[0]
        self._doc_ids.discard(to_remove.id)
        self.documents.remove(to_remove)

    def _merge_documents(self, primary: MemoryItem, secondary: MemoryItem) -> None:
        primary.importance = max(primary.importance, secondary.importance)
        
        for tag in secondary.tags:
            if tag not in primary.tags:
                primary.tags.append(tag)
        
        if secondary.source and not primary.source:
            primary.source = secondary.source
        
        primary.metadata.update(secondary.metadata)
        primary.timestamp = max(primary.timestamp, secondary.timestamp)
