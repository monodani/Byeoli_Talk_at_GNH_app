"""
Embedder and hybrid retrieval system for BYEOLI_TALK_AT_GNH_app.

Provides standardized embedding generation and hybrid search capabilities
combining FAISS semantic search with BM25 keyword search using RRF reranking.
"""

import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import json

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LLM imports based on provider
try:
    import openai
except ImportError:
    openai = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from utils.config import get_config
from utils.contracts import Citation
from utils.index_manager import get_index_manager


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Individual search result."""
    text: str
    source_id: str
    score: float
    rank: int
    metadata: Dict[str, Any]
    snippet: Optional[str] = None


@dataclass
class HybridSearchResults:
    """Combined results from hybrid search."""
    results: List[SearchResult]
    faiss_results: List[SearchResult]
    bm25_results: List[SearchResult]
    total_time_ms: float
    faiss_time_ms: float
    bm25_time_ms: float
    rerank_time_ms: float
    cache_hit: bool = False


class EmbeddingProvider:
    """Abstract base class for embedding providers."""
    
    def __init__(self, config):
        self.config = config
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        raise NotImplementedError
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.embed_texts([query])[0]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, config):
        super().__init__(config)
        if openai is None:
            raise ImportError("OpenAI package not installed")
        
        self.client = openai.OpenAI(api_key=config.get_api_key("openai"))
        self.model = config.model.embedding_model
        self.dimensions = config.model.embedding_dimensions
    
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        all_embeddings = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions if self.dimensions != 3072 else None
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"OpenAI embedding failed for batch {i//batch_size + 1}: {e}")
                # Use zero vectors as fallback
                fallback_embeddings = [[0.0] * self.dimensions] * len(batch)
                all_embeddings.extend(fallback_embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)


class CachedEmbedder:
    """Embedder with disk-based caching capabilities."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.provider = self._create_provider()
        self.cache_dir = self.config.paths.get_absolute_path(
            self.config.cache.embedding_cache_dir
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for frequently used embeddings
        self.memory_cache = {}
        self.memory_cache_size = 1000
    
    def _create_provider(self) -> EmbeddingProvider:
        """Create embedding provider based on configuration."""
        provider = self.config.model.embedding_provider.lower()
        
        if provider == "openai":
            return OpenAIEmbeddingProvider(self.config)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with caching."""
        if not texts:
            return np.array([])
        
        # Check cache for each text
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = self._hash_text(text)
            
            # Try memory cache first
            if text_hash in self.memory_cache:
                embeddings.append(self.memory_cache[text_hash])
                continue
            
            # Try disk cache
            cached_embedding = self._load_from_cache(text_hash)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
                # Store in memory cache
                if len(self.memory_cache) < self.memory_cache_size:
                    self.memory_cache[text_hash] = cached_embedding
            else:
                # Need to generate embedding
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)  # Placeholder
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            logger.debug(f"Generating {len(uncached_texts)} new embeddings")
            new_embeddings = self.provider.embed_texts(uncached_texts)
            
            # Store new embeddings in cache and result list
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                
                # Cache the new embedding
                text_hash = self._hash_text(texts[idx])
                self._save_to_cache(text_hash, embedding)
                
                # Store in memory cache
                if len(self.memory_cache) < self.memory_cache_size:
                    self.memory_cache[text_hash] = embedding
        
        return np.array(embeddings, dtype=np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.embed_texts([query])[0]
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text caching."""
        # Include model info in hash to avoid conflicts
        content = f"{self.config.model.embedding_model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load_from_cache(self, text_hash: str) -> Optional[np.ndarray]:
        """Load embedding from disk cache."""
        if not self.config.cache.enable_embedding_cache:
            return None
        
        cache_file = self.cache_dir / f"{text_hash}.npy"
        try:
            if cache_file.exists():
                return np.load(cache_file)
        except Exception as e:
            logger.warning(f"Failed to load cached embedding {text_hash}: {e}")
        
        return None
    
    def _save_to_cache(self, text_hash: str, embedding: np.ndarray):
        """Save embedding to disk cache."""
        if not self.config.cache.enable_embedding_cache:
            return
        
        cache_file = self.cache_dir / f"{text_hash}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"Failed to cache embedding {text_hash}: {e}")


class BM25Searcher:
    """BM25-based keyword searcher."""
    
    def __init__(self, texts: List[str], metadata: List[Dict[str, Any]]):
        self.texts = texts
        self.metadata = metadata
        
        # Tokenize texts for BM25
        tokenized_texts = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # Build vocabulary for fallback
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            stop_words=None,  # Keep Korean-specific handling
            token_pattern=r'[가-힣a-zA-Z0-9]+',
            ngram_range=(1, 2)
        )
        
        try:
            self.tfidf.fit(texts)
            self.tfidf_matrix = self.tfidf.transform(texts)
        except Exception as e:
            logger.warning(f"TF-IDF fallback setup failed: {e}")
            self.tfidf_matrix = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        import re
        
        # Basic Korean/English tokenization
        tokens = re.findall(r'[가-힣]+|[a-zA-Z0-9]+', text.lower())
        
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) >= 2]
        
        return tokens
    
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search using BM25 scoring."""
        tokenized_query = self._tokenize(query)
        
        if not tokenized_query:
            return []
        
        try:
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for rank, idx in enumerate(top_indices):
                if scores[idx] > 0:  # Only include positive scores
                    result = SearchResult(
                        text=self.texts[idx],
                        source_id=self.metadata[idx].get('source_id', f'doc_{idx}'),
                        score=float(scores[idx]),
                        rank=rank,
                        metadata=self.metadata[idx],
                        snippet=self._extract_snippet(self.texts[idx], query)
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            # Fallback to TF-IDF if available
            return self._tfidf_fallback(query, k)
    
    def _tfidf_fallback(self, query: str, k: int) -> List[SearchResult]:
        """Fallback to TF-IDF search."""
        if self.tfidf_matrix is None:
            return []
        
        try:
            query_vec = self.tfidf.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for rank, idx in enumerate(top_indices):
                if similarities[idx] > 0.1:  # Minimum threshold
                    result = SearchResult(
                        text=self.texts[idx],
                        source_id=self.metadata[idx].get('source_id', f'doc_{idx}'),
                        score=float(similarities[idx]),
                        rank=rank,
                        metadata=self.metadata[idx],
                        snippet=self._extract_snippet(self.texts[idx], query)
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF fallback failed: {e}")
            return []
    
    def _extract_snippet(self, text: str, query: str, max_length: int = 200) -> str:
        """Extract relevant snippet from text."""
        query_terms = set(self._tokenize(query))
        
        if not query_terms:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Find best sentence/paragraph containing query terms
        sentences = text.split('.')
        best_sentence = ""
        max_matches = 0
        
        for sentence in sentences:
            sentence_tokens = set(self._tokenize(sentence))
            matches = len(query_terms.intersection(sentence_tokens))
            
            if matches > max_matches:
                max_matches = matches
                best_sentence = sentence.strip()
        
        if best_sentence and max_matches > 0:
            if len(best_sentence) > max_length:
                return best_sentence[:max_length] + "..."
            return best_sentence
        
        # Fallback to beginning of text
        return text[:max_length] + "..." if len(text) > max_length else text


class HybridRetriever:
    """Hybrid retriever combining FAISS and BM25 with RRF reranking."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.embedder = CachedEmbedder(self.config)
        self.index_manager = get_index_manager()
        
        # RRF parameters
        self.rrf_k = self.config.retrieval.rrf_k
        self.rrf_weights = self.config.retrieval.rrf_weights
    
    def search(
        self,
        query: str,
        index_name: str,
        k: int = None,
        use_mmr: bool = True,
        mmr_diversity_threshold: float = None
    ) -> Optional[HybridSearchResults]:
        """
        Perform hybrid search combining FAISS and BM25.
        
        Args:
            query: Search query
            index_name: Index to search in
            k: Number of results to return
            use_mmr: Whether to apply MMR for diversity
            mmr_diversity_threshold: MMR threshold (default from config)
            
        Returns:
            HybridSearchResults or None if index not available
        """
        start_time = time.time()
        
        # Get configuration values
        k = k or self.config.retrieval.faiss_k
        mmr_threshold = mmr_diversity_threshold or self.config.retrieval.mmr_diversity_threshold
        
        # Get index and metadata
        index_data = self.index_manager.get_index(index_name)
        if index_data is None:
            logger.error(f"Index {index_name} not available")
            return None
        
        faiss_index, metadata = index_data
        texts = metadata.get('texts', [])
        text_metadata = metadata.get('metadata', [])
        
        if not texts:
            logger.warning(f"No texts found in index {index_name}")
            return HybridSearchResults([], [], [], 0.0, 0.0, 0.0, 0.0)
        
        # FAISS semantic search
        faiss_start = time.time()
        faiss_results = self._faiss_search(query, faiss_index, texts, text_metadata, k)
        faiss_time = (time.time() - faiss_start) * 1000
        
        # BM25 keyword search
        bm25_start = time.time()
        bm25_results = self._bm25_search(query, texts, text_metadata, k)
        bm25_time = (time.time() - bm25_start) * 1000
        
        # RRF reranking
        rerank_start = time.time()
        combined_results = self._rrf_rerank(faiss_results, bm25_results, k)
        rerank_time = (time.time() - rerank_start) * 1000
        
        # Apply MMR for diversity if requested
        if use_mmr and len(combined_results) > 1:
            combined_results = self._apply_mmr(
                combined_results, query, mmr_threshold, k
            )
        
        total_time = (time.time() - start_time) * 1000
        
        return HybridSearchResults(
            results=combined_results[:k],
            faiss_results=faiss_results,
            bm25_results=bm25_results,
            total_time_ms=total_time,
            faiss_time_ms=faiss_time,
            bm25_time_ms=bm25_time,
            rerank_time_ms=rerank_time
        )
    
    def _faiss_search(
        self,
        query: str,
        faiss_index: faiss.Index,
        texts: List[str],
        metadata: List[Dict[str, Any]],
        k: int
    ) -> List[SearchResult]:
        """Perform FAISS semantic search."""
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query)
            query_vector = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search FAISS index
            scores, indices = faiss_index.search(query_vector, k)
            
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx >= len(texts) or score < 0:  # Invalid index or negative score
                    continue
                
                result = SearchResult(
                    text=texts[idx],
                    source_id=metadata[idx].get('source_id', f'doc_{idx}'),
                    score=float(score),
                    rank=rank,
                    metadata=metadata[idx],
                    snippet=self._extract_faiss_snippet(texts[idx], query)
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def _bm25_search(
        self,
        query: str,
        texts: List[str],
        metadata: List[Dict[str, Any]],
        k: int
    ) -> List[SearchResult]:
        """Perform BM25 keyword search."""
        try:
            bm25_searcher = BM25Searcher(texts, metadata)
            return bm25_searcher.search(query, k)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def _rrf_rerank(
        self,
        faiss_results: List[SearchResult],
        bm25_results: List[SearchResult],
        k: int
    ) -> List[SearchResult]:
        """Apply Reciprocal Rank Fusion to combine results."""
        # Create mapping from source_id to result for deduplication
        combined_scores = {}
        
        # Process FAISS results
        for result in faiss_results:
            rrf_score = self.rrf_weights["faiss"] / (self.rrf_k + result.rank + 1)
            combined_scores[result.source_id] = {
                'result': result,
                'score': rrf_score,
                'faiss_rank': result.rank + 1,
                'bm25_rank': None
            }
        
        # Process BM25 results
        for result in bm25_results:
            rrf_score = self.rrf_weights["bm25"] / (self.rrf_k + result.rank + 1)
            
            if result.source_id in combined_scores:
                # Combine with existing FAISS score
                combined_scores[result.source_id]['score'] += rrf_score
                combined_scores[result.source_id]['bm25_rank'] = result.rank + 1
            else:
                # New result from BM25 only
                combined_scores[result.source_id] = {
                    'result': result,
                    'score': rrf_score,
                    'faiss_rank': None,
                    'bm25_rank': result.rank + 1
                }
        
        # Sort by combined RRF score
        sorted_items = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # Create reranked results
        reranked_results = []
        for rank, (source_id, item) in enumerate(sorted_items[:k]):
            result = item['result']
            result.score = item['score']  # Update with RRF score
            result.rank = rank
            result.metadata['faiss_rank'] = item['faiss_rank']
            result.metadata['bm25_rank'] = item['bm25_rank']
            reranked_results.append(result)
        
        return reranked_results
    
    def _apply_mmr(
        self,
        results: List[SearchResult],
        query: str,
        diversity_threshold: float,
        k: int
    ) -> List[SearchResult]:
        """Apply Maximal Marginal Relevance for diversity."""
        if len(results) <= 1:
            return results
        
        try:
            # Get embeddings for all result texts
            result_texts = [r.text for r in results]
            embeddings = self.embedder.embed_texts(result_texts)
            query_embedding = self.embedder.embed_query(query)
            
            selected_results = []
            remaining_indices = list(range(len(results)))
            
            # Select first result (highest relevance)
            if remaining_indices:
                best_idx = remaining_indices[0]
                selected_results.append(results[best_idx])
                remaining_indices.remove(best_idx)
            
            # Iteratively select diverse results
            while remaining_indices and len(selected_results) < k:
                best_idx = None
                best_mmr_score = -float('inf')
                
                for idx in remaining_indices:
                    # Relevance to query
                    relevance = cosine_similarity(
                        embeddings[idx].reshape(1, -1),
                        query_embedding.reshape(1, -1)
                    )[0, 0]
                    
                    # Max similarity to already selected documents
                    max_sim = 0
                    for selected_result in selected_results:
                        selected_idx = next(
                            i for i, r in enumerate(results) if r.source_id == selected_result.source_id
                        )
                        sim = cosine_similarity(
                            embeddings[idx].reshape(1, -1),
                            embeddings[selected_idx].reshape(1, -1)
                        )[0, 0]
                        max_sim = max(max_sim, sim)
                    
                    # MMR score: balance relevance and diversity
                    mmr_score = (1 - diversity_threshold) * relevance - diversity_threshold * max_sim
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_idx = idx
                
                if best_idx is not None:
                    selected_results.append(results[best_idx])
                    remaining_indices.remove(best_idx)
                else:
                    break
            
            # Update ranks
            for rank, result in enumerate(selected_results):
                result.rank = rank
            
            return selected_results
            
        except Exception as e:
            logger.error(f"MMR application failed: {e}")
            return results  # Return original results on failure
    
    def _extract_faiss_snippet(self, text: str, query: str, max_length: int = 200) -> str:
        """Extract relevant snippet for FAISS results."""
        # Simple snippet extraction for semantic matches
        if len(text) <= max_length:
            return text
        
        # Try to find best paragraph/sentence
        sentences = text.split('.')
        if sentences:
            best_sentence = sentences[0].strip()
            if len(best_sentence) > max_length:
                return best_sentence[:max_length] + "..."
            return best_sentence
        
        return text[:max_length] + "..."
    
    def create_citations_from_results(
        self,
        results: List[SearchResult],
        max_citations: int = 3
    ) -> List[Citation]:
        """Convert search results to Citation objects."""
        citations = []
        
        for i, result in enumerate(results[:max_citations]):
            citation = Citation(
                source_id=result.source_id,
                snippet=result.snippet,
                page_number=result.metadata.get('page_number'),
                section=result.metadata.get('section'),
                confidence_score=min(result.score, 1.0)  # Normalize score
            )
            citations.append(citation)
        
        return citations


# Global instance
_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Get the global HybridRetriever instance."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever
