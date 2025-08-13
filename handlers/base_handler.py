#!/usr/bin/env python3
"""
벼리톡@경상남도인재개발원 (경상남도인재개발원 RAG 챗봇) - base_handler 기반 클래스 (OpenAI 호환성 수정)

모든 핸들러의 공통 기능을 제공하는 추상 베이스 클래스:
- IndexManager 싱글톤을 활용한 중앙집중식 벡터스토어 관리
- 하이브리드 검색 (FAISS + BM25 + RRF)
- 컨피던스 계산 (단순화된 버전)
- Citation 추출 (200자 snippet)
- 스트리밍 응답 (50토큰 단위)
- 표준 인터페이스 (QueryRequest → HandlerResponse)

🚨 주요 수정사항:
✅ OpenAIEmbeddings 초기화 방식 수정 (호환성 문제 해결)
✅ Graceful Degradation 강화
✅ 에러 처리 개선
"""

import logging
import time
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from datetime import datetime

# 프로젝트 모듈 임포트
from utils.config import config
from utils.contracts import QueryRequest, HandlerResponse, Citation
from utils.textifier import TextChunk
from utils.index_manager import get_index_manager

# 외부 라이브러리
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

# 로깅 설정
logger = logging.getLogger(__name__)


class base_handler(ABC):
    """
    모든 핸들러의 기반 클래스 (OpenAI 호환성 수정)
    
    주요 기능:
    - IndexManager를 통한 중앙집중식 벡터스토어 관리
    - 하이브리드 검색 (FAISS + BM25 + RRF)
    - 컨피던스 기반 파이프라인
    - Citation 자동 추출
    - 스트리밍 응답 지원
    """
    
    def __init__(self, domain: str, index_name: str, confidence_threshold: float):
        """
        base_handler 초기화 (OpenAI 호환성 수정)
        
        Args:
            domain: 도메인 이름 (예: "satisfaction")
            index_name: 벡터스토어 인덱스 이름
            confidence_threshold: 컨피던스 임계값 (θ)
        """
        self.domain = domain
        self.index_name = index_name
        self.confidence_threshold = confidence_threshold
        
        # IndexManager 싱글톤 참조
        self.index_manager = get_index_manager()
        
        # ✅ OpenAIEmbeddings 안전한 초기화 (호환성 수정)
        self.embeddings = self._init_embeddings()
        
        # LLM 초기화
        self.llm = self._init_llm()
        
        logger.info(f"✨ {domain.upper()} Handler 초기화 완료 (θ={confidence_threshold}, IndexManager 통합)")
    
    def _init_embeddings(self) -> Optional[Any]:
        """
        OpenAIEmbeddings 안전한 초기화 (호환성 수정)
        """
        try:
            from langchain_openai import OpenAIEmbeddings
            
            api_key = config.OPENAI_API_KEY
            if not api_key:
                logger.warning(f"⚠️ {self.domain} 핸들러: OPENAI_API_KEY가 설정되지 않아 임베딩을 사용할 수 없습니다.")
                return None
            
            embeddings = OpenAIEmbeddings(
                api_key=api_key,
                model=config.EMBEDDING_MODEL
            )
            
            logger.debug(f"✅ {self.domain} 핸들러 OpenAIEmbeddings 초기화 완료")
            return embeddings
            
        except ImportError as e:
            logger.error(f"❌ {self.domain} 핸들러: LangChain OpenAI 라이브러리 임포트 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ {self.domain} 핸들러: OpenAIEmbeddings 초기화 실패: {e}")
            return None
    
    def _init_llm(self) -> Optional[ChatOpenAI]:
        """
        ChatOpenAI LLM 안전한 초기화
        """
        try:
            api_key = config.OPENAI_API_KEY
            if not api_key:
                logger.warning(f"⚠️ {self.domain} 핸들러: OPENAI_API_KEY가 설정되지 않아 LLM을 사용할 수 없습니다.")
                return None
            
            llm = ChatOpenAI(
                api_key=api_key,
                temperature=0.1,
                model="gpt-4o-mini",
                streaming=True
            )
            
            logger.debug(f"✅ {self.domain} 핸들러 ChatOpenAI 초기화 완료")
            return llm
            
        except Exception as e:
            logger.error(f"❌ {self.domain} 핸들러: ChatOpenAI 초기화 실패: {e}")
            return None
    
    def _get_vectorstore(self) -> Optional[FAISS]:
        """
        IndexManager를 통해 벡터스토어 획득 (중앙집중식)
        
        Returns:
            Optional[FAISS]: 벡터스토어 인스턴스 또는 None
        """
        try:
            vectorstore = self.index_manager.get_vectorstore(self.domain)
            if vectorstore is None:
                logger.warning(f"⚠️ {self.domain} 벡터스토어를 찾을 수 없습니다")
                return None
            return vectorstore
        except Exception as e:
            logger.error(f"❌ {self.domain} 벡터스토어 획득 실패: {e}")
            return None
    
    def _get_bm25(self) -> Optional[BM25Okapi]:
        """
        IndexManager를 통해 BM25 인덱스 획득
        
        Returns:
            Optional[BM25Okapi]: BM25 인덱스 또는 None
        """
        try:
            bm25 = self.index_manager.get_bm25(self.domain)
            if bm25 is None:
                logger.warning(f"⚠️ {self.domain} BM25 인덱스를 찾을 수 없습니다")
                return None
            return bm25
        except Exception as e:
            logger.error(f"❌ {self.domain} BM25 인덱스 획득 실패: {e}")
            return None
    
    def _get_documents(self) -> List[TextChunk]:
        """
        IndexManager를 통해 문서 리스트 획득
        
        Returns:
            List[TextChunk]: 문서 리스트
        """
        try:
            documents = self.index_manager.get_documents(self.domain)
            return documents
        except Exception as e:
            logger.error(f"❌ {self.domain} 문서 리스트 획득 실패: {e}")
            return []
    
    def _hybrid_search(self, query: str, k: int = 5) -> List[Tuple[TextChunk, float]]:
        """
        하이브리드 검색 (FAISS + BM25 + RRF) - 에러 처리 강화
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            List[Tuple[TextChunk, float]]: (문서, 점수) 튜플 리스트
        """
        try:
            vectorstore = self._get_vectorstore()
            bm25 = self._get_bm25()
            documents = self._get_documents()
            
            # 사용 가능한 검색 방법 확인
            faiss_available = vectorstore is not None and self.embeddings is not None
            bm25_available = bm25 is not None and len(documents) > 0
            
            if not faiss_available and not bm25_available:
                logger.warning(f"⚠️ {self.domain}: 검색 인덱스가 없어 빈 결과를 반환합니다.")
                return []
            
            # FAISS 검색 (의미론적 검색)
            faiss_results = []
            if faiss_available:
                try:
                    faiss_docs = vectorstore.similarity_search_with_score(query, k=k*2)
                    for doc, score in faiss_docs:
                        # LangChain Document → TextChunk 변환
                        text_chunk = TextChunk(
                            text=doc.page_content,
                            source_id=doc.metadata.get('source_id', 'unknown'),
                            chunk_id=doc.metadata.get('chunk_id', 0),
                            metadata=doc.metadata
                        )
                        faiss_results.append((text_chunk, 1.0 - score))  # 거리를 유사도로 변환
                    logger.debug(f"✅ {self.domain} FAISS 검색: {len(faiss_results)}개 결과")
                except Exception as e:
                    logger.warning(f"⚠️ {self.domain} FAISS 검색 실패: {e}")
                    faiss_results = []
            
            # BM25 검색 (키워드 검색)
            bm25_results = []
            if bm25_available:
                try:
                    # 쿼리 토큰화
                    query_tokens = query.lower().split()
                    
                    # BM25 점수 계산
                    bm25_scores = bm25.get_scores(query_tokens)
                    
                    # 점수 기준 정렬 및 상위 k*2 선택
                    scored_docs = list(zip(documents, bm25_scores))
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    
                    # 슬라이싱 오류 수정: min 사용
                    top_k = min(k*2, len(scored_docs))
                    bm25_results = scored_docs[:top_k]
                    
                    logger.debug(f"✅ {self.domain} BM25 검색: {len(bm25_results)}개 결과")
                except Exception as e:
                    logger.warning(f"⚠️ {self.domain} BM25 검색 실패: {e}")
                    bm25_results = []
            
            # RRF (Reciprocal Rank Fusion) 결합
            combined_results = self._rrf_combine(faiss_results, bm25_results, k=k)
            
            logger.debug(f"🔍 {self.domain} 하이브리드 검색 완료: {len(combined_results)}개 최종 결과")
            return combined_results
            
        except Exception as e:
            logger.error(f"❌ {self.domain} 하이브리드 검색 실패: {e}")
            return []
    
    def _rrf_combine(self, faiss_results: List[Tuple[TextChunk, float]], 
                     bm25_results: List[Tuple[TextChunk, float]], 
                     k: int = 60) -> List[Tuple[TextChunk, float]]:
        """
        RRF (Reciprocal Rank Fusion)로 검색 결과 결합
        """
        try:
            doc_scores = {}
            
            # FAISS 결과 처리
            for rank, (doc, score) in enumerate(faiss_results, 1):
                # ✅ getattr로 안전하게 chunk_id 접근
                doc_id = f"{doc.source_id}_{getattr(doc, 'chunk_id', rank)}"
                rrf_score = 1.0 / (k + rank)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {'doc': doc, 'score': 0}
                doc_scores[doc_id]['score'] += rrf_score * 0.6  # FAISS 가중치 60%
            
            # BM25 결과 처리
            for rank, (doc, score) in enumerate(bm25_results, 1):
                # ✅ getattr로 안전하게 chunk_id 접근
                doc_id = f"{doc.source_id}_{getattr(doc, 'chunk_id', rank)}"
                rrf_score = 1.0 / (k + rank)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {'doc': doc, 'score': 0}
                doc_scores[doc_id]['score'] += rrf_score * 0.4  # BM25 가중치 40%
            
            # 점수 기준 정렬
            if doc_scores:
                combined = [(info['doc'], info['score']) for info in doc_scores.values()]
                combined.sort(key=lambda x: x[1], reverse=True)
                return combined[:min(k, len(combined))]
            else:
                return []
                
        except Exception as e:
            logger.error(f"❌ RRF 결합 실패: {e}")
            # 폴백: FAISS 결과 우선 반환
            return faiss_results[:k] if faiss_results else bm25_results[:k]

    
    def _calculate_confidence(self, query: str, retrieved_docs: List[Tuple[TextChunk, float]], 
                            response: str) -> float:
        """
        컨피던스 점수 계산 (단순화된 버전)
        """
        try:
            if not retrieved_docs:
                return 0.0
            
            # 기본 점수: 검색된 문서의 평균 유사도
            avg_similarity = sum(score for _, score in retrieved_docs) / len(retrieved_docs)
            
            # 쿼리 길이 보정 (너무 짧거나 긴 쿼리는 신뢰도 감소)
            query_length_factor = min(1.0, len(query.split()) / 10.0)
            
            # 응답 길이 보정 (너무 짧은 응답은 신뢰도 감소)
            response_length_factor = min(1.0, len(response.split()) / 20.0)
            
            confidence = avg_similarity * query_length_factor * response_length_factor
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.warning(f"⚠️ 컨피던스 계산 실패: {e}")
            return 0.5  # 기본값
    
    def _extract_citations(self, retrieved_docs: List[Tuple[TextChunk, float]]) -> List[Citation]:
        """
        Citation 자동 추출 (200자 snippet)
        """
        citations = []
        
        try:
            for doc, score in retrieved_docs:
                # snippet 길이 제한 (200자)
                snippet = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
                
                citation = Citation(
                    source_id=doc.source_id,
                    text=snippet,  # ✅ text 필드 추가
                    relevance_score=score,  # ✅ relevance_score 필드 추가
                    snippet=snippet,
                    metadata=doc.metadata
                )
                citations.append(citation)
                
        except Exception as e:
            logger.warning(f"⚠️ Citation 추출 실패: {e}")
        
        return citations
    
    def _stream_response(self, prompt: str) -> Iterator[str]:
        """
        스트리밍 응답 생성 (50토큰 단위) - 에러 처리 강화
        """
        try:
            if not self.llm:
                logger.warning(f"⚠️ {self.domain}: LLM이 초기화되지 않아 기본 응답을 반환합니다.")
                yield "죄송합니다. 현재 AI 모델이 사용할 수 없어 정확한 답변을 드릴 수 없습니다."
                return
            
            response_stream = self.llm.stream(prompt)
            buffer = ""
            
            for chunk in response_stream:
                if hasattr(chunk, 'content'):
                    buffer += chunk.content
                    
                    # 50토큰 단위로 방출
                    if len(buffer.split()) >= 50:
                        yield buffer
                        buffer = ""
            
            # 남은 내용 방출
            if buffer.strip():
                yield buffer
                
        except Exception as e:
            logger.error(f"❌ {self.domain} 스트리밍 응답 실패: {e}")
            yield f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"
    
    @abstractmethod
    def _generate_prompt(self, query: str, retrieved_docs: List[Tuple[TextChunk, float]]) -> str:
        """
        도메인별 프롬프트 생성 (하위 클래스에서 구현)
        """
        pass
    
    def handle(self, request: QueryRequest) -> HandlerResponse:
        """
        표준 핸들러 처리 로직 (모든 하위 핸들러에서 공통 사용)
        """
        start_time = time.time()
        
        try:
            # QueryRequest에서 쿼리 텍스트 추출
            query = getattr(request, 'query', None) or getattr(request, 'text', '')
            
            # 1. 하이브리드 검색 수행
            retrieved_docs = self._hybrid_search(query, k=5)
            
            # 2. 프롬프트 생성 (하위 클래스에서 구현)
            prompt = self._generate_prompt(query, retrieved_docs)
            
            # 3. LLM 응답 생성
            if not self.llm:
                logger.warning(f"⚠️ {self.domain}: LLM이 초기화되지 않아 fallback 응답 사용")
                return self._fallback_response(query, "LLM 초기화 실패")
            
            # 스트리밍 응답 수집
            response_chunks = list(self._stream_response(prompt))
            answer = "".join(response_chunks)
            
            # 4. 컨피던스 계산
            confidence = self._calculate_confidence(query, retrieved_docs, answer)
            
            # 5. Citation 추출
            citations = self._extract_citations(retrieved_docs)
            
            # 6. 응답 생성
            elapsed_ms = (time.time() - start_time) * 1000
            
            return HandlerResponse(
                answer=answer,
                confidence=confidence,
                handler_id=self.domain,
                citations=citations,
                elapsed_ms=elapsed_ms,
                success=confidence >= self.confidence_threshold,
                diagnostics={
                    "handler": self.domain,
                    "search_results": len(retrieved_docs),
                    "confidence_threshold": self.confidence_threshold,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"❌ {self.domain} 핸들러 처리 실패: {e}")
            return self._fallback_response(query, str(e))

    
    def _fallback_response(self, query: str, error_msg: str = "") -> HandlerResponse:
        """
        폴백 응답 생성 (에러 상황 대응)
        """
        fallback_text = (
            f"죄송합니다. '{query}'에 대한 정확한 답변을 찾을 수 없습니다. "
            f"경상남도인재개발원 관련 질문이시라면 더 구체적으로 질문해 주시거나, "
            f"담당자에게 직접 문의해 주세요."
        )
        
        if error_msg:
            fallback_text += f"\n\n(기술적 오류: {error_msg})"
        
        return HandlerResponse(
            answer=fallback_text,
            confidence=0.1,
            handler_id=self.domain,
            citations=[],            # ✅ citations 필드 사용
            elapsed_ms=0,           # ✅ elapsed_ms 필드 추가
            success=False,          # ✅ success 필드 추가
            diagnostics={           # ✅ diagnostics 필드 사용
                "handler": self.domain,
                "fallback": True,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        )
