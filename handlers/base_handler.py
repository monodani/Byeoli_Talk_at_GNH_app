#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - base_handler 기반 클래스

모든 핸들러의 공통 기능을 제공하는 추상 베이스 클래스:
- 하이브리드 검색 (FAISS + BM25 + RRF)
- 컨피던스 계산 (단순화된 버전)
- Citation 추출 (200자 snippet)
- 스트리밍 응답 (50토큰 단위)
- 표준 인터페이스 (QueryRequest → HandlerResponse)
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

# 외부 라이브러리
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

# 로깅 설정
logger = logging.getLogger(__name__)


class base_handler(ABC):
    """
    모든 핸들러의 기반 클래스
    
    주요 기능:
    - 하이브리드 검색 (FAISS + BM25 + RRF)
    - 컨피던스 기반 파이프라인
    - Citation 자동 추출
    - 스트리밍 응답 지원
    """
    
    def __init__(self, domain: str, index_name: str, confidence_threshold: float):
        """
        base_Handler 초기화
        
        Args:
            domain: 도메인 이름 (예: "satisfaction")
            index_name: 벡터스토어 인덱스 이름
            confidence_threshold: 컨피던스 임계값 (θ)
        """
        self.domain = domain
        self.index_name = index_name
        self.confidence_threshold = confidence_threshold
        
        # 벡터스토어 경로 설정
        self.vectorstore_dir = config.ROOT_DIR / "vectorstores" / f"vectorstore_{domain}" if domain != "satisfaction" else config.ROOT_DIR / "vectorstores" / "vectorstore_unified_satisfaction"
        
        # 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings()
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini",
            streaming=True
        )
        
        # 벡터스토어 및 BM25 (지연 로딩)
        self.vectorstore = None
        self.bm25 = None
        self.documents = None
        
        logger.info(f"✨ {domain.upper()} Handler 초기화 완료 (θ={confidence_threshold})")
    
    def _load_vectorstore(self) -> bool:
        """벡터스토어 로드 (지연 로딩)"""
        if self.vectorstore is not None:
            return True
            
        try:
            logger.info(f"📚 {self.domain} 벡터스토어 로드 중...")
            
            self.vectorstore = FAISS.load_local(
                folder_path=str(self.vectorstore_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
                index_name=self.index_name
            )
            
            # 문서 메타데이터 추출 (BM25용)
            self.documents = []
            for i in range(len(self.vectorstore.docstore._dict)):
                doc = self.vectorstore.docstore._dict.get(str(i))
                if doc:
                    self.documents.append(doc.page_content)
            
            # BM25 인덱스 구축
            if self.documents:
                tokenized_docs = [doc.split() for doc in self.documents]
                self.bm25 = BM25Okapi(tokenized_docs)
                logger.info(f"✅ BM25 인덱스 구축 완료: {len(self.documents)}개 문서")
            
            logger.info(f"✅ {self.domain} 벡터스토어 로드 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ {self.domain} 벡터스토어 로드 실패: {e}")
            return False
    
    def hybrid_search(self, query: str, k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        하이브리드 검색 (FAISS + BM25 + RRF)
        
        Args:
            query: 검색 쿼리
            k: 검색할 문서 수
            
        Returns:
            List of (text, score, metadata) tuples
        """
        if not self._load_vectorstore():
            return []
        
        try:
            # 1. FAISS 검색
            faiss_results = self.vectorstore.similarity_search_with_score(query, k=k)
            faiss_docs = [(doc.page_content, score, doc.metadata) for doc, score in faiss_results]
            
            # 2. BM25 검색
            bm25_docs = []
            if self.bm25:
                tokenized_query = query.split()
                bm25_scores = self.bm25.get_scores(tokenized_query)
                
                # 상위 k개 BM25 결과 선택
                top_indices = np.argsort(bm25_scores)[-k:][::-1]
                for idx in top_indices:
                    if idx < len(self.documents):
                        # 메타데이터 찾기
                        doc_id = str(idx)
                        metadata = {}
                        if doc_id in self.vectorstore.docstore._dict:
                            metadata = self.vectorstore.docstore._dict[doc_id].metadata
                        
                        bm25_docs.append((self.documents[idx], bm25_scores[idx], metadata))
            
            # 3. RRF 재랭킹 (단순화)
            # FAISS 0.6 + BM25 0.4 가중치
            combined_results = {}
            
            # FAISS 결과 처리 (점수 정규화)
            max_faiss_score = max([score for _, score, _ in faiss_docs]) if faiss_docs else 1.0
            for text, score, metadata in faiss_docs:
                normalized_score = score / max_faiss_score
                text_hash = hashlib.md5(text.encode()).hexdigest()
                combined_results[text_hash] = {
                    'text': text,
                    'metadata': metadata,
                    'faiss_score': normalized_score,
                    'bm25_score': 0.0
                }
            
            # BM25 결과 처리 (점수 정규화)
            max_bm25_score = max([score for _, score, _ in bm25_docs]) if bm25_docs else 1.0
            for text, score, metadata in bm25_docs:
                normalized_score = score / max_bm25_score if max_bm25_score > 0 else 0
                text_hash = hashlib.md5(text.encode()).hexdigest()
                
                if text_hash in combined_results:
                    combined_results[text_hash]['bm25_score'] = normalized_score
                else:
                    combined_results[text_hash] = {
                        'text': text,
                        'metadata': metadata,
                        'faiss_score': 0.0,
                        'bm25_score': normalized_score
                    }
            
            # 최종 점수 계산 및 정렬
            final_results = []
            for item in combined_results.values():
                final_score = (item['faiss_score'] * 0.6) + (item['bm25_score'] * 0.4)
                final_results.append((item['text'], final_score, item['metadata']))
            
            # 점수 기준 내림차순 정렬
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"🔍 하이브리드 검색 완료: {len(final_results)}개 결과")
            return final_results[:k]
            
        except Exception as e:
            logger.error(f"❌ 하이브리드 검색 실패: {e}")
            return []
    
    def calculate_confidence(self, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> float:
        """
        컨피던스 계산 (단순화된 버전)
        
        Args:
            search_results: 검색 결과 리스트
            
        Returns:
            컨피던스 점수 (0.0 ~ 1.0)
        """
        if not search_results:
            return 0.0
        
        # 상위 3개 결과의 평균 점수 사용
        top_scores = [score for _, score, _ in search_results[:3]]
        avg_score = sum(top_scores) / len(top_scores)
        
        # 0-1 범위로 정규화
        confidence = min(max(avg_score, 0.0), 1.0)
        
        logger.debug(f"📊 컨피던스 계산: {confidence:.3f} (상위 {len(top_scores)}개 평균)")
        return confidence
    
    def extract_citations(self, search_results: List[Tuple[str, float, Dict[str, Any]]], max_citations: int = 3) -> List[Citation]:
        """
        Citation 추출 (200자 snippet)
        
        Args:
            search_results: 검색 결과
            max_citations: 최대 Citation 수
            
        Returns:
            Citation 객체 리스트
        """
        citations = []
        
        for i, (text, score, metadata) in enumerate(search_results[:max_citations]):
            # source_id 생성
            source_file = metadata.get('source_file', 'unknown')
            if 'page_number' in metadata:
                source_id = f"{self.domain}/{source_file}#page_{metadata['page_number']}"
            elif 'row_number' in metadata:
                source_id = f"{self.domain}/{source_file}#row_{metadata['row_number']}"
            else:
                source_id = f"{self.domain}/{source_file}#chunk_{i}"
            
            # snippet 생성 (200자 + 문장 단위 절단)
            snippet = text[:200]
            if len(text) > 200:
                # 마지막 문장 경계에서 자르기
                last_period = snippet.rfind('.')
                last_question = snippet.rfind('?')
                last_exclamation = snippet.rfind('!')
                
                cut_point = max(last_period, last_question, last_exclamation)
                if cut_point > 100:  # 너무 짧지 않도록
                    snippet = snippet[:cut_point + 1]
                else:
                    snippet += "..."
            
            citation = Citation(
                source_id=source_id,
                snippet=snippet
            )
            citations.append(citation)
        
        logger.info(f"📝 Citation 추출 완료: {len(citations)}건")
        return citations
    
    def streaming_response(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """
        스트리밍 응답 생성 (50토큰 단위)
        
        Args:
            messages: LLM 메시지 리스트
            
        Yields:
            응답 토큰들
        """
        try:
            token_buffer = ""
            token_count = 0
            
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    token_buffer += chunk.content
                    token_count += 1
                    
                    # 50토큰마다 또는 문장 끝에서 flush
                    if token_count >= 50 or chunk.content in '.!?':
                        if token_buffer.strip():
                            yield token_buffer
                            token_buffer = ""
                            token_count = 0
            
            # 남은 버퍼 출력
            if token_buffer.strip():
                yield token_buffer
                
        except Exception as e:
            logger.error(f"❌ 스트리밍 응답 생성 실패: {e}")
            yield f"응답 생성 중 오류가 발생했습니다: {str(e)}"
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """도메인별 시스템 프롬프트 반환 (서브클래스에서 구현)"""
        pass
    
    @abstractmethod
    def format_context(self, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """검색 결과를 컨텍스트로 포맷 (서브클래스에서 구현)"""
        pass
    
    def handle(self, request: QueryRequest) -> HandlerResponse:
        """
        표준 핸들러 인터페이스
        
        Args:
            request: QueryRequest 객체
            
        Returns:
            HandlerResponse 객체
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎯 {self.domain} 핸들러 처리 시작: {request.text[:50]}...")
            
            # 1. 하이브리드 검색
            search_results = self.hybrid_search(request.text, k=10)
            
            # 2. 컨피던스 계산
            confidence = self.calculate_confidence(search_results)
            
            # 3. 컨피던스 체크
            if confidence < self.confidence_threshold:
                logger.warning(f"⚠️ 낮은 컨피던스: {confidence:.3f} < {self.confidence_threshold}")
                # TODO: k 확장 검색 또는 재질문 로직 구현
            
            # 4. Citation 추출
            citations = self.extract_citations(search_results)
            
            # 5. 응답 생성
            system_prompt = self.get_system_prompt()
            context = self.format_context(search_results)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"컨텍스트:\n{context}\n\n질문: {request.text}"}
            ]
            
            # 스트리밍이 아닌 일반 응답으로 구현 (단순화)
            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # 6. HandlerResponse 생성
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            handler_response = HandlerResponse(
                answer=answer,
                citations=citations,
                confidence=confidence,
                handler_id=self.domain,
                elapsed_ms=elapsed_ms
            )
            
            logger.info(f"✅ {self.domain} 핸들러 처리 완료 ({elapsed_ms}ms, confidence={confidence:.3f})")
            return handler_response
            
        except Exception as e:
            logger.error(f"❌ {self.domain} 핸들러 처리 실패: {e}")
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # 에러 응답
            return HandlerResponse(
                answer=f"죄송합니다. {self.domain} 정보 처리 중 오류가 발생했습니다.",
                citations=[],
                confidence=0.0,
                handler_id=self.domain,
                elapsed_ms=elapsed_ms
            )
