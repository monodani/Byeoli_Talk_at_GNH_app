#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - base_handler 기반 클래스 (IndexManager 통합 + BM25 오류 수정)

모든 핸들러의 공통 기능을 제공하는 추상 베이스 클래스:
- IndexManager 싱글톤을 활용한 중앙집중식 벡터스토어 관리
- 하이브리드 검색 (FAISS + BM25 + RRF)
- 컨피던스 계산 (단순화된 버전)
- Citation 추출 (200자 snippet)
- 스트리밍 응답 (50토큰 단위)
- 표준 인터페이스 (QueryRequest → HandlerResponse)

주요 수정사항:
✅ BM25 검색 슬라이싱 오류 수정
✅ Citation snippet 길이 제한 개선
✅ 토큰화 로직 개선
✅ 에러 처리 강화
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
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

# 로깅 설정
logger = logging.getLogger(__name__)


class base_handler(ABC):
    """
    모든 핸들러의 기반 클래스 (IndexManager 통합 + 오류 수정)
    
    주요 기능:
    - IndexManager를 통한 중앙집중식 벡터스토어 관리
    - 하이브리드 검색 (FAISS + BM25 + RRF)
    - 컨피던스 기반 파이프라인
    - Citation 자동 추출
    - 스트리밍 응답 지원
    """
    
    def __init__(self, domain: str, index_name: str, confidence_threshold: float):
        """
        base_Handler 초기화 (IndexManager 통합)
        
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
        
        # 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings()
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini",
            streaming=True
        )
        
        logger.info(f"✨ {domain.upper()} Handler 초기화 완료 (θ={confidence_threshold}, IndexManager 통합)")
    
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
        ✅ 수정: TextChunk 타입으로 반환
        IndexManager를 통해 문서 목록 획득
        
        Returns:
            List[TextChunk]: 문서 청크 목록
        """
        try:
            documents = self.index_manager.get_documents(self.domain)
            return documents if documents else []
        except Exception as e:
            logger.error(f"❌ {self.domain} 문서 목록 획득 실패: {e}")
            return []
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        ✅ 추가: 개선된 토큰화 함수
        
        Args:
            text: 토큰화할 텍스트
            
        Returns:
            List[str]: 토큰 리스트
        """
        # 한국어와 영어를 고려한 토큰화
        # 공백, 구두점 기준으로 분리
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def hybrid_search(self, query: str, k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        ✅ 수정: BM25 검색 오류 수정
        하이브리드 검색 (FAISS + BM25 + RRF) - IndexManager 활용
        
        Args:
            query: 검색 쿼리
            k: 검색할 문서 수
            
        Returns:
            List of (text, score, metadata) tuples
        """
        # IndexManager에서 벡터스토어 및 BM25 획득
        vectorstore = self._get_vectorstore()
        bm25 = self._get_bm25()
        documents = self._get_documents()  # TextChunk 리스트
        
        if not vectorstore:
            logger.warning(f"⚠️ {self.domain} 벡터스토어 없음, 빈 결과 반환")
            return []
        
        try:
            # 1. FAISS 검색
            faiss_results = vectorstore.similarity_search_with_score(query, k=k)
            faiss_docs = [(doc.page_content, score, doc.metadata) for doc, score in faiss_results]
            
            # 2. BM25 검색 (사용 가능한 경우)
            bm25_docs = []
            if bm25 and documents:
                # ✅ 개선된 토큰화
                tokenized_query = self._tokenize_text(query)
                
                # ✅ 수정: documents가 TextChunk 리스트임을 고려
                doc_texts = []
                for doc in documents:
                    if isinstance(doc, TextChunk):
                        doc_texts.append(doc.text)
                    elif isinstance(doc, str):
                        doc_texts.append(doc)
                    else:
                        doc_texts.append(str(doc))
                
                # BM25 스코어 계산
                if doc_texts:
                    try:
                        bm25_scores = bm25.get_scores(tokenized_query)
                        
                        # 상위 k개 선택
                        if len(bm25_scores) > 0:
                            # ✅ 수정: numpy 배열 인덱싱 안전하게 처리
                            top_k = min(k, len(bm25_scores))
                            top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
                            
                            for idx in top_indices:
                                if 0 <= idx < len(documents):
                                    doc = documents[idx]
                                    text = doc.text if isinstance(doc, TextChunk) else str(doc)
                                    metadata = doc.metadata if isinstance(doc, TextChunk) else {}
                                    score = float(bm25_scores[idx])
                                    
                                    bm25_docs.append((text, score, metadata))
                    except Exception as e:
                        logger.error(f"❌ {self.domain} BM25 스코어 계산 실패: {e}")
            
            # 3. RRF (Reciprocal Rank Fusion) 적용
            combined_results = self._apply_rrf(faiss_docs, bm25_docs, k=k)
            
            logger.debug(f"🔍 {self.domain} 하이브리드 검색 완료: FAISS {len(faiss_docs)}, BM25 {len(bm25_docs)}, 결합 {len(combined_results)}")
            return combined_results
            
        except Exception as e:
            logger.error(f"❌ {self.domain} 하이브리드 검색 실패: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # 에러 시 FAISS 결과만이라도 반환
            try:
                if vectorstore:
                    faiss_results = vectorstore.similarity_search_with_score(query, k=k)
                    return [(doc.page_content, score, doc.metadata) for doc, score in faiss_results]
            except:
                pass
            
            return []
    
    def _apply_rrf(self, faiss_docs: List[Tuple[str, float, Dict]], 
                   bm25_docs: List[Tuple[str, float, Dict]], k: int = 60) -> List[Tuple[str, float, Dict]]:
        """
        RRF (Reciprocal Rank Fusion) 적용
        
        Args:
            faiss_docs: FAISS 검색 결과
            bm25_docs: BM25 검색 결과
            k: RRF 파라미터 (기본값: 60)
            
        Returns:
            List: RRF 점수로 정렬된 결합 결과
        """
        doc_scores = {}
        
        # FAISS 점수 (거리 기반이므로 역순으로 랭킹)
        for rank, (text, score, metadata) in enumerate(faiss_docs):
            doc_key = text[:100]  # 텍스트 앞부분을 키로 사용
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_key] = {
                'text': text,
                'metadata': metadata,
                'rrf_score': rrf_score,
                'faiss_score': score
            }
        
        # BM25 점수 추가
        for rank, (text, score, metadata) in enumerate(bm25_docs):
            doc_key = text[:100]
            rrf_score = 1.0 / (k + rank + 1)
            
            if doc_key in doc_scores:
                # 기존 문서면 RRF 점수 합산
                doc_scores[doc_key]['rrf_score'] += rrf_score
                doc_scores[doc_key]['bm25_score'] = score
            else:
                # 새 문서면 추가
                doc_scores[doc_key] = {
                    'text': text,
                    'metadata': metadata,
                    'rrf_score': rrf_score,
                    'bm25_score': score
                }
        
        # RRF 점수 기준 정렬 후 상위 결과 반환
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
        
        result = []
        for doc in sorted_docs[:len(faiss_docs)]:  # 원래 요청한 k개만큼 반환
            result.append((doc['text'], doc['rrf_score'], doc['metadata']))
        
        return result
    
    def calculate_confidence(self, search_results: List[Tuple[str, float, Dict[str, Any]]], 
                           query: str) -> float:
        """
        ✅ 개선: 문서가 적을 때도 적절한 컨피던스 계산
        
        Args:
            search_results: 검색 결과
            query: 원본 쿼리
            
        Returns:
            float: 컨피던스 점수 (0.0 - 1.0)
        """
        if not search_results:
            return 0.0
        
        # 검색 결과 수에 따른 기본 점수 조정
        result_count = len(search_results)
        if result_count == 1:
            base_confidence = 0.3  # 결과가 1개면 기본 0.3
        elif result_count == 2:
            base_confidence = 0.4  # 결과가 2개면 기본 0.4
        else:
            base_confidence = 0.5  # 3개 이상이면 기본 0.5
        
        # 상위 결과의 평균 점수 (정규화)
        top_scores = [score for _, score, _ in search_results[:min(3, result_count)]]
        if top_scores:
            # 점수 정규화 (0-1 범위로)
            max_score = max(top_scores)
            if max_score > 0:
                normalized_scores = [s / max_score for s in top_scores]
                avg_score = sum(normalized_scores) / len(normalized_scores)
                score_boost = avg_score * 0.3  # 최대 0.3 추가
            else:
                score_boost = 0.0
        else:
            score_boost = 0.0
        
        # 문서 다양성 보너스
        sources = set()
        for _, _, metadata in search_results[:min(3, result_count)]:
            source = metadata.get('source', 'unknown')
            sources.add(source)
        diversity_bonus = min(len(sources) * 0.05, 0.15)  # 최대 0.15 보너스
        
        # 쿼리 유형 매칭 보너스
        query_lower = query.lower()
        domain_keywords = self._get_domain_keywords()
        keyword_matches = sum(1 for kw in domain_keywords if kw in query_lower)
        keyword_bonus = min(keyword_matches * 0.05, 0.15)  # 최대 0.15 보너스
        
        # 최종 컨피던스 계산
        confidence = min(base_confidence + score_boost + diversity_bonus + keyword_bonus, 1.0)
        
        logger.debug(f"🎯 {self.domain} 컨피던스: {confidence:.3f} (기본: {base_confidence:.3f}, 점수: {score_boost:.3f}, 다양성: {diversity_bonus:.3f}, 키워드: {keyword_bonus:.3f})")
        return confidence
    
    def _get_domain_keywords(self) -> List[str]:
        """도메인별 키워드 반환 (컨피던스 계산용)"""
        domain_keywords = {
            'satisfaction': ['만족도', '평가', '설문', '조사', '점수', '순위'],
            'general': ['학칙', '규정', '전결', '연락처', '담당자', '부서'],
            'menu': ['식단', '메뉴', '구내식당', '급식', '식사'],
            'cyber': ['사이버교육', '온라인교육', '나라배움터', '민간위탁'],
            'publish': ['교육계획', '훈련계획', '평가서', '계획서'],
            'notice': ['공지', '안내', '알림', '공지사항', '새소식']
        }
        return domain_keywords.get(self.domain, [])
    
    def extract_citations(self, search_results: List[Tuple[str, float, Dict[str, Any]]], 
                         max_citations: int = 3) -> List[Citation]:
        """
        ✅ 개선: Citation snippet 길이 제한 처리
        
        Args:
            search_results: 검색 결과
            max_citations: 최대 Citation 수
            
        Returns:
            List[Citation]: Citation 목록
        """
        citations = []
        
        for i, (text, score, metadata) in enumerate(search_results[:max_citations]):
            try:
                # 소스 ID 생성
                source = metadata.get('source', 'unknown')
                page = metadata.get('page', '')
                source_id = f"{source}#{page}" if page else source
                
                # ✅ 개선된 스니펫 생성 (200자 제한, 단어 단위로 자르기)
                if len(text) <= 200:
                    snippet = text
                else:
                    # 200자 근처에서 단어 경계 찾기
                    cutoff = 197  # "..." 3자 고려
                    # 공백이나 구두점에서 자르기
                    while cutoff > 0 and cutoff < len(text):
                        if text[cutoff] in ' \n\t.,;!?':
                            break
                        cutoff -= 1
                    
                    if cutoff == 0:
                        snippet = text[:197] + "..."
                    else:
                        snippet = text[:cutoff].strip() + "..."
                
                citation = Citation(
                    source_id=source_id,
                    snippet=snippet
                )
                citations.append(citation)
                
            except Exception as e:
                logger.error(f"Citation 생성 실패: {e}")
                # Citation 없이 계속 진행
                continue
        
        # Citation이 없으면 경고
        if not citations:
            logger.warning("Citation이 없습니다. 소스 인용이 필요합니다.")
        
        return citations
    
    def handle(self, request: QueryRequest) -> HandlerResponse:
        """
        메인 핸들링 함수 (IndexManager 통합)
        
        Args:
            request: 사용자 요청
            
        Returns:
            HandlerResponse: 처리 결과
        """
        start_time = time.time()
        
        try:
            # 1. 하이브리드 검색 수행
            search_results = self.hybrid_search(request.text, k=5)
            
            if not search_results:
                logger.warning(f"⚠️ {self.domain} 검색 결과 없음")
                return self._create_no_results_response(request, start_time)
            
            # 2. 컨피던스 계산
            confidence = self.calculate_confidence(search_results, request.text)
            
            # 3. 임계값 확인 및 재질문 처리
            reask = None
            if confidence < self.confidence_threshold:
                logger.info(f"📉 {self.domain} 컨피던스 부족: {confidence:.3f} < {self.confidence_threshold}")
                
                # k 확장 재검색 시도
                extended_results = self.hybrid_search(request.text, k=12)
                if extended_results:
                    extended_confidence = self.calculate_confidence(extended_results, request.text)
                    if extended_confidence >= self.confidence_threshold:
                        search_results = extended_results
                        confidence = extended_confidence
                        logger.info(f"📈 {self.domain} 확장 검색으로 컨피던스 회복: {confidence:.3f}")
                    else:
                        # 여전히 낮으면 재질문 생성
                        reask = f"더 구체적인 정보를 제공해주시면 정확한 답변을 드릴 수 있습니다. 예를 들어, 특정 과정명이나 기간을 알려주세요."
                        logger.warning(f"⚠️ 낮은 컨피던스로 재질문 유도 (confidence={confidence:.3f})")
            
            # 4. 컨텍스트 포맷팅
            context = self.format_context(search_results)
            
            # 5. LLM 응답 생성
            system_prompt = self.get_system_prompt()
            user_prompt = f"""컨텍스트:
{context}

질문: {request.text}

위 컨텍스트를 바탕으로 정확하고 도움이 되는 답변을 제공해주세요."""
            
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # 6. Citation 추출
            citations = self.extract_citations(search_results)
            
            # 7. 응답 생성
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            logger.info(f"✅ {self.domain} 핸들러 완료: confidence={confidence:.3f}")
            
            return HandlerResponse(
                answer=answer,
                citations=citations,
                confidence=confidence,
                handler_id=self.domain,
                elapsed_ms=elapsed_ms,
                reask=reask,
                diagnostics={
                    'search_results_count': len(search_results),
                    'threshold_met': confidence >= self.confidence_threshold,
                    'indexmanager_integration': True
                }
            )
            
        except Exception as e:
            logger.error(f"❌ {self.domain} 핸들링 실패: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            return HandlerResponse(
                answer=f"죄송합니다. {self.domain} 정보 처리 중 오류가 발생했습니다.",
                citations=[],
                confidence=0.0,
                handler_id=self.domain,
                elapsed_ms=elapsed_ms,
                diagnostics={'error': str(e)}
            )
    
    def _create_no_results_response(self, request: QueryRequest, start_time: float) -> HandlerResponse:
        """검색 결과 없음 응답 생성"""
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Citation이 없을 때 기본 Citation 생성
        empty_citation = Citation(
            source_id="no_results",
            snippet="검색 결과가 없습니다."
        )
        
        return HandlerResponse(
            answer=f"죄송합니다. '{request.text}' 관련 {self.domain} 정보를 찾을 수 없습니다. 다른 키워드로 검색해 보시거나, 더 구체적인 정보를 제공해주세요.",
            citations=[empty_citation],  # 빈 Citation 대신 기본 Citation 제공
            confidence=0.0,
            handler_id=self.domain,
            elapsed_ms=elapsed_ms,
            diagnostics={'no_search_results': True}
        )
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """도메인별 시스템 프롬프트 반환 (추상 메서드)"""
        pass
    
    @abstractmethod
    def format_context(self, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """도메인별 컨텍스트 포맷팅 (추상 메서드)"""
        pass


# ================================================================
# 스트리밍 응답 지원 (미래 확장용)
# ================================================================

class StreamingHandlerMixin:
    """스트리밍 응답을 위한 믹스인 클래스"""
    
    def stream_response(self, prompt: str, max_tokens: int = 1000) -> Iterator[str]:
        """
        스트리밍 응답 생성
        
        Args:
            prompt: LLM 프롬프트
            max_tokens: 최대 토큰 수
            
        Yields:
            str: 토큰 단위 응답
        """
        try:
            stream = self.llm.stream(prompt, max_tokens=max_tokens)
            for chunk in stream:
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"❌ 스트리밍 응답 실패: {e}")
            yield f"스트리밍 응답 중 오류가 발생했습니다: {str(e)}"


# ================================================================
# 모듈 로드 완료 로그
# ================================================================

logger.info("✅ base_handler.py 모듈 로드 완료 - BM25 오류 수정 버전")