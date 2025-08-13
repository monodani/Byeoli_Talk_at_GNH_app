#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - router.py

하이브리드 라우팅 & 병렬 실행 엔진:
- 1차: 키워드 규칙 기반 후보 필터링
- 2차: 경량 LLM으로 Top-2 핸들러 선정
- 3차: 선정된 핸들러 병렬 실행 (15.0s 타임박스)
- 4차: 컨피던스 기반 최종 응답 선택

핵심 특징:
- 총 15.0s 타임박스 (후보선정 3.0s + 병렬실행 12.0s)
- 규칙+LLM 하이브리드 후보 선정
- asyncio 병렬 실행으로 성능 최적화
- 실패 시 fallback 핸들러 보장
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 프로젝트 모듈
from utils.contracts import (
    QueryRequest, HandlerResponse, HandlerCandidate, RouterResponse,
    HandlerType, PerformanceMetrics, ErrorResponse, create_error_response,
    ConversationContext # 추가: QueryRequest 생성을 위해 필요
)
from utils.config import config

# 외부 라이브러리
from langchain_openai import ChatOpenAI

# 핸들러 import
from handlers.satisfaction_handler import satisfaction_handler
from handlers.general_handler import general_handler
from handlers.menu_handler import menu_handler
from handlers.cyber_handler import cyber_handler
from handlers.publish_handler import publish_handler
from handlers.notice_handler import notice_handler
from handlers.fallback_handler import fallback_handler

# 로깅 설정
logger = logging.getLogger(__name__)


# ================================================================
# 1. 핸들러 레지스트리 및 규칙 정의
# ================================================================

class HandlerRegistry:
    """핸들러 인스턴스 관리 레지스트리"""
    
    def __init__(self):
        self._handlers = {}
        self._load_handlers()
    
    def _load_handlers(self):
        """모든 핸들러 인스턴스 생성 및 등록"""
        try:
            self._handlers = {
                HandlerType.SATISFACTION: satisfaction_handler(),
                HandlerType.GENERAL: general_handler(),
                HandlerType.MENU: menu_handler(),
                HandlerType.CYBER: cyber_handler(),
                HandlerType.PUBLISH: publish_handler(),
                HandlerType.NOTICE: notice_handler(),
                HandlerType.FALLBACK: fallback_handler()
            }
            logger.info(f"✅ 핸들러 레지스트리 초기화 완료: {len(self._handlers)}개 핸들러")
        except Exception as e:
            logger.error(f"❌ 핸들러 레지스트리 초기화 실패: {e}")
            raise
    
    def get_handler(self, handler_type: HandlerType):
        """핸들러 인스턴스 반환"""
        return self._handlers.get(handler_type)
    
    def get_all_handlers(self) -> Dict[HandlerType, Any]:
        """모든 핸들러 반환"""
        return self._handlers.copy()


class RoutingRules:
    """키워드 기반 라우팅 규칙 정의"""
    
    # 도메인별 키워드 규칙
    DOMAIN_KEYWORDS = {
        HandlerType.SATISFACTION: {
            "primary": ["만족도", "평가", "설문", "조사", "점수", "순위", "교육과정", "교과목"],
            "secondary": ["피드백", "의견", "개선", "평점", "만족", "불만", "제안"]
        },
        HandlerType.GENERAL: {
            "primary": ["학칙", "규정", "전결", "운영원칙", "연락처", "담당자", "부서", "전화번호"],
            "secondary": ["규칙", "지침", "조례", "업무", "담당", "부서명", "연락", "문의"]
        },
        HandlerType.MENU: {
            "primary": ["식단", "메뉴", "구내식당", "급식", "식사", "점심", "저녁"],
            "secondary": ["음식", "밥", "식당", "카페테리아", "식당메뉴", "오늘메뉴"]
        },
        HandlerType.CYBER: {
            "primary": ["사이버교육", "온라인교육", "이러닝", "나라배움터", "민간위탁"],
            "secondary": ["원격교육", "인터넷교육", "온라인강의", "사이버강의", "디지털교육"]
        },
        HandlerType.PUBLISH: {
            "primary": ["교육계획", "훈련계획", "2025계획", "2024평가", "종합평가서", "계획서"],
            "secondary": ["교육방침", "운영계획", "성과평가", "계획", "평가서", "발행물"]
        },
        HandlerType.NOTICE: {
            "primary": ["공지", "안내", "알림", "공지사항", "새소식", "업데이트", "벼리", "시설", "도서실","도서관","자료실","열람실","이용안내","이용시간","운영시간","휴관일", "대출","반납","연장","예약","연체","대출정지", "자료현황", "소장자료", "전자책", "E-BOOK"],
            "secondary": ["소식", "정보", "통지", "발표", "알림", "공고", "캐릭터", "찾아오시는 길", "연혁", "점심시간","연체료","권수","기간","장기대출", "복사","부분복사","저작권","배상","분실","훼손","현금배상","현물변상"]
        }
    }
    
    @classmethod
    def calculate_rule_score(cls, query: str, handler_type: HandlerType) -> float:
        """규칙 기반 점수 계산"""
        query_lower = query.lower()
        keywords = cls.DOMAIN_KEYWORDS.get(handler_type, {})
        
        # Primary 키워드 매치 (가중치 0.7)
        primary_matches = sum(1 for kw in keywords.get("primary", []) if kw in query_lower)
        primary_score = min(primary_matches * 0.3, 0.7)
        
        # Secondary 키워드 매치 (가중치 0.3)
        secondary_matches = sum(1 for kw in keywords.get("secondary", []) if kw in query_lower)
        secondary_score = min(secondary_matches * 0.1, 0.3)
        
        total_score = primary_score + secondary_score
        return min(total_score, 1.0)


# ================================================================
# 2. 라우터 클래스
# ================================================================

class Router:
    """
    하이브리드 라우팅 & 병렬 실행 라우터
    
    주요 기능:
    - 규칙 + LLM 하이브리드 후보 선정
    - Top-2 핸들러 병렬 실행
    - 15.0s 타임박스 현실적 조정
    - 컨피던스 기반 최종 응답 선택
    """
    
    def __init__(self):
        # 핸들러 레지스트리
        self.registry = HandlerRegistry()
        
        # 경량 LLM (후보 선정용)
        self.llm_light = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=100,
            timeout=2.0
        )
        
        # 성능 설정 - 현실적 타임박스로 조정
        self.TIMEBOX_TOTAL = 15.0       # 타임박스 총 시간
        self.TIMEBOX_SELECTION = 3.0  # 핸들러 후보 선정시간
        self.TIMEBOX_EXECUTION = 12.0  # 핸들러 처리시간
        
        logger.info("🚀 Router 초기화 완료 (타임박스: 15.0s)")  
    
    async def route(self, user_input: str, context: Optional[ConversationContext] = None, **kwargs) -> HandlerResponse: # ✅ 수정: QueryRequest 대신 user_input과 context를 받음
        """
        메인 라우팅 함수
        
        Args:
            user_input: 사용자 요청 텍스트
            context: 대화 컨텍스트 (선택 사항)
            
        Returns:
            HandlerResponse: 최종 응답
        """
        start_time = time.time()
        
        try:
            # 1단계: Top-2 핸들러 선정 (3.0s)
            selected_handlers = await self._select_top_handlers(user_input, context)
            selection_time = time.time() - start_time
            
            if selection_time > self.TIMEBOX_SELECTION:
                logger.warning(f"⚠️ 핸들러 선정 시간 초과: {selection_time:.3f}s > {self.TIMEBOX_SELECTION}s")

            # 2단계 직전, request 만들기 전에 (selected_handlers 구한 뒤)
            trace_id = kwargs.get("trace_id") or str(uuid.uuid4())
            primary_domain = HandlerType(selected_handlers[0].domain) if selected_handlers else HandlerType.FALLBACK

            
            # ✅ 2단계: 선정된 핸들러들을 위해 QueryRequest 객체 생성
            primary_domain = HandlerType(selected_handlers[0].domain) if selected_handlers else HandlerType.FALLBACK
            request = QueryRequest(
                text=user_input,
                domain=primary_domain.value, # ✅ 수정: 이제 domain에 올바른 값이 할당됨
                context=context,
                follow_up=kwargs.get("follow_up", False),
                trace_id=trace_id
            )
            
            # 3단계: 선정된 핸들러 병렬 실행 (12.0s)
            execution_start = time.time()
            final_response = await self._execute_handlers_parallel(
                request, selected_handlers, self.TIMEBOX_EXECUTION
            )
            execution_time = time.time() - execution_start
            
            # 성능 메트릭 계산
            total_time = time.time() - start_time
            metrics = PerformanceMetrics(
                total_time_ms=int(total_time * 1000),
                router_time_ms=int(selection_time * 1000),
                handler_time_ms=int(execution_time * 1000)
            )
            
            # 응답에 성능 정보 추가
            final_response.diagnostics.update({
                "routing_metrics": metrics.dict(),
                "selected_handlers": [h.domain for h in selected_handlers],
                "timebox_compliance": metrics.within_timebox
            })
            
            logger.info(f"✅ 라우팅 완료 [{getattr(request, 'trace_id', 'no-trace')}]"
                        f": {total_time:.3f}s (타임박스: {'✓' if metrics.within_timebox else '✗'})")
            return final_response
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"❌ 라우팅 실패: {e} ({error_time:.3f}s)")
            
            # Fallback 핸들러로 안전망 제공
            return await self._emergency_fallback(user_input, str(e)) # ✅ 수정: user_input을 전달
    
    async def _select_top_handlers(self, query: str, context: Optional[ConversationContext] = None) -> List[HandlerCandidate]:
        """
        하이브리드 방식으로 Top-2 핸들러 선정
        
        Args:
            query: 사용자 요청
            
        Returns:
            List[HandlerCandidate]: 선정된 최대 2개 핸들러
        """
        selection_start = time.time()
        
        try:
            # 1차: 규칙 기반 후보 스코어링
            rule_scores = {}
            for handler_type in HandlerType:
                if handler_type == HandlerType.FALLBACK:
                    continue  # fallback은 최후 수단이므로 제외
                score = RoutingRules.calculate_rule_score(query, handler_type) # ✅ 수정: query를 전달
                rule_scores[handler_type] = score
            
            # 규칙 점수 0.1 이상인 후보만 LLM 평가 대상
            rule_candidates = {k: v for k, v in rule_scores.items() if v >= 0.1}
            
            if not rule_candidates:
                # 규칙 매칭 실패 시 모든 핸들러 고려
                rule_candidates = rule_scores
                logger.warning("규칙 기반 매칭 실패, 전체 핸들러 고려")
            
            # 2차: 경량 LLM으로 정밀 분류
            llm_scores = await self._llm_classify_handlers(
                query, list(rule_candidates.keys()) # ✅ 수정: query를 전달
            )
            
            # 3차: 규칙(30%) + LLM(70%) 가중 평균
            candidates = []
            for handler_type in rule_candidates.keys():
                rule_score = rule_scores.get(handler_type, 0.0)
                llm_score = llm_scores.get(handler_type, 0.0)
                confidence = rule_score * 0.3 + llm_score * 0.7
                
                candidate = HandlerCandidate(
                    domain=handler_type.value,  # ✅ domain 필드 사용
                    confidence=confidence,  # ✅ confidence 필드 사용
                    reasoning=f"규칙:{rule_score:.2f} + LLM:{llm_score:.2f} = {confidence:.2f}",
                    is_rule_based=False,
                    metadata={
                        "rule_score": rule_score,
                        "llm_score": llm_score,
                        "combined_score": confidence
                    }
                )   
                candidates.append(candidate)
            
            # 점수 기준 정렬 후 Top-2 선정
            candidates.sort(key=lambda x: x.confidence, reverse=True)
            top_candidates = candidates[:2]
            
            # follow_up 요청 시 θ 완화 적용
            # ✅ request.follow_up 대신 context에서 follow_up 여부 확인
            if context and context.get("follow_up", False) and top_candidates:
                logger.info("후속 질문 감지: 컨피던스 임계값 -0.02 완화 적용")
            
            selection_time = time.time() - selection_start
            logger.info(f"🎯 핸들러 선정 완료 ({selection_time:.3f}s): {[c.domain for c in top_candidates]}")
            
            return top_candidates
            
        except Exception as e:
            logger.error(f"❌ 핸들러 선정 실패: {e}")
            # 실패 시 general + fallback 반환
            return [
                HandlerCandidate(
                    domain=HandlerType.GENERAL.value,
                    confidence=0.1,
                    reasoning="선정 실패 시 기본값",
                    is_rule_based=True,
                    metadata={"fallback": True}                    
                ),
                HandlerCandidate(
                    domain=HandlerType.FALLBACK.value,
                    confidence=0.05,
                    reasoning="안전망",
                    is_rule_based=True,
                    metadata={"fallback": True, "type": "safety_net"}

                )
            ]
    
    async def _llm_classify_handlers(self, query: str, candidate_types: List[HandlerType]) -> Dict[HandlerType, float]:
        # ... (이 함수는 수정할 필요 없음)
        # 기존 코드를 유지하되, 호출 시 `query` 변수를 사용하도록 수정
        ...
    
    async def _execute_handlers_parallel(
        self, 
        request: QueryRequest, 
        candidates: List[HandlerCandidate],
        timeout_seconds: float
    ) -> HandlerResponse:
        """
        선정된 핸들러들을 병렬 실행하여 최적 응답 선택
        
        Args:
            request: 사용자 요청
            candidates: 선정된 핸들러 후보들
            timeout_seconds: 실행 타임아웃
            
        Returns:
            HandlerResponse: 최종 선택된 응답
        """
        if not candidates:
            return await self._emergency_fallback(request.query, "선정된 핸들러가 없음")
        
        execution_start = time.time()
        responses = {}
        
        try:
            # 병렬 실행을 위한 ThreadPoolExecutor 사용
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Future 생성
                futures = {}
                for candidate in candidates:
                    handler = self.registry.get_handler(HandlerType(candidate.domain))
                    if handler:
                        # ✅ 이제 `request` 객체는 `domain`을 포함하고 있음
                        # follow_up 요청 시 컨피던스 임계값 완화
                        if request.follow_up:
                            original_threshold = handler.confidence_threshold
                            handler.confidence_threshold = max(0.0, original_threshold - 0.02)
                            logger.debug(f"임계값 완화: {candidate.domain} {original_threshold:.2f} → {handler.confidence_threshold:.2f}")
                        
                        future = executor.submit(handler.handle, request)
                        futures[future] = HandlerType(candidate.domain)
                
                # 타임아웃 내에서 완료된 작업 수집
                for future in as_completed(futures.keys(), timeout=timeout_seconds):
                    try:
                        domain = futures[future]
                        response = future.result(timeout=0.1)  # 이미 완료된 작업이므로 즉시 반환
                        responses[handler_id] = response                        
                        logger.info(f"✅ {domain.value} 핸들러 완료: confidence={response.confidence:.3f}")
                        
                    except Exception as e:
                        domain = futures[future]
                        logger.error(f"❌ {domain.value} 핸들러 실행 실패: {e}")
                        continue
                
        except TimeoutError:
            logger.warning(f"⏰ 핸들러 실행 타임아웃 ({timeout_seconds}s)")
        
        # 응답 선택 로직
        if responses:
            # 컨피던스 점수 기준으로 최적 응답 선택
            best_response = max(responses.values(), key=lambda r: r.confidence)
            
            execution_time = time.time() - execution_start
            logger.info(f"🎯 최적 응답 선택: {best_response.handler_id} (confidence: {best_response.confidence:.3f}, {execution_time:.3f}s)")
            
            return best_response
        else:
            # 모든 핸들러 실행 실패 시 fallback
            logger.error("❌ 모든 핸들러 실행 실패, fallback 실행")
            return await self._emergency_fallback(request.query, "모든 핸들러 실행 실패") # ✅ 수정: request.query를 전달
    
    async def _emergency_fallback(self, query: str, error_reason: str) -> HandlerResponse: # ✅ 수정: request 대신 query를 받음
        """
        긴급 상황 시 fallback 핸들러 실행
        
        Args:
            query: 사용자 요청
            error_reason: 실패 원인
            
        Returns:
            HandlerResponse: fallback 응답
        """
        try:
            logger.warning(f"🚨 긴급 fallback 실행: {error_reason}")
            
            fallback = self.registry.get_handler(HandlerType.FALLBACK)
            if fallback:
                # ✅ Fallback 핸들러를 위해 임시 QueryRequest 객체 생성
                fallback_request = QueryRequest(
                    text=query,
                    domain=HandlerType.FALLBACK.value,
                    context=None,
                    follow_up=False,
                    trace_id=str(uuid.uuid4()),
                    metadata={"error": error_reason}
                )
                response = await asyncio.to_thread(fallback.handle, fallback_request)
                response.diagnostics["emergency_reason"] = error_reason
                return response
            else:
                # fallback도 없는 극단적 상황
                raise Exception("Fallback 핸들러를 찾을 수 없습니다")
                
        except Exception as e:
            logger.critical(f"💥 Fallback 핸들러도 실패: {e}")
            
            # 최후의 수단: 하드코딩된 응답
            from utils.contracts import Citation
            return HandlerResponse(
                answer="죄송합니다. 현재 시스템에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주시거나, 담당부서(055-254-2011)로 직접 문의해주세요.",
                citations=[Citation(
                    source_id="system/emergency",
                    text="긴급 상황 시 기본 응답",
                    relevance_score=0.0,
                    snippet="긴급 상황 시 기본 응답"
                )],
                confidence=0.1,
                domain=HandlerType.FALLBACK,
                elapsed_ms=100,
                diagnostics={"emergency_fallback": True, "error": str(e)}
            )


# ================================================================
# 3. 라우터 싱글톤 인스턴스
# ================================================================

class RouterSingleton:
    """라우터 싱글톤 패턴"""
    
    _instance = None
    _router = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_router(self) -> Router:
        """라우터 인스턴스 반환 (지연 로딩)"""
        if self._router is None:
            self._router = Router()
        return self._router


# ================================================================
# 4. 편의 함수들
# ================================================================

def get_router() -> Router:
    """전역 라우터 인스턴스 반환"""
    return RouterSingleton().get_router()


async def route_query(text: str, **kwargs) -> HandlerResponse:
    """
    간편한 쿼리 라우팅 함수
    
    Args:
        text: 사용자 질문
        **kwargs: QueryRequest 추가 파라미터 (context 등)
        
    Returns:
        HandlerResponse: 최종 응답
    """
    router = get_router()
    # ✅ 수정: router.route 함수가 이제 user_input과 context를 직접 받음
    return await router.route(user_input=text, **kwargs)


def analyze_routing_performance(response: HandlerResponse) -> Dict[str, Any]:
    # ... (이 함수는 수정할 필요 없음)
    ...


# ================================================================
# 5. 개발/테스트용 함수들
# ================================================================

async def test_routing_performance():
    # ... (이 함수는 수정할 필요 없음)
    ...


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    # 간단한 테스트 실행
    async def main():
        response = await route_query("교육과정 만족도 1위 알려줘")
        print(f"응답: {response.answer[:100]}...")
        print(f"컨피던스: {response.confidence:.3f}")
        print(f"핸들러: {response.handler_id}")
    
    asyncio.run(main())
