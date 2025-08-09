"""
Router Module: 하이브리드 라우팅 및 병렬 실행 엔진

주요 기능:
1. 규칙 기반 키워드 매칭 + 경량 LLM 의도분류
2. Top-2 핸들러 선정 및 병렬 실행 (타임박스: 1.5s)
3. 컨피던스 기반 응답 선택 및 페일오버 처리
"""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Dict, List, Tuple, Optional, Any
import logging
import openai
from dataclasses import dataclass

from .config import get_config, get_keyword_rules, get_stop_words
from .contracts import QueryRequest, HandlerResponse, ConversationContext
from .logging_utils import get_logger, log_timer

# 핸들러 임포트 (실제 구현 시 주석 해제)
# from handlers.handle_general_query import handle_general_query
# from handlers.handle_publish_query import handle_publish_query
# from handlers.handle_satisfaction_query import handle_satisfaction_query
# from handlers.handle_cyber_query import handle_cyber_query
# from handlers.handle_menu_query import handle_menu_query
# from handlers.handle_notice_query import handle_notice_query
# from handlers.handle_fallback_query import handle_fallback_query

logger = get_logger(__name__)
config = get_config()


@dataclass
class HandlerCandidate:
    """핸들러 후보 정보"""
    handler_id: str
    rule_score: float
    llm_score: float
    final_score: float
    future: Optional[Future] = None


class RouterEngine:
    """하이브리드 라우팅 엔진 - 싱글톤"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="handler")
        
        # 핸들러 매핑 (실제 구현 시 주석 해제)
        self.handlers = {
            # 'general': handle_general_query,
            # 'publish': handle_publish_query,
            # 'satisfaction': handle_satisfaction_query,
            # 'cyber': handle_cyber_query,
            # 'menu': handle_menu_query,
            # 'notice': handle_notice_query,
            # 'fallback': handle_fallback_query,
        }
        
        # config에서 설정값 로드
        self.confidence_thresholds = config.confidence_thresholds
        self.keyword_rules = get_keyword_rules()
        self.stop_words = get_stop_words()
        
        # 타이밍 설정
        self.CANDIDATE_SELECTION_TIMEOUT = config.ROUTER_CANDIDATE_SELECTION_TIMEOUT
        self.HANDLER_EXECUTION_TIMEOUT = config.ROUTER_HANDLER_EXECUTION_TIMEOUT
        self.TOTAL_TIMEOUT = config.ROUTER_TOTAL_TIMEOUT
        
        self._initialized = True
        logger.info("RouterEngine initialized")
    
    def _extract_keywords_score(self, query: str) -> Dict[str, float]:
        """규칙 기반 키워드 매칭으로 핸들러별 점수 계산"""
        query_lower = query.lower().strip()
        scores = {handler_id: 0.0 for handler_id in self.handlers.keys()}
        
        # 불용어 체크 (전체 쿼리가 불용어만 포함된 경우 스킵)
        query_words = set(query_lower.split())
        if query_words.issubset(self.stop_words):
            logger.debug("Query contains only stop words, using uniform low scores")
            return {handler_id: 0.1 for handler_id in self.handlers.keys()}
        
        # 각 핸들러별 키워드 매칭
        for handler_id, keywords in self.keyword_rules.items():
            if handler_id not in self.handlers:
                continue
                
            max_score = 0.0
            matched_keywords = []
            
            for keyword, weight in keywords.items():
                if keyword in query_lower:
                    max_score = max(max_score, weight)
                    matched_keywords.append((keyword, weight))
            
            scores[handler_id] = max_score
            
            if matched_keywords:
                logger.debug(f"{handler_id}: matched keywords {matched_keywords}, final score: {max_score:.3f}")
        
        # fallback은 항상 최소 점수 보장 (다른 핸들러가 모두 0일 때를 위해)
        if 'fallback' in scores:
            scores['fallback'] = max(scores['fallback'], 0.1)
        
        # 디버그 로그
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.debug(f"Keyword matching top scores: {top_scores}")
        
        return scores
    
    def _get_llm_classification(self, query: str, context: ConversationContext) -> Dict[str, float]:
        """경량 LLM을 사용한 의도분류"""
        
        # 토큰 제한: 200-300토큰으로 쿼리 자르기
        if len(query) > 1000:  # 대략 250토큰 추정
            query = query[:1000] + "..."
        
        # 컨텍스트 요약 포함 (간단히)
        context_hint = ""
        if context and context.summary:
            context_hint = f"\n이전 대화 요약: {context.summary[:200]}"
        
        prompt = f"""경상남도인재개발원 RAG 챗봇의 의도분류를 수행합니다.

다음 핸들러 중에서 사용자 질의를 가장 잘 처리할 수 있는 Top-2를 선정하고 점수(0.0~1.0)를 부여하세요:

1. general: 학칙, 규정, 전결규정, 운영원칙, 담당자 연락처 등 일반 정보
2. publish: 2025 교육훈련계획서, 2024 종합평가서 등 공식 발행물
3. satisfaction: 교육과정 만족도, 교과목 만족도 조사 결과
4. cyber: 민간위탁/나라배움터 사이버 교육 일정 및 정보
5. menu: 구내식당 식단표 및 메뉴 정보
6. notice: 최신 공지사항 및 안내사항
7. fallback: 위 카테고리에 맞지 않는 일반적인 질문

사용자 질의: {query}{context_hint}

응답 형식 (JSON only):
{{"top1": {{"handler_id": "핸들러명", "score": 0.85}}, "top2": {{"handler_id": "핸들러명", "score": 0.72}}}}"""

        try:
            with log_timer("llm_classification"):
                response = self.openai_client.chat.completions.create(
                    model=config.OPENAI_MODEL_ROUTER,  # gpt-4o-mini
                    messages=[
                        {"role": "system", "content": "You are an intent classifier. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.1,
                    timeout=0.35  # 0.4초 타임박스 내 완료
                )
                
                result_text = response.choices[0].message.content.strip()
                result = json.loads(result_text)
                
                # 점수 추출 및 검증
                scores = {handler_id: 0.0 for handler_id in self.handlers.keys()}
                
                for key in ['top1', 'top2']:
                    if key in result:
                        handler_id = result[key].get('handler_id', '')
                        score = result[key].get('score', 0.0)
                        
                        if handler_id in scores:
                            scores[handler_id] = max(scores[handler_id], float(score))
                
                return scores
                
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, falling back to rule-based only")
            # LLM 실패 시 균등 분배
            return {handler_id: 0.2 for handler_id in self.handlers.keys()}
    
    def _select_top_candidates(self, query: str, context: ConversationContext) -> List[HandlerCandidate]:
        """Top-2 핸들러 후보 선정 (0.4초 타임박스)"""
        start_time = time.time()
        
        try:
            # 1. 규칙 기반 점수 계산
            rule_scores = self._extract_keywords_score(query)
            
            # 2. LLM 기반 점수 계산 (타임아웃 내)
            remaining_time = self.CANDIDATE_SELECTION_TIMEOUT - (time.time() - start_time)
            if remaining_time > 0.1:
                llm_scores = self._get_llm_classification(query, context)
            else:
                logger.warning("Insufficient time for LLM classification, using rule-based only")
                llm_scores = {handler_id: 0.0 for handler_id in self.handlers.keys()}
            
            # 3. 최종 점수 계산 (가중평균: 0.5/0.5)
            candidates = []
            for handler_id in self.handlers.keys():
                rule_score = rule_scores.get(handler_id, 0.0)
                llm_score = llm_scores.get(handler_id, 0.0)
                final_score = (rule_score * 0.5) + (llm_score * 0.5)
                
                candidates.append(HandlerCandidate(
                    handler_id=handler_id,
                    rule_score=rule_score,
                    llm_score=llm_score,
                    final_score=final_score
                ))
            
            # 4. Top-2 선정 (동점 시 규칙 기반 우선)
            candidates.sort(key=lambda x: (x.final_score, x.rule_score), reverse=True)
            top_candidates = candidates[:2]
            
            # 5. fallback 보장 (Top-2에 포함되지 않았고, 다른 후보들의 점수가 낮다면 강제 추가)
            if not any(c.handler_id == 'fallback' for c in top_candidates):
                fallback_candidate = next((c for c in candidates if c.handler_id == 'fallback'), None)
                if fallback_candidate and len(top_candidates) == 2 and top_candidates[1].final_score < 0.3:
                    top_candidates[1] = fallback_candidate
                    logger.debug("Replaced low-score candidate with fallback")
            
            elapsed = time.time() - start_time
            logger.info(f"Selected candidates in {elapsed:.3f}s: {[(c.handler_id, c.final_score) for c in top_candidates]}")
            
            return top_candidates
            
        except Exception as e:
            logger.error(f"Candidate selection failed: {e}")
            # 긴급 fallback: general + fallback (하드코딩 점수)
            return [
                HandlerCandidate('general', 0.5, 0.5, 0.5),
                HandlerCandidate('fallback', 0.1, 0.1, 0.1)
            ]
    
    def _execute_handler_with_timeout(self, candidate: HandlerCandidate, request: QueryRequest) -> Optional[HandlerResponse]:
        """개별 핸들러 실행 (타임아웃 포함)"""
        try:
            # TODO: 실제 핸들러 함수 호출
            # handler_func = self.handlers[candidate.handler_id]
            # return handler_func(request, context_manager_instance)
            
            # 임시 모킹 응답
            import time
            time.sleep(0.3)  # 시뮬레이션
            
            return HandlerResponse(
                answer=f"[{candidate.handler_id}] 모킹된 응답입니다: {request.text}",
                citations=[],
                confidence=candidate.final_score * 0.9,  # 실제 실행 시 약간 감소
                handler_id=candidate.handler_id,
                elapsed_ms=300
            )
            
        except Exception as e:
            logger.error(f"Handler {candidate.handler_id} execution failed: {e}")
            return None
    
    def _execute_parallel_handlers(self, candidates: List[HandlerCandidate], request: QueryRequest) -> HandlerResponse:
        """병렬 핸들러 실행 및 응답 선택 (1.1초 타임박스)"""
        start_time = time.time()
        
        # Future 제출
        for candidate in candidates:
            candidate.future = self.executor.submit(
                self._execute_handler_with_timeout, candidate, request
            )
        
        completed_responses = []
        partial_responses = []
        
        try:
            # 완료된 작업 대기 (타임아웃 포함)
            for future in as_completed([c.future for c in candidates], timeout=self.HANDLER_EXECUTION_TIMEOUT):
                response = future.result()
                if response:
                    completed_responses.append(response)
                    
                    # 컨피던스 임계값 확인
                    threshold = self.confidence_thresholds.get(response.handler_id, 0.7)
                    
                    # follow_up인 경우 임계값 완화
                    if request.follow_up:
                        threshold -= 0.02
                    
                    if response.confidence >= threshold:
                        logger.info(f"High confidence response from {response.handler_id}: {response.confidence:.3f} >= {threshold:.3f}")
                        # 다른 Future들 취소
                        for c in candidates:
                            if c.future and not c.future.done():
                                c.future.cancel()
                        return response
        
        except Exception as e:
            logger.warning(f"Parallel execution timeout or error: {e}")
            
            # 타임아웃 시 페일오버: 응답 길이가 가장 긴 것 선택
            for candidate in candidates:
                if candidate.future and not candidate.future.done():
                    try:
                        # 강제로 결과 가져오기 (논블로킹)
                        response = candidate.future.result(timeout=0.1)
                        if response:
                            partial_responses.append(response)
                    except:
                        pass
                    finally:
                        candidate.future.cancel()
        
        # 최종 응답 선택
        all_responses = completed_responses + partial_responses
        
        if all_responses:
            # 1. 컨피던스 우선, 2. 응답 길이 우선
            best_response = max(all_responses, key=lambda r: (r.confidence, len(r.answer)))
            
            elapsed = time.time() - start_time
            logger.info(f"Selected response from {best_response.handler_id} after {elapsed:.3f}s")
            return best_response
        else:
            # 모든 핸들러 실패 시 응급 응답
            logger.error("All handlers failed, returning emergency fallback")
            return HandlerResponse(
                answer="죄송합니다. 일시적인 문제로 답변을 생성할 수 없습니다. 잠시 후 다시 시도해 주세요.",
                citations=[],
                confidence=0.0,
                handler_id="emergency_fallback",
                elapsed_ms=int((time.time() - start_time) * 1000)
            )
    
    def route(self, request: QueryRequest) -> HandlerResponse:
        """메인 라우팅 인터페이스"""
        total_start = time.time()
        
        try:
            logger.info(f"Routing query: {request.text[:100]}...")
            
            # 1. 후보 선정 (0.4초)
            candidates = self._select_top_candidates(request.text, request.context)
            
            # 2. 병렬 실행 (1.1초)
            response = self._execute_parallel_handlers(candidates, request)
            
            # 3. 응답 후처리
            total_elapsed = time.time() - total_start
            response.elapsed_ms = int(total_elapsed * 1000)
            
            logger.info(f"Routing completed in {total_elapsed:.3f}s, selected: {response.handler_id}")
            return response
            
        except Exception as e:
            logger.error(f"Router critical error: {e}")
            
            # 크리티컬 에러 시 응급 응답
            return HandlerResponse(
                answer="시스템 오류가 발생했습니다. 관리자에게 문의해 주세요.",
                citations=[],
                confidence=0.0,
                handler_id="critical_error",
                elapsed_ms=int((time.time() - total_start) * 1000)
            )


# 싱글톤 인스턴스
_router_engine = None


def get_router() -> RouterEngine:
    """Router 싱글톤 인스턴스 반환"""
    global _router_engine
    if _router_engine is None:
        _router_engine = RouterEngine()
    return _router_engine


def route(request: QueryRequest) -> HandlerResponse:
    """라우팅 메인 엔트리포인트"""
    router = get_router()
    return router.route(request)


# 테스트용 함수
def test_router():
    """Router 테스트 함수"""
    from .contracts import ConversationContext
    
    test_context = ConversationContext(
        conversation_id="test-123",
        summary="사용자가 교육 관련 질문을 하고 있음",
        recent_messages=[],
        entities=["교육", "만족도"],
        updated_at=None
    )
    
    test_request = QueryRequest(
        text="2024년 교육과정 만족도 결과를 보여주세요",
        context=test_context,
        follow_up=False,
        trace_id="test-trace-123"
    )
    
    response = route(test_request)
    print(f"Test result: {response.handler_id}, confidence: {response.confidence}")
    return response


if __name__ == "__main__":
    test_router()
