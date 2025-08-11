#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - fallback_handler (도메인 매핑 수정 및 추상 메서드 구현)

미매칭/저신뢰 답변 처리 전용 핸들러
모든 전문 핸들러가 실패했을 때 최후의 보루 역할

주요 수정사항:
✅ domain="general" 사용 (IndexManager 호환)
✅ 추상 메서드 'format_context'와 'get_system_prompt' 구현
✅ 벡터스토어 없이도 작동하는 안전 로직
✅ Citation 생성 보장
✅ 에러 처리 강화
"""

import time
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# 프로젝트 모듈
from handlers.base_handler import base_handler
from utils.contracts import QueryRequest, HandlerResponse, Citation, ConversationContext

# 로깅 설정
logger = logging.getLogger(__name__)


class fallback_handler(base_handler):
    """
    최종 백업 핸들러 (추상 메서드 구현 완료)
    
    처리 범위:
    - 모든 전문 핸들러 실패 시 작동
    - 일반적인 기관 정보 제공
    - 적절한 담당부서 연결
    - 재질문 유도 및 도움말 제공
    
    특징:
    - ✅ general 도메인 사용 (IndexManager 호환)
    - 최저 컨피던스 임계값 (θ=0.00)
    - 항상 성공 보장 (절대 실패하지 않음)
    - EWMA 자동 튜닝 지원
    - 대화 맥락 고려한 맞춤형 안내
    """
    
    def __init__(self):
        super().__init__(
            domain="general",  # fallback → general로 변경
            index_name="general_index",
            confidence_threshold=0.00
        )
        
        self.handler_type = "fallback"
        
        self.question_categories = {
            'satisfaction': {
                'keywords': ['만족도', '평가', '점수', '성과', '결과', '설문', '조사'],
                'department': '평가분석담당',
                'contact': '055-254-2021',
                'description': '교육평가 및 만족도 조사'
            },
            'planning': {
                'keywords': ['계획', '일정', '교육과정', '신청', '모집', '등록'],
                'department': '교육기획담당',
                'contact': '055-254-2051',
                'description': '교육훈련계획 및 운영'
            },
            'cyber': {
                'keywords': ['사이버', '온라인', '이러닝', '민간위탁', '나라배움터', '인터넷'],
                'department': '사이버담당',
                'contact': '055-254-2081',
                'description': '사이버교육 운영'
            },
            'facility': {
                'keywords': ['전결', '규정', '식당', '기숙사', '주차', '시설', '연락처'],
                'department': '총무담당',
                'contact': '055-254-2011',
                'description': '시설 및 총무업무'
            },
            'general': {
                'keywords': ['학칙', '운영', '원칙', '안내', '정보', '위치', '주소'],
                'department': '교육기획담당',
                'contact': '055-254-2051',
                'description': '일반 운영 및 안내'
            }
        }
        
        self.institution_info = {
            'name': '경상남도인재개발원',
            'address': '경상남도 진주시 월아산로 2026 경상남도 서부청사 4~6층',
            'main_phone': '055-254-2011',
            'website': 'https://gyeongnam.go.kr/hrd',
            'business_hours': '평일 09:00-18:00'
        }
        
        self.ewma_success_rate = 0.8
        self.ewma_alpha = 0.1
        
        logger.info(f"🛡️ fallback_handler 초기화 완료 (θ={self.confidence_threshold:.2f})")

    def format_context(self, context: ConversationContext, query: str) -> str:
        """
        [필수 구현]
        Fallback 핸들러는 검색 실패를 가정하므로, 질문과 대화 요약만 활용합니다.
        """
        formatted_context = f"사용자 질문: {query}"
        if context.summary:
            formatted_context += f"\n\n이전 대화 요약: {context.summary}"
        return formatted_context

    def get_system_prompt(self, context: ConversationContext) -> str:
        """
        [필수 구현]
        Fallback 핸들러는 모든 전문 핸들러가 실패했을 때 호출되므로,
        어떤 정보를 찾지 못했음을 알리는 프롬프트를 반환합니다.
        """
        return """
당신은 경상남도인재개발원의 RAG 챗봇 벼리톡(BYEOLI TALK)입니다.
사용자의 질문에 대해 적절한 답변을 찾지 못했거나 관련 데이터가 없는 경우, 아래의 지침을 따라 응답해 주세요.

- 사용자에게 현재는 해당 정보를 찾을 수 없음을 정중하게 알립니다.
- "죄송합니다. 현재 데이터베이스에서는 해당 정보를 찾을 수 없습니다." 와 같은 표현을 사용합니다.
- 대신 다른 질문을 하거나, 문의사항이 있다면 직접 전화 문의(055-254-2011)를 안내합니다.
- 절대 없는 내용을 지어내거나 추측하지 마세요.
"""

    def handle(self, request: QueryRequest) -> HandlerResponse:
        start_time = time.time()
        logger.info(f"🛡️ fallback 핸들러 처리 시작: {request.text[:50]}...")
        
        try:
            category_info = self._classify_question_category(request.text)
            
            search_results = []
            try:
                # 'general' 도메인에서 검색 시도
                search_results = self._search_documents(request.text, k=3)
                logger.info(f"✅ general 도메인에서 {len(search_results)}개 문서 검색됨")
            except Exception as e:
                logger.warning(f"⚠️ general 도메인 검색 실패: {e}, 기본 응답 생성")
            
            answer = self._generate_fallback_answer(
                request.text, 
                category_info, 
                search_results
            )
            
            citations = self._generate_citations(search_results, category_info)
            reask_suggestions = self._generate_reask_suggestions(category_info)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            logger.info(f"✅ fallback 핸들러 처리 완료 ({elapsed_ms}ms, 카테고리={category_info.get('department', 'general')})")
            
            self._update_success_rate(True)
            
            return HandlerResponse(
                answer=answer,
                citations=citations,
                confidence=1.0,
                handler_id=self.handler_type,
                elapsed_ms=elapsed_ms,
                reask=reask_suggestions if reask_suggestions else None,
                diagnostics={
                    'category_detected': category_info.get('department', 'general'),
                    'search_results_count': len(search_results),
                    'fallback_reason': 'all_handlers_failed',
                    'always_success': True,
                    'domain_used': self.domain
                }
            )
            
        except Exception as e:
            logger.error(f"❌ fallback 핸들러에서도 예외 발생: {e}")
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            try:
                emergency_citation = Citation(
                    source_id="emergency/system",
                    snippet="시스템 오류 시 긴급 응답"
                )
                emergency_citations = [emergency_citation]
            except:
                emergency_citations = []
            
            emergency_response = HandlerResponse(
                answer=f"""죄송합니다. 일시적인 시스템 문제가 발생했습니다.

**긴급 연락처:**
📞 대표전화: {self.institution_info['main_phone']}
🌐 홈페이지: {self.institution_info['website']}

**담당부서별 연락처:**
• 총무담당: 055-254-2011
• 평가분석담당: 055-254-2021  
• 교육기획담당: 055-254-2051
• 교육운영1담당: 055-254-2061
• 교육운영2담당: 055-254-2071
• 교육운영1담당: 055-254-2081

잠시 후 다시 시도해주시거나, 위 연락처로 직접 문의해 주세요.""",
                citations=emergency_citations,
                confidence=1.0,
                handler_id=self.handler_type,
                elapsed_ms=elapsed_ms,
                diagnostics={
                    "emergency_fallback": True,
                    "error": str(e),
                    "always_success_guarantee": True
                }
            )
            
            self._update_success_rate(False)
            return emergency_response
    
    def _classify_question_category(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        
        category_scores = {}
        for category, info in self.question_categories.items():
            score = sum(1 for keyword in info['keywords'] if keyword in query_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            return self.question_categories[best_category]
        else:
            return self.question_categories['general']
    
    def _generate_fallback_answer(
        self, 
        query: str, 
        category_info: Dict[str, Any], 
        search_results: List[Any]
    ) -> str:
        
        base_answer = f"""안녕하세요! 경상남도인재개발원 벼리톡(BYEOLI TALK)입니다.

요청하신 내용에 대해 정확한 답변을 드리기 어려운 상황입니다."""
        
        if search_results:
            base_answer += f"\n\n**관련 정보:**\n"
            for i, result in enumerate(search_results[:2]):
                try:
                    snippet = getattr(result, 'page_content', str(result))[:100]
                    base_answer += f"• {snippet}...\n"
                except:
                    continue
        
        base_answer += f"""

**담당부서 안내:**
📞 {category_info['department']}: {category_info['contact']}
📝 담당업무: {category_info['description']}

**기관 정보:**
🏢 {self.institution_info['name']}
📍 {self.institution_info['address']}
📞 대표전화: {self.institution_info['main_phone']}
🌐 홈페이지: {self.institution_info['website']}
🕒 운영시간: {self.institution_info['business_hours']}"""

        return base_answer
    
    def _generate_citations(self, search_results: List[Any], category_info: Dict[str, Any]) -> List[Citation]:
        citations = []
        
        for i, result in enumerate(search_results[:2]):
            try:
                source_id = getattr(result, 'metadata', {}).get('source', f'general/doc_{i+1}')
                snippet = getattr(result, 'page_content', str(result))[:150]
                
                if len(snippet) == 150:
                    last_space = snippet.rfind(' ', 0, 150)
                    if last_space > 100:
                        snippet = snippet[:last_space] + "..."
                
                citation = Citation(
                    source_id=source_id,
                    snippet=snippet
                )
                citations.append(citation)
            except Exception as e:
                logger.warning(f"Citation 생성 실패: {e}")
                continue
        
        if not citations:
            try:
                default_citation = Citation(
                    source_id="institution/contact_info",
                    snippet=f"{category_info['department']} {category_info['contact']} - {category_info['description']}"
                )
                citations.append(default_citation)
            except Exception as e:
                logger.error(f"기본 Citation 생성 실패: {e}")
                try:
                    emergency_citation = Citation(
                        source_id="emergency/fallback",
                        snippet="시스템 응답 - 담당부서 연결 안내"
                    )
                    citations.append(emergency_citation)
                except:
                    pass
        
        return citations
    
    def _generate_reask_suggestions(self, category_info: Dict[str, Any]) -> str:
        department = category_info.get('department', '교육기획담당')
        
        suggestions = {
            '평가분석담당': "교육과정 만족도나 평가 결과에 대해 구체적으로 문의해주세요.",
            '교육기획담당': "교육계획, 과정 신청, 일정에 대해 구체적으로 문의해주세요.",
            '교육운영1담당': "신규 임용(후보)자, 리더십 및 역량 교육과정 운영 및 강사 관리 등에 대해 구체적으로 문의해주세요.",
            '교육운영2담당': "중견리더 과정, 직무역량 및 핵심과제 교육과정 운영 및 강사 관리 등에 대해 구체적으로 문의해주세요.",
            '사이버담당': "사이버교육 과정명이나 수강 방법을 구체적으로 문의해주세요.",
            '총무담당': "시설 이용이나 구내식당 등과 관련해서 구체적으로 문의해주세요."
        }
        
        return suggestions.get(department, "더 구체적인 질문으로 다시 문의해주세요.")
    
    def _update_success_rate(self, success: bool):
        self.ewma_success_rate = (
            self.ewma_alpha * (1.0 if success else 0.0) + 
            (1 - self.ewma_alpha) * self.ewma_success_rate
        )
        
        if self.ewma_success_rate > 0.9:
            adjustment = "+0.02"
        elif self.ewma_success_rate < 0.7:
            adjustment = "-0.02"
        else:
            adjustment = "+0.00"
        
        logger.info(f"📊 EWMA 성공률 업데이트: {self.ewma_success_rate:.3f}, 제안 조정: {adjustment}")

# -----------------------------------------------------------------
# 인스턴스 생성 함수
# -----------------------------------------------------------------

# NOTE: 이전에 무한 재귀 호출을 일으켰던 함수를 제거하거나 주석 처리해야 합니다.
# def fallback_handler() -> fallback_handler:
#     """Fallback 핸들러 인스턴스 생성"""
#     return fallback_handler()

# -----------------------------------------------------------------
# 테스트 코드 (개발용)
# -----------------------------------------------------------------
# ... (기존 테스트 코드)