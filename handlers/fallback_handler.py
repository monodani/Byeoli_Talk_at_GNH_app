#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - fallback_handler

미매칭/저신뢰 답변 처리 전용 핸들러
모든 전문 핸들러가 실패했을 때 최후의 보루 역할

주요 특징:
- 컨피던스 임계값 θ=0.00 (항상 작동)
- EWMA 기반 자동 튜닝 지원 (±0.02)
- 적절한 담당부서 안내 및 재질문 유도
- 검색 실패 시 일반적인 기관 정보 제공
- 대화 맥락을 고려한 안내 강화
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# 프로젝트 모듈
from handlers.base_handler import base_handler
from utils.contracts import QueryRequest, HandlerResponse

# 로깅 설정
logger = logging.getLogger(__name__)


class fallback_handler(base_handler):
    """
    최종 백업 핸들러
    
    처리 범위:
    - 모든 전문 핸들러 실패 시 작동
    - 일반적인 기관 정보 제공
    - 적절한 담당부서 연결
    - 재질문 유도 및 도움말 제공
    
    특징:
    - 최저 컨피던스 임계값 (θ=0.00)
    - 항상 성공 보장 (실패하지 않음)
    - EWMA 자동 튜닝 지원
    - 대화 맥락 고려한 맞춤형 안내
    """
    
    def __init__(self):
        super().__init__(
            domain="fallback",
            index_name="general_index",  # general 인덱스 활용
            confidence_threshold=0.00
        )
        
        # 질문 유형별 매핑
        self.question_categories = {
            'satisfaction': {
                'keywords': ['만족도', '평가', '점수', '성과', '결과'],
                'department': '평가분석담당',
                'contact': '055-254-2022',
                'description': '교육평가 및 만족도 조사'
            },
            'planning': {
                'keywords': ['계획', '일정', '교육과정', '신청', '모집'],
                'department': '교육기획담당',
                'contact': '055-254-2052',
                'description': '교육훈련계획 및 운영'
            },
            'cyber': {
                'keywords': ['사이버', '온라인', '이러닝', '민간위탁', '나라배움터'],
                'department': '사이버담당',
                'contact': '055-254-2052',
                'description': '사이버교육 운영'
            },
            'facility': {
                'keywords': ['식당', '기숙사', '주차', '시설', '의무실'],
                'department': '총무담당',
                'contact': '055-254-2096',
                'description': '시설 관리 및 운영'
            },
            'general': {
                'keywords': ['학칙', '규정', '연락처', '위치', '찾아오는'],
                'department': '교육기획담당',
                'contact': '055-254-2052',
                'description': '일반 업무'
            }
        }
        
        # 기본 기관 정보
        self.institution_info = {
            'name': '경상남도인재개발원',
            'address': '경상남도 진주시 동진로 248',
            'main_phone': '055-254-2000',
            'website': 'https://gnhi.go.kr',
            'description': '경상남도 공무원 교육훈련 전문기관'
        }
        
        # EWMA 자동 튜닝을 위한 성공률 추적
        self.ewma_alpha = 0.1  # 평활화 계수
        self.success_rate = 0.7  # 초기 성공률
        
        logger.info("🛡️ fallback_handler 초기화 완료 (θ=0.00)")
    
    def get_system_prompt(self) -> str:
        """fallback 전용 시스템 프롬프트"""
        return """당신은 "벼리(영문명: Byeoli)"입니다. 경상남도인재개발원의 종합 안내 챗봇으로, 구체적인 답변을 제공하기 어려운 질문에 대해 적절한 안내와 도움을 제공하는 역할을 합니다.

다음 지침을 따라 도움이 되는 응답을 제공하세요:

1. **현재 상황 인정**: 구체적인 정보를 찾지 못했음을 솔직하게 인정하세요.

2. **대안 제시**: 질문 내용을 분석하여 가장 적절한 담당부서를 안내하세요.

3. **담당부서별 연락처**:
   - 총무담당 (055-254-2013): 시설 관리, 구내식당, 기숙사
   - 평가분석담당 (055-254-2023): 교육평가, 만족도 조사, 성과분석
   - 교육기획담당 (055-254-2053): 교육훈련계획, 교육과정 운영, 일반 업무
   - 교육운영1담당 (055-254-2063): 신규 임용(후보)자, 리더십 및 역량 교육과정
   - 교육운영2담당 (055-254-2073): 중견리더 과정, 직무역량 및 핵심과제 교육과정  
   - 사이버담당 (055-254-2083): 사이버교육, 온라인 교육

4. **재질문 유도**: 더 구체적인 질문을 할 수 있도록 안내하세요.

5. **기관 정보 제공**: 필요시 기본적인 기관 정보를 제공하세요.
   - 주소: 경상남도 진주시 월아산로 2026
   - 대표전화: (인재개발지원과) 055-254-2011, (인재양성과) 055-254-2051
   - 홈페이지: https://www.gyeongnam.go.kr/hrd/index.gyeong

6. **친절한 마무리**: 추가 도움이 필요하면 언제든 문의하도록 안내하세요.

7. **응답 형식**:
   ```
   💭 죄송합니다. [구체적인 상황 설명]
   
   📞 관련 문의처:
   • [담당부서]: [연락처] ([업무 범위])
   
   💡 더 정확한 답변을 위해:
   [구체적인 재질문 제안]
   
   📋 기관 정보:
   [필요시 기본 정보 제공]
   ```

8. **추가 도움**: "추가로 궁금한 사항이 있으시면 언제든 말씀해 주세요!"로 마무리하세요."""

    def format_context(self, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """fallback용 컨텍스트 포맷 (기본 기관 정보 위주)"""
        context_parts = []
        
        # 검색 결과가 있는 경우 활용
        if search_results:
            context_parts.append("=== 📋 관련 정보 ===")
            for i, (text, score, metadata) in enumerate(search_results[:3], 1):
                source_info = metadata.get('source_file', '기관 정보')
                context_parts.append(f"[참고 {i}: {source_info}]")
                context_parts.append(f"{text[:200]}...")
                context_parts.append("")
        
        # 기본 기관 정보 추가
        context_parts.append("=== 🏢 기관 정보 ===")
        context_parts.append(f"기관명: {self.institution_info['name']}")
        context_parts.append(f"주소: {self.institution_info['address']}")
        context_parts.append(f"대표전화: {self.institution_info['main_phone']}")
        context_parts.append(f"홈페이지: {self.institution_info['website']}")
        context_parts.append(f"설명: {self.institution_info['description']}")
        context_parts.append("")
        
        # 담당부서 정보
        context_parts.append("=== 📞 주요 담당부서 ===")
        context_parts.append("• 교육기획담당 (055-254-2052): 교육훈련계획, 교육과정 운영")
        context_parts.append("• 평가분석담당 (055-254-2022): 교육평가, 만족도 조사")
        context_parts.append("• 사이버담당 (055-254-2052): 사이버교육 운영")
        context_parts.append("• 총무담당 (055-254-2096): 시설 관리, 구내식당")
        
        return "\n".join(context_parts)
    
    def _detect_question_category(self, query: str) -> Optional[Dict[str, str]]:
        """질문에서 가장 적절한 담당부서 카테고리 감지"""
        query_lower = query.lower()
        
        best_category = None
        best_score = 0
        
        for category, info in self.question_categories.items():
            score = sum(1 for keyword in info['keywords'] if keyword in query_lower)
            if score > best_score:
                best_score = score
                best_category = info
        
        return best_category if best_score > 0 else self.question_categories['general']
    
    def _generate_reask_suggestions(self, query: str, category_info: Dict[str, str]) -> str:
        """재질문 제안 생성"""
        suggestions = []
        
        if category_info['department'] == '평가분석담당':
            suggestions.extend([
                "구체적인 교육과정명을 포함해 주세요",
                "어떤 연도의 만족도 결과를 원하시는지 알려주세요",
                "전반만족도, 역량향상도 등 구체적인 평가 항목을 명시해 주세요"
            ])
        elif category_info['department'] == '교육기획담당':
            suggestions.extend([
                "원하시는 교육과정명이나 분야를 구체적으로 알려주세요",
                "교육 신청, 일정 확인 등 구체적인 목적을 말씀해 주세요",
                "특정 기간이나 날짜가 있다면 함께 알려주세요"
            ])
        elif category_info['department'] in ['교육운영1담당', '교육운영2담당']:
            suggestions.extend([
                "신규 임용자인지 중견리더인지 대상을 명확히 해주세요",
                "관심 있는 교육과정명이나 분야를 구체적으로 알려주세요",
                "교육 신청, 일정 문의 등 구체적인 목적을 말씀해 주세요"
            ])
        elif category_info['department'] == '사이버담당':
            suggestions.extend([
                "민간위탁 또는 나라배움터 중 어떤 유형인지 알려주세요",
                "관심 있는 교육 분야나 주제를 구체적으로 말씀해 주세요",
                "학습시간이나 평가 여부 등 특별한 조건이 있다면 알려주세요"
            ])
        else:
            suggestions.extend([
                "더 구체적인 정보나 상황을 설명해 주세요",
                "찾고 계신 정보의 정확한 목적을 알려주세요",
                "관련된 문서나 자료명이 있다면 함께 말씀해 주세요"
            ])
        
        return "\n".join([f"• {suggestion}" for suggestion in suggestions[:3]])
    
    def _update_success_rate(self, success: bool):
        """EWMA를 사용한 성공률 업데이트"""
        current_success = 1.0 if success else 0.0
        self.success_rate = (1 - self.ewma_alpha) * self.success_rate + self.ewma_alpha * current_success
        
        # 임계값 자동 조정 로직 (다른 핸들러들을 위한 참고용)
        if self.success_rate < 0.6:
            suggested_adjustment = -0.02
        elif self.success_rate > 0.8:
            suggested_adjustment = +0.02
        else:
            suggested_adjustment = 0.0
        
        logger.info(f"📊 EWMA 성공률 업데이트: {self.success_rate:.3f}, 제안 조정: {suggested_adjustment:+.2f}")
    
    def handle(self, request: QueryRequest) -> HandlerResponse:
        """
        fallback 핸들러 처리
        항상 성공하며, 적절한 안내 정보 제공
        """
        start_time = time.time()
        
        try:
            logger.info(f"🛡️ fallback 핸들러 처리 시작: {request.text[:50]}...")
            
            # 1. 질문 카테고리 감지
            category_info = self._detect_question_category(request.text)
            
            # 2. 간단한 검색 시도 (general 인덱스 활용)
            search_results = self.hybrid_search(request.text, k=5)
            
            # 3. 항상 성공으로 간주 (confidence = 1.0)
            confidence = 1.0
            
            # 4. 기본 Citation (검색 결과가 있는 경우)
            citations = self.extract_citations(search_results) if search_results else []
            
            # 5. fallback 전용 응답 생성
            system_prompt = self.get_system_prompt()
            context = self.format_context(search_results)
            
            # 재질문 제안 생성
            reask_suggestions = self._generate_reask_suggestions(request.text, category_info)
            
            # 맞춤형 fallback 메시지 생성
            fallback_message = f"""💭 죄송합니다. 질문하신 내용에 대한 구체적인 정보를 찾지 못했습니다.

📞 관련 문의처:
• {category_info['department']}: {category_info['contact']} ({category_info['description']})

💡 더 정확한 답변을 위해 다음과 같이 질문해 주세요:
{reask_suggestions}

📋 기관 정보:
• 기관명: {self.institution_info['name']}
• 주소: {self.institution_info['address']}
• 대표전화: {self.institution_info['main_phone']}
• 홈페이지: {self.institution_info['website']}

🔍 다른 방법으로 도움을 드릴 수 있습니다:
• 교육과정 관련: "2025년 교육계획" 또는 "리더십 교육"
• 만족도 관련: "교육만족도 결과" 또는 "과정별 평가"
• 사이버교육 관련: "민간위탁 교육" 또는 "나라배움터"
• 시설 관련: "구내식당 메뉴" 또는 "기숙사 정보"

추가로 궁금한 사항이 있으시면 언제든 말씀해 주세요! 😊"""
            
            # 6. HandlerResponse 생성
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            handler_response = HandlerResponse(
                answer=fallback_message,
                citations=citations,
                confidence=confidence,
                handler_id=self.domain,
                elapsed_ms=elapsed_ms,
                reask=f"더 구체적인 질문을 해주세요. 예: {reask_suggestions.split()[1] if reask_suggestions else '구체적인 내용'}",
                diagnostics={
                    "category": category_info['department'],
                    "search_results_count": len(search_results),
                    "ewma_success_rate": self.success_rate
                }
            )
            
            # 7. 성공률 업데이트 (fallback은 항상 성공으로 간주)
            self._update_success_rate(True)
            
            logger.info(f"✅ fallback 핸들러 처리 완료 ({elapsed_ms}ms, 카테고리={category_info['department']})")
            return handler_response
            
        except Exception as e:
            logger.error(f"❌ fallback 핸들러 예외 발생: {e}")
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # 최후의 최후 응답 (절대 실패하지 않음)
            emergency_response = HandlerResponse(
                answer=f"""시스템 오류가 발생했습니다. 직접 문의해 주세요.

📞 대표전화: {self.institution_info['main_phone']}
🌐 홈페이지: {self.institution_info['website']}

주요 담당부서:
• 총무담당: 055-254-2013
• 평가분석담당: 055-254-2023  
• 교육기획담당: 055-254-2053
• 교육운영1담당: 055-254-2063
• 교육운영2담당: 055-254-2073
• 사이버담당: 055-254-2083

죄송합니다. 😔""",
                citations=[],
                confidence=1.0,
                handler_id=self.domain,
                elapsed_ms=elapsed_ms,
                diagnostics={"error": str(e)}
            )
            
            self._update_success_rate(False)
            return emergency_response


# ================================================================
# 테스트 코드 (개발용)
# ================================================================

if __name__ == "__main__":
    """fallback_handler 개발 테스트"""
    print("🛡️ Fallback Handler 테스트 시작")
    
    test_queries = [
        "알 수 없는 질문입니다",
        "이상한 교육과정이 있나요?",
        "만족도가 궁금해요",  # 평가분석담당 안내
        "교육 신청하려면 어떻게 해야 하나요?",  # 교육기획담당 안내
        "사이버교육 찾고 있어요",  # 사이버담당 안내
        "구내식당 위치가 어디인가요?",  # 총무담당 안내
        "완전히 관련 없는 질문",
    ]
    
    handler = fallback_handler()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n=== 테스트 {i}: {query} ===")
        
        try:
            from utils.contracts import QueryRequest
            import uuid
            
            request = QueryRequest(
                text=query,
                context=None,
                follow_up=False,
                trace_id=str(uuid.uuid4())
            )
            
            response = handler.handle(request)
            print(f"응답: {response.answer[:200]}...")
            print(f"컨피던스: {response.confidence:.3f}")
            print(f"소요시간: {response.elapsed_ms}ms")
            print(f"재질문: {response.reask}")
            
            if response.diagnostics:
                print(f"진단정보: {response.diagnostics}")
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
    
    print("\n✅ Fallback 핸들러 테스트 완료")
