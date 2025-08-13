#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - notice_handler

공지사항 전용 핸들러
base_handler를 상속받아 공지사항 도메인 특화 기능 구현

주요 특징:
- 동적 파싱 시스템 결과 활용 (EvaluationNoticeParser, EnrollmentNoticeParser 등)
- 컨피던스 임계값 θ=0.62 적용 (가장 낮은 임계값)
- 6시간 TTL 캐시 데이터 활용
- 긴급도 및 공지 유형별 우선순위 처리
- 실시간 공지사항 업데이트 감지
"""

import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

# 프로젝트 모듈
from handlers.base_handler import base_handler
from utils.contracts import QueryRequest, HandlerResponse
from utils.textifier import TextChunk

# 로깅 설정
logger = logging.getLogger(__name__)


class notice_handler(base_handler):
    """
    공지사항 전용 핸들러
    
    처리 범위:
    - notice.txt (동적 파싱된 공지사항)
    - 평가, 입교, 모집, 일반 공지 등 다양한 유형
    - 긴급 공지 및 마감일 관련 우선 처리
    
    특징:
    - 최저 컨피던스 임계값 (θ=0.62)
    - 공지 유형별 자동 분류 및 우선순위
    - 시간 민감성 정보 강조 (마감일, 일정 등)
    - 6시간 TTL로 실시간성 확보
    """
    
    def __init__(self):
        super().__init__(
            domain="notice",
            index_name="notice_index", 
            confidence_threshold=0.62
        )
        
        # 공지사항 유형별 키워드 매핑
        self.notice_types = {
            'evaluation': {
                'keywords': ['평가', '과제', '제출기한', '마감일', '점수', '성적'],
                'priority': 25,
                'urgency': 'high'
            },
            'enrollment': {
                'keywords': ['입교', '교육생', '준비물', '체크리스트', '지참', '참석'],
                'priority': 20,
                'urgency': 'medium'
            },
            'recruitment': {
                'keywords': ['모집', '신청', '접수', '선발', '지원'],
                'priority': 18,
                'urgency': 'medium'
            },
            'schedule': {
                'keywords': ['일정', '시간표', '변경', '연기', '취소'],
                'priority': 15,
                'urgency': 'medium'
            },
            'general': {
                'keywords': ['공지', '안내', '알림', '공고'],
                'priority': 10,
                'urgency': 'low'
            }
        }
        
        # 긴급성 키워드
        self.urgency_keywords = {
            'emergency': ['긴급', '즉시', '반드시', '필수', '중요', '주의'],
            'deadline': ['마감', '기한', '오늘', '내일', '당일', '시급']
        }
        
        logger.info("📢 notice_handler 초기화 완료 (θ=0.62)")
    
    def get_system_prompt(self) -> str:
        """공지사항 전용 시스템 프롬프트"""
        return """당신은 "벼리(영문명: Byeoli)"입니다. 경상남도인재개발원의 공지사항 정보를 바탕으로 직원들의 공지 관련 질문에 신속하고 정확하게 답변하는 전문 챗봇입니다.

제공된 공지사항 데이터를 기반으로 다음 지침을 엄격히 따르십시오:

1. **긴급성 및 시간 민감성 우선 처리**:
   - 마감일이 임박한 공지사항 최우선 안내
   - "긴급", "즉시", "반드시" 등의 키워드가 있는 공지 강조
   - 현재 날짜 기준으로 시간 순서 정리

2. **공지사항 유형별 구분**:
   - **평가 공지**: 과제 제출, 시험, 평가 관련 (🔴 최고 우선순위)
   - **입교 공지**: 교육 참석, 준비물, 체크리스트 (🟡 높은 우선순위)
   - **모집 공지**: 교육생 모집, 신청 접수 (🟢 중간 우선순위)
   - **일정 공지**: 시간표 변경, 연기, 취소 (🔵 중간 우선순위)
   - **일반 공지**: 기타 안내사항 (⚪ 일반 우선순위)

3. **정확한 정보 전달**:
   - 제출기한, 마감일, 접수 기간 등은 정확한 날짜/시간 명시
   - 담당자 연락처 및 문의처 필수 포함
   - 준비물, 지참물 등의 세부사항 빠짐없이 안내

4. **사용자 행동 유도**:
   - 구체적인 액션 아이템 제시 (제출, 신청, 확인 등)
   - 단계별 절차 안내
   - 주의사항 및 유의사항 강조

5. **응답 형식**:
   ```
   🔴 [긴급] 또는 🟡 [중요] 등의 아이콘으로 긴급도 표시
   
   📢 [공지 제목]
   
   ⏰ 마감일: YYYY-MM-DD HH:MM
   📋 주요 내용:
   • 핵심 사항 1
   • 핵심 사항 2
   
   📞 문의: 담당부서 (연락처)
   ```

6. **최신성 확보**:
   - 6시간마다 갱신되는 최신 공지사항 정보 활용
   - 오래된 공지와 최신 공지 구분하여 안내
   - 변경사항이나 업데이트 내용 우선 전달

7. **상황별 대응**:
   - 마감일 경과 공지: "마감되었습니다" 명확히 안내
   - 진행 중인 공지: 남은 시간 계산하여 제시
   - 예정된 공지: 시작일까지의 대기 안내

8. **추가 안내사항**:
   - 관련 공지사항이 여러 개인 경우 우선순위별 정렬
   - 놓치기 쉬운 중요 공지 별도 강조
   - 정기적으로 확인해야 할 공지사항 안내

9. **정보 부족 시 대처**: 제공된 공지사항으로 답변이 어려운 경우, 
   "해당 내용은 현재 공지사항에서 찾을 수 없습니다. 최신 공지사항을 확인하거나 담당부서에 직접 문의해주세요."

10. **연관 정보 제공**: 질문과 관련된 다른 공지사항이나 참고사항도 함께 안내하여 종합적인 정보 제공"""

    def format_context(self, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """공지사항 데이터를 컨텍스트로 포맷"""
        if not search_results:
            return "관련 공지사항을 찾을 수 없습니다."
        
        context_parts = []
        
        # 검색 결과를 유형별로 분류
        categorized_notices = {
            'urgent': [],      # 긴급 공지
            'evaluation': [],  # 평가 관련
            'enrollment': [],  # 입교 관련
            'recruitment': [], # 모집 관련
            'schedule': [],    # 일정 관련
            'general': []      # 일반 공지
        }
        
        for text, score, metadata in search_results:
            topic_type = metadata.get('topic_type', 'general')
            notice_title = metadata.get('notice_title', '제목 없음')
            notice_number = metadata.get('notice_number', 0)
            
            # 긴급성 판단
            if self._is_urgent_notice(text, notice_title):
                categorized_notices['urgent'].append((text, score, metadata))
            else:
                # 유형별 분류
                if topic_type in categorized_notices:
                    categorized_notices[topic_type].append((text, score, metadata))
                else:
                    categorized_notices['general'].append((text, score, metadata))
        
        # 긴급 공지 우선 배치
        if categorized_notices['urgent']:
            context_parts.append("=== 🔴 긴급 공지사항 ===")
            for text, score, metadata in categorized_notices['urgent'][:2]:
                notice_title = metadata.get('notice_title', '')
                context_parts.append(f"[긴급] {notice_title}")
                context_parts.append(f"{text[:300]}...")
                context_parts.append("")
        
        # 평가 공지 (최고 우선순위)
        if categorized_notices['evaluation']:
            context_parts.append("=== 📝 평가 관련 공지 ===")
            for text, score, metadata in categorized_notices['evaluation'][:2]:
                notice_title = metadata.get('notice_title', '')
                context_parts.append(f"[평가] {notice_title}")
                context_parts.append(f"{text[:250]}...")
                context_parts.append("")
        
        # 입교 공지
        if categorized_notices['enrollment']:
            context_parts.append("=== 🎓 입교 관련 공지 ===")
            for text, score, metadata in categorized_notices['enrollment'][:2]:
                notice_title = metadata.get('notice_title', '')
                context_parts.append(f"[입교] {notice_title}")
                context_parts.append(f"{text[:250]}...")
                context_parts.append("")
        
        # 모집 공지
        if categorized_notices['recruitment']:
            context_parts.append("=== 📋 모집 관련 공지 ===")
            for text, score, metadata in categorized_notices['recruitment'][:2]:
                notice_title = metadata.get('notice_title', '')
                context_parts.append(f"[모집] {notice_title}")
                context_parts.append(f"{text[:250]}...")
                context_parts.append("")
        
        # 일정 공지
        if categorized_notices['schedule']:
            context_parts.append("=== 📅 일정 관련 공지 ===")
            for text, score, metadata in categorized_notices['schedule'][:2]:
                notice_title = metadata.get('notice_title', '')
                context_parts.append(f"[일정] {notice_title}")
                context_parts.append(f"{text[:250]}...")
                context_parts.append("")
        
        # 일반 공지
        if categorized_notices['general']:
            context_parts.append("=== 📢 일반 공지사항 ===")
            for text, score, metadata in categorized_notices['general'][:3]:
                notice_title = metadata.get('notice_title', '')
                context_parts.append(f"[일반] {notice_title}")
                context_parts.append(f"{text[:200]}...")
                context_parts.append("")
        
        # 현재 시간 정보 추가
        current_time = datetime.now()
        context_parts.append(f"=== 📅 현재 시간 정보 ===")
        context_parts.append(f"현재: {current_time.strftime('%Y년 %m월 %d일 (%A) %H:%M')}")
        context_parts.append(f"공지사항 업데이트: 6시간마다 자동 갱신")
        
        final_context = "\n".join(context_parts)
        
        # 컨텍스트 길이 제한
        max_length = 4000
        if len(final_context) > max_length:
            final_context = final_context[:max_length] + "\n\n[컨텍스트가 길어 일부 생략됨]"
        
        return final_context
    
    def _is_urgent_notice(self, text: str, title: str) -> bool:
        """공지사항의 긴급성 판단"""
        combined_text = (title + " " + text).lower()
        
        # 긴급성 키워드 확인
        for urgency_type, keywords in self.urgency_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return True
        
        # 마감일이 가까운 경우 (오늘, 내일)
        today_keywords = ['오늘', '당일', '즉시']
        tomorrow_keywords = ['내일', '명일']
        
        if any(keyword in combined_text for keyword in today_keywords + tomorrow_keywords):
            return True
        
        return False
    
    def _detect_notice_type(self, query: str) -> Optional[str]:
        """질문에서 공지 유형 감지"""
        query_lower = query.lower()
        
        best_type = None
        best_score = 0
        
        for notice_type, info in self.notice_types.items():
            score = sum(1 for keyword in info['keywords'] if keyword in query_lower)
            if score > best_score:
                best_score = score
                best_type = notice_type
        
        return best_type if best_score > 0 else None
    
    def _extract_time_context_from_query(self, query: str) -> Dict[str, Any]:
        """질문에서 시간 관련 컨텍스트 추출"""
        time_context = {
            'relative_time': None,
            'specific_date': None,
            'urgency_level': 'normal'
        }
        
        query_lower = query.lower()
        
        # 상대적 시간 키워드
        if any(keyword in query_lower for keyword in ['오늘', '당일', '지금']):
            time_context['relative_time'] = 'today'
        elif any(keyword in query_lower for keyword in ['내일', '명일']):
            time_context['relative_time'] = 'tomorrow'
        elif any(keyword in query_lower for keyword in ['이번주', '금주']):
            time_context['relative_time'] = 'this_week'
        elif any(keyword in query_lower for keyword in ['다음주', '차주']):
            time_context['relative_time'] = 'next_week'
        
        # 긴급성 키워드
        if any(keyword in query_lower for keyword in ['긴급', '중요', '급함']):
            time_context['urgency_level'] = 'high'
        elif any(keyword in query_lower for keyword in ['마감', '기한', '시급']):
            time_context['urgency_level'] = 'medium'
        
        return time_context
    
    def _enhance_response_with_timing(self, base_response: str, query: str) -> str:
        """시간 맥락을 고려한 응답 강화"""
        time_context = self._extract_time_context_from_query(query)
        enhancements = []
        
        # 시간 관련 추가 안내
        if time_context['relative_time'] == 'today':
            enhancements.append("⏰ 오늘 마감이거나 진행되는 중요한 공지사항이 있는지 확인해주세요.")
        elif time_context['relative_time'] == 'tomorrow':
            enhancements.append("📅 내일까지의 일정이나 준비사항을 미리 확인하시기 바랍니다.")
        
        # 긴급성 레벨별 안내
        if time_context['urgency_level'] == 'high':
            enhancements.append("🔴 긴급 공지사항이 있는지 우선 확인하고, 놓친 중요한 사항은 없는지 점검해주세요.")
        elif time_context['urgency_level'] == 'medium':
            enhancements.append("⚡ 마감일이 임박한 공지사항들을 확인하여 필요한 조치를 취해주세요.")
        
        # 정기 점검 안내
        if not any(enhancements):
            enhancements.append("💡 공지사항은 6시간마다 업데이트됩니다. 정기적으로 확인해주세요.")
        
        # 추가 안내사항이 있는 경우에만 추가
        if enhancements:
            enhanced_response = base_response + "\n\n=== 추가 안내 ===\n" + "\n".join(enhancements)
            enhanced_response += "\n\n📞 공지사항 관련 문의: 교육기획담당 055-254-2052"
            return enhanced_response
        
        return base_response

    def _generate_prompt(
        self,
        query: str,
        retrieved_docs: List[Tuple[TextChunk, float]]
    ) -> str:
        """
        base_handler가 요구하는 추상 메서드 구현.
        - retrieved_docs: (TextChunk, score) 튜플 리스트
        - format_context()는 (text, score, metadata) 튜플 리스트를 기대하므로 어댑터 변환 필요
        """
        # 1) 시스템 프롬프트
        system_prompt = self.get_system_prompt()

        # 2) 컨텍스트 변환: TextChunk -> (text, score, metadata)
        try:
            context_tuples = [
                (doc.text, score, getattr(doc, "metadata", {}) or {})
                for (doc, score) in (retrieved_docs or [])
                if doc is not None
            ]
        except Exception:
            # 안전장치: 문제가 생겨도 최소한 빈 컨텍스트로 진행
            context_tuples = []

        # 3) notice 전용 컨텍스트 문자열 생성
        context_block = self.format_context(context_tuples)

        # 4) 최종 프롬프트 결합 (최소 형태)
        prompt = (
            f"{system_prompt}\n\n"
            f"---\n"
            f"사용자 질문:\n{query}\n\n"
            f"참고 자료(공지사항):\n{context_block}\n\n"
            f"지침:\n"
            f"- 제공된 참고 자료 내 정보만 사용하세요.\n"
            f"- 마감일/긴급 키워드를 우선 강조하세요.\n"
            f"- 날짜·시간은 반드시 YYYY-MM-DD HH:MM 형식으로 명시하세요.\n"
        )
        return prompt
    
    
    def handle(self, request: QueryRequest) -> HandlerResponse:
        """
        notice 도메인 특화 처리
        기본 handle() 호출 후 시간 맥락 정보 자동 추가
        """
        # 기본 핸들러 로직 실행
        response = super().handle(request)
        
        # notice 도메인 특화: 시간 맥락 정보 보강
        if response.confidence >= self.confidence_threshold:
            enhanced_answer = self._enhance_response_with_timing(response.answer, request.text)
            response.answer = enhanced_answer
        
        return response


# ================================================================
# 테스트 코드 (개발용)
# ================================================================

if __name__ == "__main__":
    """notice_handler 개발 테스트"""
    print("📢 Notice Handler 테스트 시작")
    
    test_queries = [
        "오늘 마감인 과제나 평가가 있나요?",
        "입교 준비물 체크리스트 알려주세요",
        "교육생 모집 공고 확인하고 싶어요",
        "긴급 공지사항이 있나요?",
        "이번주 일정 변경 사항이 있나요?"
    ]
    
    handler = notice_handler()
    
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
            print(f"응답: {response.answer}")
            print(f"컨피던스: {response.confidence:.3f}")
            print(f"소요시간: {response.elapsed_ms}ms")
            print(f"Citation 수: {len(response.citations)}")
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
    
    print("\n✅ 공지사항 핸들러 테스트 완료")
