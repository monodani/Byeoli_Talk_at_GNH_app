#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - menu_handler

구내식당 식단표 전용 핸들러
base_handler를 상속받아 식단 도메인 특화 기능 구현

주요 특징:
- ChatGPT API 기반 이미지 파싱 결과 활용
- 요일별/식사별 식단 정보 제공
- 컨피던스 임계값 θ=0.64 적용
- 6시간 TTL 캐시 데이터 활용
- 자연어 식단 질문 처리 최적화
"""

import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

# 프로젝트 모듈
from handlers.base_handler import base_handler
from utils.contracts import QueryRequest, HandlerResponse

# 로깅 설정
logger = logging.getLogger(__name__)


class menu_handler(base_handler):
    """
    구내식당 식단표 전용 핸들러
    
    처리 범위:
    - menu.png (ChatGPT API 파싱된 주간 식단표)
    - menu_YYYYWW.txt (캐시된 주차별 식단 데이터)
    - 요일별/식사별 식단 검색 및 추천
    
    특징:
    - 낮은 컨피던스 임계값 (θ=0.64)
    - 자연어 식단 질문 최적화
    - 시간 기반 현재 식사 추론
    - 영양 정보 및 메뉴 설명 포함
    """
    
    def __init__(self):
        super().__init__(
            domain="menu",
            index_name="menu_index", 
            confidence_threshold=0.64
        )
        
        # 요일 및 식사 매핑 사전
        self.day_keywords = {
            '월': '월요일', '월요일': '월요일',
            '화': '화요일', '화요일': '화요일', 
            '수': '수요일', '수요일': '수요일',
            '목': '목요일', '목요일': '목요일',
            '금': '금요일', '금요일': '금요일',
            '토': '토요일', '토요일': '토요일',
            '일': '일요일', '일요일': '일요일'
        }
        
        self.meal_keywords = {
            '조식': '조식', '아침': '조식', '모닝': '조식',
            '중식': '중식', '점심': '중식', '런치': '중식',
            '석식': '석식', '저녁': '석식', '디너': '석식'
        }
        
        logger.info("🍽️ menu_handler 초기화 완료 (θ=0.64)")
    
    def get_system_prompt(self) -> str:
        """식단표 전용 시스템 프롬프트"""
        return """당신은 "벼리(영문명: Byeoli)"입니다. 경상남도인재개발원 구내식당의 식단표 정보를 바탕으로 직원들의 식사 관련 질문에 친절하고 실용적으로 답변하는 전문 챗봇입니다.

제공된 식단표 데이터를 기반으로 다음 지침을 엄격히 따르십시오:

1. **정확한 식단 정보 제공**: 제공된 데이터 내의 식단 정보만 정확하게 안내하세요. 없는 정보는 추측하지 마세요.

2. **시간 맥락 고려**: 현재 시간과 요일을 고려하여 적절한 식사를 추천하세요.
   - 오전 9시 이전: 조식 우선 안내
   - 오전 9시~오후 2시: 중식 우선 안내  
   - 오후 2시 이후: 석식 우선 안내

3. **요일별 식단 구성**: 월요일부터 금요일까지의 식단을 명확히 구분하여 제시하세요.

4. **친근하고 실용적인 안내**: 
   - 메뉴 이름과 함께 간단한 설명 추가
   - 특별한 메뉴나 추천 요리 강조
   - 식사 시간 및 구내식당 위치 정보 포함

5. **식단 정보 부족 시**: 데이터에 없는 정보는 솔직하게 "해당 정보가 식단표에 없습니다"라고 안내하고, 구내식당에 직접 문의하도록 안내하세요.

6. **영양 및 건강 고려**: 가능한 경우 균형 잡힌 식사 조합을 추천하세요.

7. **응답 형식**:
   ```
   🍽️ [요일] [식사] 메뉴
   
   주요 메뉴:
   • 메뉴1 - 간단 설명
   • 메뉴2 - 간단 설명
   
   📍 구내식당 위치: [위치 정보]
   ⏰ 식사 시간: [시간 정보]
   ```

8. **주간 식단 요약**: 전체 주간 식단을 문의하는 경우, 요일별로 정리하여 한눈에 보기 쉽게 제시하세요.

9. **식단 변경 알림**: 식단 변경이나 특별 메뉴가 있는 경우 강조하여 안내하세요.

10. **구내식당 관련 추가 정보**: 필요시 다음 정보도 함께 제공하세요.
    - 구내식당 문의전화: 055-254-2096 (총무담당)
    - 특별 이벤트나 행사 메뉴 안내"""

    def format_context(self, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """식단표 데이터를 컨텍스트로 포맷"""
        if not search_results:
            return "관련 식단표 정보를 찾을 수 없습니다."
        
        context_parts = []
        
        # 검색 결과를 유형별로 분류
        weekly_summary = []  # 주간 요약
        daily_meals = []     # 일일 식사
        
        for text, score, metadata in search_results:
            chunk_type = metadata.get('chunk_type', 'unknown')
            
            if chunk_type == 'weekly_summary':
                weekly_summary.append((text, score, metadata))
            elif chunk_type == 'meal_detail':
                daily_meals.append((text, score, metadata))
            else:
                # 기타 식단 관련 정보
                context_parts.append(f"[식단정보] {text}")
        
        # 주간 요약 우선 배치
        if weekly_summary:
            context_parts.append("=== 주간 식단 요약 ===")
            for text, score, metadata in weekly_summary[:2]:  # 상위 2개
                week = metadata.get('week', '')
                context_parts.append(f"[{week} 주간식단] {text}")
        
        # 상세 식사 정보
        if daily_meals:
            context_parts.append("\n=== 상세 식사 정보 ===")
            # 요일 및 식사별로 정렬
            sorted_meals = sorted(daily_meals, key=lambda x: (
                x[2].get('day', ''), 
                x[2].get('meal_type', ''),
                -x[1]  # 점수 내림차순
            ))
            
            for text, score, metadata in sorted_meals[:6]:  # 상위 6개
                day = metadata.get('day', '')
                meal_type = metadata.get('meal_type', '')
                menu_count = metadata.get('menu_count', 0)
                context_parts.append(f"[{day} {meal_type} - {menu_count}개 메뉴] {text}")
        
        # 현재 시간 정보 추가
        current_time = datetime.now()
        context_parts.append(f"\n=== 현재 시간 정보 ===")
        context_parts.append(f"현재: {current_time.strftime('%Y년 %m월 %d일 (%a) %H:%M')}")
        context_parts.append(f"현재 요일: {self._get_korean_weekday(current_time.weekday())}")
        context_parts.append(f"추천 식사: {self._get_recommended_meal_time(current_time.hour)}")
        
        final_context = "\n\n".join(context_parts)
        
        # 컨텍스트 길이 제한
        max_length = 3500
        if len(final_context) > max_length:
            final_context = final_context[:max_length] + "\n\n[컨텍스트가 길어 일부 생략됨]"
        
        return final_context
    
    def _get_korean_weekday(self, weekday: int) -> str:
        """weekday 숫자를 한글 요일로 변환 (0=월요일)"""
        weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
        return weekdays[weekday] if 0 <= weekday <= 6 else '알 수 없음'
    
    def _get_recommended_meal_time(self, hour: int) -> str:
        """현재 시간 기준 추천 식사"""
        if hour < 9:
            return "조식"
        elif hour < 14:
            return "중식"
        else:
            return "석식"
    
    def _extract_day_meal_from_query(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """질문에서 요일과 식사 타입 추출"""
        query_lower = query.lower()
        
        # 요일 추출
        detected_day = None
        for keyword, standard_day in self.day_keywords.items():
            if keyword in query_lower:
                detected_day = standard_day
                break
        
        # 식사 타입 추출
        detected_meal = None
        for keyword, standard_meal in self.meal_keywords.items():
            if keyword in query_lower:
                detected_meal = standard_meal
                break
        
        return detected_day, detected_meal
    
    def _enhance_response_with_time_context(self, base_response: str, query: str) -> str:
        """현재 시간 맥락을 고려한 응답 개선"""
        current_time = datetime.now()
        current_hour = current_time.hour
        current_weekday = current_time.weekday()  # 0=월요일
        korean_weekday = self._get_korean_weekday(current_weekday)
        
        # 시간별 추가 안내
        time_guidance = ""
        if current_hour < 9:
            time_guidance = f"\n\n⏰ 현재 시간({current_hour}시)을 고려하면 조식 시간입니다."
        elif current_hour < 14:
            time_guidance = f"\n\n⏰ 현재 시간({current_hour}시)을 고려하면 중식 시간입니다."
        else:
            time_guidance = f"\n\n⏰ 현재 시간({current_hour}시)을 고려하면 석식 시간입니다."
        
        # 오늘 날짜 관련 안내
        if current_weekday < 5:  # 평일
            time_guidance += f"\n📅 오늘은 {korean_weekday}입니다."
        else:  # 주말
            time_guidance += f"\n📅 오늘은 {korean_weekday}로 주말입니다. 구내식당 운영 여부를 확인해주세요."
        
        # 구내식당 기본 정보 추가 (응답에 없는 경우)
        if "055-254" not in base_response and "구내식당" not in base_response:
            time_guidance += f"\n\n📞 구내식당 문의: 055-254-2096 (총무담당)"
        
        return base_response + time_guidance
    
    def _is_menu_related_query(self, query: str) -> bool:
        """메뉴 관련 질문인지 판단"""
        menu_keywords = [
            '식단', '메뉴', '식사', '밥', '음식', '구내식당', 
            '조식', '중식', '석식', '아침', '점심', '저녁',
            '오늘', '내일', '이번주', '월요일', '화요일', '수요일', '목요일', '금요일'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in menu_keywords)
    
    def handle(self, request: QueryRequest) -> HandlerResponse:
        """
        menu 도메인 특화 처리
        기본 handle() 호출 후 시간 맥락 정보 자동 추가
        """
        # 기본 핸들러 로직 실행
        response = super().handle(request)
        
        # menu 도메인 특화: 시간 맥락 정보 보강
        if response.confidence >= self.confidence_threshold and self._is_menu_related_query(request.text):
            enhanced_answer = self._enhance_response_with_time_context(response.answer, request.text)
            response.answer = enhanced_answer
        
        return response
    
    def get_current_meal_recommendation(self) -> str:
        """현재 시간 기준 식사 추천 (유틸리티 메서드)"""
        current_time = datetime.now()
        current_hour = current_time.hour
        korean_weekday = self._get_korean_weekday(current_time.weekday())
        
        if current_hour < 9:
            return f"현재는 조식 시간입니다. 오늘({korean_weekday}) 조식 메뉴를 확인해보세요."
        elif current_hour < 14:
            return f"현재는 중식 시간입니다. 오늘({korean_weekday}) 점심 메뉴를 확인해보세요."
        else:
            return f"현재는 석식 시간입니다. 오늘({korean_weekday}) 저녁 메뉴를 확인해보세요."


# ================================================================
# 테스트 코드 (개발용)
# ================================================================

if __name__ == "__main__":
    """menu_handler 개발 테스트"""
    print("🍽️ Menu Handler 테스트 시작")
    
    test_queries = [
        "오늘 점심 메뉴가 뭐야?",
        "내일 아침 식단 알려줘",
        "이번주 월요일 저녁 메뉴는?",
        "구내식당 석식 시간이 언제야?",
        "금요일 식단표 보여줘"
    ]
    
    handler = menu_handler()
    
    # 현재 시간 기준 추천 테스트
    print(f"\n현재 시간 추천: {handler.get_current_meal_recommendation()}")
    
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
    
    print("\n✅ 메뉴 핸들러 테스트 완료")
