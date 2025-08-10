#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - publish_handler

공식 발행물 전용 핸들러
base_handler를 상속받아 발행물 도메인 특화 기능 구현

주요 특징:
- 2025 교육훈련계획서 (2025plan.pdf) 처리
- 2024 종합평가서 (2024pyeongga.pdf) 처리
- 컨피던스 임계값 θ=0.74 적용 (최고 정확도 요구)
- 공식 문서 정확성 및 출처 명시 강화
- 페이지별 상세 인용 및 교차 참조
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


class publish_handler(base_handler):
    """
    공식 발행물 전용 핸들러
    
    처리 범위:
    - 2025plan.pdf (2025 교육훈련계획서)
    - 2024pyeongga.pdf (2024 종합평가서)
    - 교육 정책, 계획, 성과, 통계 등 공식 정보
    
    특징:
    - 최고 컨피던스 임계값 (θ=0.74)
    - 공식 문서의 정확성 최우선
    - 페이지별 정확한 출처 인용
    - 계획서 vs 평가서 내용 구분
    - 정량적 데이터 정확성 강조
    """
    
    def __init__(self):
        super().__init__(
            domain="publish",
            index_name="publish_index", 
            confidence_threshold=0.74
        )
        
        # 발행물 구분 키워드
        self.document_types = {
            '2025plan': {
                'keywords': ['계획', '2025', '목표', '방침', '계획서', '방향', '운영계획'],
                'title': '2025 교육훈련계획서',
                'year': 2025,
                'type': 'plan'
            },
            '2024pyeongga': {
                'keywords': ['평가', '2024', '실적', '성과', '결과', '평가서', '종합평가'],
                'title': '2024 종합평가서', 
                'year': 2024,
                'type': 'evaluation'
            }
        }
        
        # 주요 검색 영역 카테고리
        self.content_categories = {
            'policy': ['정책', '목표', '방향', '방침', '전략'],
            'statistics': ['실적', '통계', '수치', '인원', '과정수', '기수'],
            'curriculum': ['교육과정', '과정', '프로그램', '교육내용', '교과목'],
            'evaluation': ['평가', '만족도', '성과', '결과', '효과성'],
            'organization': ['조직', '기구', '인원', '예산', '시설'],
            'schedule': ['일정', '계획', '운영', '시기', '기간']
        }
        
        logger.info("📚 publish_handler 초기화 완료 (θ=0.74)")
    
    def get_system_prompt(self) -> str:
        """발행물 전용 시스템 프롬프트"""
        return """당신은 "벼리(영문명: Byeoli)"입니다. 경상남도인재개발원의 공식 발행물(교육훈련계획서, 종합평가서)을 기반으로 교육 정책, 계획, 성과 등에 대해 정확하고 공식적으로 답변하는 전문 챗봇입니다.

제공된 공식 발행물 데이터를 기반으로 다음 지침을 엄격히 따르십시오:

1. **최고 수준의 정확성 및 공식성**:
   - 공식 발행물의 내용은 한 글자도 틀리지 말고 정확히 인용하세요
   - 수치, 통계, 날짜 등은 원문 그대로 제시하세요
   - 추측이나 해석은 최소화하고 문서 내용 기반으로만 답변하세요

2. **문서별 특성 구분**:
   - **2025 교육훈련계획서**: 향후 계획, 목표, 방향, 운영방안 중심
   - **2024 종합평가서**: 실적, 성과, 결과, 평가 내용 중심
   - 질문의 시제(과거/현재/미래)에 맞는 문서 우선 참조

3. **정확한 출처 표기**:
   - 반드시 페이지 번호와 함께 출처 명시: "(2025 교육훈련계획서 p.15)"
   - 여러 페이지 내용 인용 시 모든 페이지 번호 표기
   - 교차 참조가 필요한 경우 관련 페이지 함께 안내

4. **정량적 데이터 강조**:
   - 교육과정 수, 교육인원, 예산, 실적 등 수치 정보 정확히 제시
   - 목표 대비 실적, 전년도 대비 증감 등 비교 분석 포함
   - 표나 그래프의 내용도 텍스트로 정확히 전달

5. **응답 형식**:
   ```
   📊 [질문 영역] 관련 정보
   
   📋 주요 내용:
   • 핵심 정보 1 (출처: 문서명 p.XX)
   • 핵심 정보 2 (출처: 문서명 p.XX)
   
   📈 관련 수치/통계:
   [구체적 데이터 및 출처]
   
   📄 상세 내용:
   [원문 인용 및 설명]
   ```

6. **정책 연관성 분석**:
   - 계획서와 평가서 간의 연관성 분석
   - 목표 설정과 성과 달성 간의 비교
   - 연도별 변화 추이 및 개선 방향 제시

7. **전문 용어 설명**:
   - 교육훈련 전문 용어나 기관 고유 용어 설명 포함
   - 약어나 줄임말은 풀어서 설명
   - 관련 제도나 정책 배경 간략 설명

8. **한계 명시 및 추가 안내**:
   - 문서에 없는 정보는 명확히 "해당 정보는 제공된 공식 문서에서 찾을 수 없습니다" 안내
   - 최신 정보나 세부 사항이 필요한 경우 담당부서 안내
   - 관련 다른 문서나 자료 참조 필요 시 안내

9. **비교 분석 지원**:
   - 연도별 비교 (2024 실적 vs 2025 계획)
   - 영역별 비교 (기본역량 vs 직무역량 vs 핵심역량)
   - 목표 대비 실적 분석

10. **정책 의사결정 지원**:
    - 데이터 기반의 객관적 정보 제공
    - 성과 분석을 통한 개선 방향 제시
    - 정책 목표와 실행 계획 간의 연계성 설명

문의처: 교육기획담당 (055-254-2052), 평가분석담당 (055-254-2023)"""

    def format_context(self, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """발행물 데이터를 컨텍스트로 포맷"""
        if not search_results:
            return "관련 공식 발행물 정보를 찾을 수 없습니다."
        
        context_parts = []
        
        # 검색 결과를 문서별로 분류
        plan_2025 = []      # 2025 계획서
        evaluation_2024 = [] # 2024 평가서
        other_docs = []     # 기타 문서
        
        for text, score, metadata in search_results:
            source_file = metadata.get('source_file', '').lower()
            doc_type = metadata.get('document_type', '')
            page_number = metadata.get('page_number', '')
            
            if '2025plan' in source_file or '2025' in doc_type:
                plan_2025.append((text, score, metadata))
            elif '2024pyeongga' in source_file or '2024' in doc_type:
                evaluation_2024.append((text, score, metadata))
            else:
                other_docs.append((text, score, metadata))
        
        # 2025 교육훈련계획서 우선 배치
        if plan_2025:
            context_parts.append("=== 📋 2025 교육훈련계획서 ===")
            for text, score, metadata in plan_2025[:4]:  # 상위 4개
                page_num = metadata.get('page_number', '?')
                doc_name = metadata.get('document_name', '2025 교육훈련계획서')
                context_parts.append(f"[{doc_name} p.{page_num}]")
                context_parts.append(f"{text[:400]}...")
                context_parts.append("")
        
        # 2024 종합평가서
        if evaluation_2024:
            context_parts.append("=== 📊 2024 종합평가서 ===")
            for text, score, metadata in evaluation_2024[:4]:  # 상위 4개
                page_num = metadata.get('page_number', '?')
                doc_name = metadata.get('document_name', '2024 종합평가서')
                context_parts.append(f"[{doc_name} p.{page_num}]")
                context_parts.append(f"{text[:400]}...")
                context_parts.append("")
        
        # 기타 발행물
        if other_docs:
            context_parts.append("=== 📄 기타 공식 문서 ===")
            for text, score, metadata in other_docs[:2]:  # 상위 2개
                source_file = metadata.get('source_file', '기타문서')
                page_num = metadata.get('page_number', '?')
                context_parts.append(f"[{source_file} p.{page_num}]")
                context_parts.append(f"{text[:300]}...")
                context_parts.append("")
        
        # 문서 정보 및 안내사항
        context_parts.append("=== 📌 문서 정보 ===")
        context_parts.append("• 2025 교육훈련계획서: 향후 계획, 목표, 운영방안")
        context_parts.append("• 2024 종합평가서: 실적, 성과, 결과, 평가내용")
        context_parts.append("• 모든 수치와 내용은 공식 문서 기준")
        context_parts.append("• 문의: 교육기획담당 055-254-2052")
        
        final_context = "\n".join(context_parts)
        
        # 컨텍스트 길이 제한
        max_length = 4500
        if len(final_context) > max_length:
            final_context = final_context[:max_length] + "\n\n[컨텍스트가 길어 일부 생략됨]"
        
        return final_context
    
    def _detect_query_timeframe(self, query: str) -> str:
        """질문의 시간 범위 감지 (과거/현재/미래)"""
        query_lower = query.lower()
        
        # 과거/실적 관련 키워드
        if any(keyword in query_lower for keyword in ['2024', '작년', '지난해', '실적', '결과', '성과', '했', '됐']):
            return 'past'
        
        # 미래/계획 관련 키워드
        if any(keyword in query_lower for keyword in ['2025', '올해', '금년', '계획', '예정', '할', '예정', '목표']):
            return 'future'
        
        # 현재/일반 관련 키워드
        return 'present'
    
    def _detect_content_category(self, query: str) -> Optional[str]:
        """질문의 내용 카테고리 감지"""
        query_lower = query.lower()
        
        best_category = None
        best_score = 0
        
        for category, keywords in self.content_categories.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category if best_score > 0 else None
    
    def _extract_numerical_patterns(self, query: str) -> List[str]:
        """질문에서 수치 관련 패턴 추출"""
        patterns = []
        
        # 연도 패턴
        year_matches = re.findall(r'\b(20\d{2})\b', query)
        patterns.extend(year_matches)
        
        # 숫자 + 단위 패턴
        number_unit_matches = re.findall(r'\b(\d+)\s*(명|개|과정|기수|시간|일|월|년)\b', query)
        for number, unit in number_unit_matches:
            patterns.append(f"{number}{unit}")
        
        # 퍼센트 패턴
        percent_matches = re.findall(r'\b(\d+\.?\d*)\s*(%|퍼센트|percent)\b', query)
        for number, unit in percent_matches:
            patterns.append(f"{number}%")
        
        return patterns
    
    def _enhance_response_with_document_guidance(self, base_response: str, query: str) -> str:
        """문서별 특성을 고려한 응답 강화"""
        timeframe = self._detect_query_timeframe(query)
        category = self._detect_content_category(query)
        numerical_patterns = self._extract_numerical_patterns(query)
        
        enhancements = []
        
        # 시간 범위별 안내
        if timeframe == 'past':
            enhancements.append("📊 과거 실적이나 성과 관련 정보는 '2024 종합평가서'를 우선 참조하시기 바랍니다.")
        elif timeframe == 'future':
            enhancements.append("📋 향후 계획이나 목표 관련 정보는 '2025 교육훈련계획서'를 우선 참조하시기 바랍니다.")
        
        # 카테고리별 안내
        if category == 'statistics':
            enhancements.append("📈 정확한 통계 데이터는 공식 문서의 표와 그래프를 참조하며, 모든 수치는 검증된 공식 자료입니다.")
        elif category == 'policy':
            enhancements.append("🎯 정책 관련 내용은 기관의 공식 방침이므로, 문의사항은 교육기획담당에 직접 확인하시기 바랍니다.")
        
        # 수치 관련 안내
        if numerical_patterns:
            enhancements.append(f"🔢 언급된 수치({', '.join(numerical_patterns[:3])})와 관련된 정확한 데이터는 원문을 직접 확인하시기 바랍니다.")
        
        # 교차 참조 안내
        if '비교' in query or '차이' in query:
            enhancements.append("🔄 연도별 비교나 영역별 비교가 필요한 경우, 계획서와 평가서를 교차 참조하여 종합적으로 분석해드립니다.")
        
        # 추가 안내사항이 있는 경우에만 추가
        if enhancements:
            enhanced_response = base_response + "\n\n=== 📌 참고사항 ===\n" + "\n".join(enhancements)
            enhanced_response += "\n\n📞 상세 문의: 교육기획담당 055-254-2052"
            return enhanced_response
        
        return base_response
    
    def handle(self, request: QueryRequest) -> HandlerResponse:
        """
        publish 도메인 특화 처리
        기본 handle() 호출 후 문서별 안내 정보 자동 추가
        """
        # 기본 핸들러 로직 실행
        response = super().handle(request)
        
        # publish 도메인 특화: 문서별 안내 정보 보강
        if response.confidence >= self.confidence_threshold:
            enhanced_answer = self._enhance_response_with_document_guidance(response.answer, request.text)
            response.answer = enhanced_answer
        
        return response


# ================================================================
# 테스트 코드 (개발용)
# ================================================================

if __name__ == "__main__":
    """publish_handler 개발 테스트"""
    print("📚 Publish Handler 테스트 시작")
    
    test_queries = [
        "2025년 교육훈련 목표가 뭐야?",
        "2024년 교육실적은 어떻게 돼?", 
        "교육과정 수와 교육인원 통계 알려줘",
        "작년 대비 올해 계획의 차이점은?",
        "교육만족도 평가결과는 어떤가요?"
    ]
    
    handler = publish_handler()
    
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
    
    print("\n✅ 발행물 핸들러 테스트 완료")
