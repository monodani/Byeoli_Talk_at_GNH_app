#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - cyber_handler

사이버교육 일정 전용 핸들러
base_handler를 상속받아 사이버교육 도메인 특화 기능 구현

주요 특징:
- 민간위탁 사이버교육 (mingan.csv) 처리
- 나라배움터 사이버교육 (nara.csv) 처리
- 컨피던스 임계값 θ=0.66 적용
- 교육과정명, 분류, 학습시간, 평가유무 등 상세 정보 제공
- 기존 코랩 템플릿 보존 및 활용
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

# 프로젝트 모듈
from handlers.base_handler import BaseHandler
from utils.contracts import QueryRequest, HandlerResponse
from utils.textifier import TextChunk

# 로깅 설정
logger = logging.getLogger(__name__)


class cyber_handler(BaseHandler):
    """
    사이버교육 일정 전용 핸들러
    
    처리 범위:
    - mingan.csv (민간위탁 사이버교육)
    - nara.csv (나라배움터 사이버교육)
    - 교육과정 검색, 분류별 필터링, 학습시간 안내
    
    특징:
    - 중간 컨피던스 임계값 (θ=0.66)
    - 교육 유형별 구분 (민간위탁 vs 나라배움터)
    - 분류 체계 및 학습 정보 상세 제공
    - 평가 필요성 및 인정시간 안내
    """
    
    def __init__(self):
        super().__init__(
            domain="cyber",
            index_name="cyber_index", 
            confidence_threshold=0.66
        )
        
        # 교육 분류 키워드 매핑
        self.education_categories = {
            # 나라배움터 분류
            '직무': ['직무', '업무', '실무', '법률', '제도', '시스템'],
            '소양': ['소양', '교양', '인문', '문화', '예술', '건강', '취미'],
            '시책': ['시책', '정책', '제도', '법령', '규정', '청렴', '인권'],
            '디지털': ['디지털', 'IT', '컴퓨터', '데이터', '온라인', '사이버'],
            'Gov-MOOC': ['gov-mooc', 'mooc', '무크', '온라인강의'],
            
            # 민간위탁 일반 분류
            '경영': ['경영', '관리', '리더십', '조직', '전략'],
            '기술': ['기술', '공학', '과학', '연구', '개발'],
            '외국어': ['영어', '중국어', '일본어', '외국어', '언어']
        }
        
        # 교육 플랫폼 키워드
        self.platform_keywords = {
            '민간': ['민간', '민간위탁', '외부', '위탁', 'mingan'],
            '나라': ['나라', '나라배움터', '정부', '공공', 'nara', '국가'],
        }
        
        logger.info("💻 cyber_handler 초기화 완료 (θ=0.66)")
    
    def get_system_prompt(self) -> str:
        """사이버교육 전용 시스템 프롬프트"""
        return """당신은 "벼리(영문명: Byeoli)"입니다. 경상남도인재개발원의 사이버교육 과정 정보를 바탕으로 직원들의 온라인 교육 관련 질문에 정확하고 체계적으로 답변하는 전문 챗봇입니다.

제공된 사이버교육 데이터를 기반으로 다음 지침을 엄격히 따르십시오:

1. **교육 플랫폼 구분**: 두 가지 사이버교육 유형을 명확히 구분하여 안내하세요.
   - **민간위탁 사이버교육**: 외부 기관에서 개발한 전문 교육과정
   - **나라배움터**: 정부에서 운영하는 공공 온라인 교육플랫폼

2. **정확한 교육 정보 제공**:
   - 교육과정명: 정확한 과정명 제시
   - 분류체계: 직무/소양/시책/디지털/Gov-MOOC 등 명확한 분류
   - 학습시간: 총 학습시간 및 인정시간 구분
   - 평가여부: 수료를 위한 평가 필요성 안내

3. **민간위탁 교육 상세 정보**:
   - 개발연도/월: 콘텐츠 제작 시기
   - 세부 분류: 구분 > 대분류 > 중분류 > 소분류 > 세분류 체계
   - 학습시간 vs 인정시간 차이점 설명

4. **나라배움터 교육 상세 정보**:
   - 학습차시: 총 차시 수 및 예상 소요시간
   - 평가유무: "있습니다" 또는 "없습니다"로 명확히 표기
   - Gov-MOOC 특별과정 구분

5. **검색 및 추천 기능**:
   - 분류별 교육과정 목록 제공
   - 학습시간별 교육과정 추천
   - 평가 없는 과정 별도 안내

6. **응답 형식**:
   ```
   💻 [교육과정명]
   
   📚 교육 분류: [분류체계]
   ⏱️ 학습시간: [시간] / 인정시간: [시간]
   📝 평가: [있음/없음]
   🏢 플랫폼: [민간위탁/나라배움터]
   
   📖 과정 설명:
   [상세 설명]
   ```

7. **교육 신청 안내**: 
   - 나라배움터: 개별 계정 생성 후 신청
   - 민간위탁: 교육담당부서를 통한 신청
   - 문의처: 교육기획담당 (055-254-2052)

8. **분류별 특화 안내**:
   - **직무교육**: 업무와 직접 관련된 전문교육
   - **소양교육**: 교양 및 개인역량 개발교육  
   - **시책교육**: 정부정책 및 제도 이해교육
   - **디지털교육**: IT 및 디지털 역량 강화교육

9. **학습 계획 지원**: 요청 시 분류별, 시간별 맞춤 교육과정 조합 추천

10. **최신성 안내**: 2025년 기준 교육과정 정보이며, 변경사항은 교육담당부서에 확인 요청"""

    def format_context(self, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """사이버교육 데이터를 컨텍스트로 포맷"""
        if not search_results:
            return "관련 사이버교육 정보를 찾을 수 없습니다."
        
        context_parts = []
        
        # 검색 결과를 플랫폼별로 분류
        mingan_courses = []  # 민간위탁
        nara_courses = []    # 나라배움터
        
        for text, score, metadata in search_results:
            template_type = metadata.get('template_type', '')
            
            if template_type == 'mingan':
                mingan_courses.append((text, score, metadata))
            elif template_type == 'nara':
                nara_courses.append((text, score, metadata))
            else:
                # 기타 사이버교육 관련 정보
                context_parts.append(f"[사이버교육] {text}")
        
        # 민간위탁 교육 우선 배치
        if mingan_courses:
            context_parts.append("=== 민간위탁 사이버교육 ===")
            for text, score, metadata in mingan_courses[:4]:  # 상위 4개
                course_name = metadata.get('education_course', '')
                category_path = metadata.get('category_path', '')
                learning_hours = metadata.get('learning_hours', '')
                recognition_hours = metadata.get('recognition_hours', '')
                
                context_parts.append(f"[민간위탁] {course_name}")
                context_parts.append(f"분류: {category_path}")
                context_parts.append(f"시간: {learning_hours}h → 인정: {recognition_hours}h")
                context_parts.append(f"내용: {text[:200]}...")
                context_parts.append("")
        
        # 나라배움터 교육
        if nara_courses:
            context_parts.append("=== 나라배움터 사이버교육 ===")
            for text, score, metadata in nara_courses[:4]:  # 상위 4개
                course_name = metadata.get('education_course', '')
                category = metadata.get('category', '')
                learning_sessions = metadata.get('learning_sessions', '')
                recognition_hours = metadata.get('recognition_hours', '')
                evaluation_required = metadata.get('evaluation_required', '')
                
                context_parts.append(f"[나라배움터] {course_name}")
                context_parts.append(f"분류: {category}")
                context_parts.append(f"차시: {learning_sessions} / 인정: {recognition_hours}h")
                context_parts.append(f"평가: {evaluation_required}")
                context_parts.append(f"내용: {text[:200]}...")
                context_parts.append("")
        
        # 교육 신청 안내 추가
        context_parts.append("=== 교육 신청 안내 ===")
        context_parts.append("민간위탁: 교육기획담당 (055-254-2052) 문의")
        context_parts.append("나라배움터: 개별 계정 생성 후 직접 신청")
        context_parts.append("문의처: 교육기획담당 055-254-2052")
        
        final_context = "\n".join(context_parts)
        
        # 컨텍스트 길이 제한
        max_length = 4000
        if len(final_context) > max_length:
            final_context = final_context[:max_length] + "\n\n[컨텍스트가 길어 일부 생략됨]"
        
        return final_context
    
    def _detect_platform_preference(self, query: str) -> Optional[str]:
        """질문에서 선호하는 교육 플랫폼 감지"""
        query_lower = query.lower()
        
        for platform, keywords in self.platform_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return platform
        
        return None
    
    def _detect_category_preference(self, query: str) -> Optional[str]:
        """질문에서 선호하는 교육 분류 감지"""
        query_lower = query.lower()
        
        for category, keywords in self.education_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return None
    
    def _extract_learning_time_preference(self, query: str) -> Optional[Tuple[int, int]]:
        """질문에서 선호하는 학습시간 범위 추출"""
        # "3시간 이하", "5-10시간", "짧은", "긴" 등의 패턴 감지
        time_patterns = [
            r'(\d+)시간?\s*이하',
            r'(\d+)-(\d+)시간?',
            r'(\d+)시간?\s*미만',
            r'(\d+)시간?\s*정도'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, query)
            if match:
                if len(match.groups()) == 1:
                    # "N시간 이하" 형태
                    max_hours = int(match.group(1))
                    return (0, max_hours)
                elif len(match.groups()) == 2:
                    # "N-M시간" 형태
                    min_hours = int(match.group(1))
                    max_hours = int(match.group(2))
                    return (min_hours, max_hours)
        
        # 키워드 기반 감지
        if any(keyword in query.lower() for keyword in ['짧은', '간단한', '빠른']):
            return (0, 5)  # 5시간 이하
        elif any(keyword in query.lower() for keyword in ['긴', '상세한', '심화']):
            return (10, 100)  # 10시간 이상
        
        return None
    
    def _enhance_response_with_recommendations(self, base_response: str, query: str) -> str:
        """사용자 선호도 기반 추가 추천 정보 제공"""
        recommendations = []
        
        # 플랫폼 선호도 기반 안내
        platform_pref = self._detect_platform_preference(query)
        if platform_pref == '민간':
            recommendations.append("💡 민간위탁 교육은 전문성이 높고 체계적인 분류체계를 제공합니다.")
        elif platform_pref == '나라':
            recommendations.append("💡 나라배움터는 무료이며 정부 정책과 연계된 최신 교육을 제공합니다.")
        
        # 분류 선호도 기반 안내
        category_pref = self._detect_category_preference(query)
        if category_pref:
            recommendations.append(f"📚 {category_pref} 분야 교육을 원하시는군요. 관련 과정들을 우선 확인해보세요.")
        
        # 학습시간 선호도 기반 안내
        time_pref = self._extract_learning_time_preference(query)
        if time_pref:
            min_h, max_h = time_pref
            recommendations.append(f"⏰ {min_h}-{max_h}시간 범위의 교육과정을 찾으시는군요.")
        
        # 평가 부담 고려 안내
        if any(keyword in query.lower() for keyword in ['평가', '시험', '부담', '쉬운']):
            recommendations.append("📝 평가가 부담스러우시다면 '평가: 없습니다' 과정을 우선 검토해보세요.")
        
        # 추천 정보가 있는 경우에만 추가
        if recommendations:
            enhanced_response = base_response + "\n\n=== 맞춤 안내 ===\n" + "\n".join(recommendations)
            enhanced_response += "\n\n📞 상세 문의: 교육기획담당 055-254-2052"
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

        # 3) cyber 전용 컨텍스트 문자열 생성
        context_block = self.format_context(context_tuples)

        # 4) 최종 프롬프트 결합 (최소 형태)
        prompt = (
            f"{system_prompt}\n\n"
            f"---\n"
            f"사용자 질문:\n{query}\n\n"
            f"참고 자료(사이버교육):\n{context_block}\n\n"
            f"지침:\n"
            f"- 제공된 참고 자료 내 정보만 사용하세요.\n"
            f"- 없는 정보는 '데이터에 없음'이라고 답하세요.\n"
            f"- 플랫폼(민간/나라), 분류, 학습시간, 평가유무를 명확히 표기하세요.\n"
        )
        return prompt
    
    def handle(self, request: QueryRequest) -> HandlerResponse:
        """
        cyber 도메인 특화 처리
        기본 handle() 호출 후 맞춤 추천 정보 자동 추가
        """
        # 기본 핸들러 로직 실행
        response = super().handle(request)
        
        # QueryRequest에서 쿼리 텍스트 추출
        query = getattr(request, 'query', None) or getattr(request, 'text', '')
        
        # cyber 도메인 특화: 맞춤 추천 정보 보강
        if response.confidence >= self.confidence_threshold:
            enhanced_answer = self._enhance_response_with_recommendations(response.answer, query)
            response.answer = enhanced_answer
        
        return response



# ================================================================
# 테스트 코드 (개발용)
# ================================================================

if __name__ == "__main__":
    """cyber_handler 개발 테스트"""
    print("💻 Cyber Handler 테스트 시작")
    
    test_queries = [
        "나라배움터에서 들을 수 있는 직무교육 추천해줘",
        "민간위탁 사이버교육 중 5시간 이하 과정 찾아줘",
        "평가 없는 소양교육 과정이 있나?",
        "디지털 역량 관련 온라인 교육 알려줘",
        "Gov-MOOC 과정은 어떤 게 있어?"
    ]
    
    handler = cyber_handler()
    
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
    
    print("\n✅ 사이버교육 핸들러 테스트 완료")
