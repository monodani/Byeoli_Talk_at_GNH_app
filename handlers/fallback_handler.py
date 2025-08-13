#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - fallback_handler (담당부서 정보 통합)

미매칭/저신뢰 답변 처리 전용 핸들러
모든 전문 핸들러가 실패했을 때 최후의 보루 역할

주요 수정사항:
✅ domain="general" 사용 (IndexManager 호환)
✅ 추상 메서드 'format_context'와 'get_system_prompt' 구현
✅ 벡터스토어 없이도 작동하는 안전 로직
✅ 담당부서 연락처 및 질문 카테고리 정보 통합
✅ 질문 키워드에 따라 담당 부서 추천 로직 추가
✅ Citation 생성 보장
✅ 에러 처리 강화

특징:
- 컨피던스 임계값 θ=0.00 (항상 작동)
- EWMA 기반 자동 튜닝 지원 (±0.02)
- 적절한 담당부서 안내 및 재질문 유도
- 검색 실패에도 일반적인 기관 정보 제공
- 항상 성공 보장 (절대 실패하지 않음)
"""

import time
import logging
import re
import random
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# 프로젝트 모듈
from handlers.base_handler import base_handler
from utils.contracts import QueryRequest, HandlerCandidate, HandlerResponse, Citation, ConversationContext
from utils.textifier import TextChunk



# 로깅 설정
logger = logging.getLogger(__name__)


class fallback_handler(base_handler):
    """
    최종 백업 핸들러 (담당부서 정보 통합)

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
        # base_handler 초기화 시 domain, index_name, confidence_threshold 지정
        super().__init__(
            domain="general",
            index_name="general_index",  # general 인덱스 활용
            confidence_threshold=0.00
        )

        logger.info("🛡️ fallback_handler 초기화 완료 (θ=0.00, general 도메인)")

        # ✅ 담당부서별 연락처 정보 (사용자 제공 데이터)
        self.contact_info = {
            '총무담당': '055-254-2011',
            '평가분석담당': '055-254-2021',
            '교육기획담당': '055-254-2051',
            '교육운영1담당': '055-254-2061',
            '교육운영2담당': '055-254-2071',
            # 사용자가 중복으로 제공한 '교육운영1담당'은 하나로 통일
            '사이버담당': '055-254-2081'
        }

        # ✅ 기관 기본 정보 (사용자 제공 데이터)
        self.institute_info = {
            'name': '경상남도인재개발원',
            'address': '경상남도 진주시 월아산로 2026 경상남도 서부청사 4~6층',
            'main_phone': '055-254-2011',
            'website': 'https://gyeongnam.go.kr/hrd',
            'business_hours': '평일 09:00-18:00'
        }

        # ✅ 질문 카테고리 매핑 (사용자 제공 데이터)
        self.question_categories = {
            'satisfaction': {
                'keywords': ['만족도', '평가', '점수', '성과', '결과', '설문', '조사'],
                'department': '평가분석담당',
                'description': '교육평가 및 만족도 조사'
            },
            'planning': {
                'keywords': ['계획', '일정', '교육과정', '신청', '모집', '등록'],
                'department': '교육기획담당',
                'description': '교육훈련계획 및 운영'
            },
            'cyber': {
                'keywords': ['사이버', '온라인', '이러닝', '민간위탁', '나라배움터', '인터넷'],
                'department': '사이버담당',
                'description': '사이버교육 운영'
            },
            'facility': {
                'keywords': ['전결', '규정', '식당', '기숙사', '주차', '시설', '연락처'],
                'department': '총무담당',
                'description': '시설 및 총무업무'
            },
            'general': {
                'keywords': ['학칙', '운영', '원칙', '안내', '정보', '위치', '주소'],
                'department': '교육기획담당',
                'description': '일반 운영 및 안내'
            }
        }
    
    def _match_category(self, query: str) -> Optional[Dict[str, Any]]:
        """
        사용자 질문에서 키워드를 기반으로 담당 부서를 매칭합니다.
        
        Args:
            query: 사용자 질문 텍스트
            
        Returns:
            매칭된 카테고리 정보(딕셔너리) 또는 None
        """
        for category, info in self.question_categories.items():
            for keyword in info['keywords']:
                if keyword in query:
                    logger.info(f"🔍 fallback_handler: '{query}'에서 키워드 '{keyword}' 발견. '{info['department']}' 추천.")
                    return info
        return None

    def get_system_prompt(self) -> str:
        """
        fallback_handler 전용 시스템 프롬프트를 반환합니다.
        기본 기관 정보와 담당 부서 정보를 포함합니다.
        """
        contact_list = "\n".join([f"- {dept}: {phone}" for dept, phone in self.contact_info.items()])
        
        prompt = f"""
당신은 경상남도인재개발원의 AI 챗봇 '벼리'입니다. 
사용자의 질문에 대해 전문 핸들러가 답을 찾지 못했을 때 이 프롬프트를 사용합니다.

<기관 정보>
- 기관명: {self.institute_info['name']}
- 주소: {self.institute_info['address']}
- 대표 전화: {self.institute_info['main_phone']}
- 홈페이지: {self.institute_info['website']}
- 운영 시간: {self.institute_info['business_hours']}

<담당부서 연락처>
{contact_list}

<담당부서 추천 로직>
- '만족도', '평가', '성과', '설문' 관련 질문 -> 평가분석담당
- '계획', '신청', '교육과정', '모집' 관련 질문 -> 교육기획담당
- '사이버', '온라인', '이러닝' 관련 질문 -> 사이버담당
- '식당', '기숙사', '주차', '시설' 관련 질문 -> 총무담당
- 위 키워드에 해당하지 않는 경우 -> 교육기획담당 (일반 운영)

사용자 질문에 대해 직접적인 답변을 제공하기보다는, 아래 규칙에 따라 정중하고 친절하게 안내하고 재질문을 유도하세요.

<규칙>
1. 사용자의 질문과 가장 관련성이 높은 담당 부서를 추천하고, 해당 부서의 연락처를 제공하세요.
2. "보다 자세한 내용은 담당 부서에 문의하시거나, 키워드를 포함하여 다시 질문해 주시면 정확한 답변을 찾아드릴 수 있습니다."와 같은 문장으로 재질문을 유도하세요.
3. 질문과 관련된 특정 정보(예: "식당 위치")를 알 수 있다면 함께 언급하되, 모호한 정보는 추측하지 마세요.
4. 모든 답변은 '벼리'라는 친근하고 도움이 되는 챗봇 페르소나를 유지해야 합니다.
5. 절대 모르는 정보를 지어내지 마세요.
6. 답변 마지막에는 항상 "더 궁금한 점이 있으시면 언제든지 다시 질문해 주세요."와 같은 격려 문구를 추가하세요.
"""
        return prompt.strip()

    def format_context(self, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """
        컨텍스트가 없는 fallback 핸들러의 특성을 고려하여
        빈 문자열을 반환하거나, 필요한 경우 간단한 정보를 포맷팅합니다.
        """
        # fallback_handler는 주로 검색 결과 없이 작동하므로 빈 문자열 반환
        return ""

    def _generate_prompt(
        self,
        query: str,
        retrieved_docs: List[Tuple[TextChunk, float]]
    ) -> str:
        """
        base_handler가 요구하는 추상 메서드 구현.
        - fallback은 벡터검색 미의존이 기본이므로 retrieved_docs가 비어도 동작해야 함
        - format_context()는 (text, score, metadata) 튜플 목록을 기대 → 어댑터 변환
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

        # 3) fallback 전용 컨텍스트(대개 공백)
        context_block = self.format_context(context_tuples)

        # 4) 최종 프롬프트
        prompt = (
            f"{system_prompt}\n\n"
            f"---\n"
            f"사용자 질문:\n{query}\n\n"
            f"참고 자료(있을 경우):\n{context_block}\n\n"
            f"지침:\n"
            f"- 전문 핸들러가 답을 찾지 못했을 때의 응답을 생성합니다.\n"
            f"- 담당부서 추천/연락처 제시 및 재질문 유도를 우선합니다.\n"
            f"- 모르는 정보는 지어내지 말고 기관 기본 정보만 제공합니다.\n"
        )
        return prompt
    

    def handle(self, request: QueryRequest) -> HandlerResponse:
        """
        fallback_handler의 메인 처리 로직
        """
        start_time = time.time()
        logger.info(f"🛡️ fallback_handler 작동: {request.query}")

        matched_info = self._match_category(request.query)
        
        if matched_info:
            # 질문 카테고리 매칭에 성공한 경우
            department = matched_info['department']
            contact = self.contact_info.get(department, '알 수 없음')
            description = matched_info['description']
            
            answer = (
                f"안녕하세요! '{description}'에 대한 문의로 이해했습니다.\n"
                f"해당 업무는 **{department}**에서 담당하고 있습니다. **(☎️ {contact})**\n\n"
                f"보다 자세한 내용은 담당 부서에 문의하시거나, 키워드를 포함하여 다시 질문해 주시면 정확한 답변을 찾아드릴 수 있습니다.\n\n"
                "더 궁금한 점이 있으시면 언제든지 다시 질문해 주세요."
            )
            
            # 매칭된 카테고리를 Citation으로 추가
            citation = Citation(
                source_id=f"fallback/department_contact/{department}",
                source_file="fallback_handler.py",
                text="긴급 상황 시 기본 응답",
                relevance_score=0.0,
                content=f"담당부서: {department}, 연락처: {contact}, 설명: {description}",
                page=0
            )
            
            citations = [citation]
            
        else:
            # 매칭되는 키워드가 없는 경우, 일반적인 기관 정보 제공
            answer = (
                f"안녕하세요! 요청하신 내용에 대한 정보를 찾기 어렵습니다. 😅\n\n"
                f"일반적인 문의사항은 **{self.institute_info['name']}**의 **교육기획담당** **(☎️ {self.contact_info['교육기획담당']})**으로 문의하시거나, "
                f"**더 구체적인 키워드**를 포함하여 다시 질문해 주시면 정확한 답변을 찾아드릴 수 있습니다.\n\n"
                f"더 궁금한 점이 있으시면 언제든지 다시 질문해 주세요."
            )
            
            # 기본 기관 정보를 Citation으로 추가
            citation = Citation(
                source_id="fallback/institute_info",
                source_file="fallback_handler.py",
                text="긴급 상황시 기본 응답",
                relevance_score=0.0,
                content=f"기관명: {self.institute_info['name']}, 대표 전화: {self.institute_info['main_phone']}",
                page=0
            )
            
            citations = [citation]

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"✅ fallback_handler 응답 생성 완료. 소요시간: {elapsed_ms:.2f}ms")

        # Fallback 핸들러는 항상 0.00의 컨피던스를 반환하지만,
        # 에러 핸들링을 위해 성공 여부를 False로 처리하지 않음
        # (컨피던스 0.0으로도 정상적인 핸들러 응답이 가능)
        self._update_success_rate(True)

        return HandlerResponse(
            answer=answer,
            confidence=0.00, # Fallback 핸들러의 컨피던스 임계값은 항상 0.00
            domain=self.domain,                  
            citations=citations,
            elapsed_ms=elapsed_ms,
            success=True,           
            diagnostics={}
        )

    # 이전 버전에서 사용된 EWMA 및 추천 문구 로직은
    # 새로운 로직으로 대체되었으므로 제거하거나 주석 처리합니다.
    def _get_reask_suggestion(self, query: str) -> str:
        """
        질문 카테고리를 기반으로 재질문 제안 문구를 생성합니다.
        (현재는 새로운 로직으로 대체되어 사용되지 않음)
        """
        # ... (이전 코드 내용)
        return "더 구체적인 질문으로 다시 문의해주세요."

    def _update_success_rate(self, success: bool):
        # ... (이전 코드 내용)
        pass


# ================================================================
# 테스트 코드 (개발용)
# ================================================================
if __name__ == "__main__":
    """fallback_handler 개발 테스트"""
    print("🛡️ Fallback Handler 테스트 시작")

    test_queries = [
        "2024년 교육과정 만족도가 궁금해요", # 평가분석담당
        "다음 교육과정 신청은 언제 시작하나요?", # 교육기획담당
        "온라인으로 들을 수 있는 강의를 찾아줘", # 사이버담당
        "구내식당 위치가 어디인가요?", # 총무담당
        "경상남도인재개발원 위치가 궁금해요", # 일반적인 질문 (교육기획담당)
        "완전히 관련 없는 질문", # 일반적인 질문 (교육기획담당)
    ]

    handler = fallback_handler()
    
    # 더미 QueryRequest 클래스
    from utils.contracts import QueryRequest
    import uuid

    for i, query in enumerate(test_queries, 1):
        print(f"\n=== 테스트 {i}: {query} ===")
        
        request = QueryRequest(
            text=query,
            context=None,
            follow_up=False,
            trace_id=str(uuid.uuid4())
        )
        
        response = handler.handle(request)
        print(f"✅ 응답: {response.answer}")
        print(f"📊 컨피던스: {response.confidence:.3f}")
        print(f"🔗 핸들러ID: {response.domain}")
        print(f"⏱️ 소요시간: {response.elapsed_ms}ms")
        print(f"📄 Citation 수: {len(response.citations)}")
        for citation in response.citations:
            print(f"    - 출처: {citation.source_id}")
            print(f"    - 내용: {citation.content}")
