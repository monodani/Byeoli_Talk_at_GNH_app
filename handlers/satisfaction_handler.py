#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - satisfaction_handler

교육과정 및 교과목 만족도 조사 데이터 전용 핸들러
base_handler를 상속받아 만족도 도메인 특화 기능 구현

주요 특징:
- 기존 코랩 검증된 "벼리" 프롬프트 보존
- 교육과정/교과목 만족도 통합 처리
- 컨피던스 임계값 θ=0.68 적용
- 만족도 점수, 순위, 의견 등 정량/정성 정보 제공
"""

import logging
from typing import List, Dict, Any, Tuple

# 프로젝트 모듈
from handlers.base_handler import base_handler
from utils.contracts import QueryRequest, HandlerResponse
from utils.textifier import TextChunk

# 로깅 설정
logger = logging.getLogger(__name__)


class satisfaction_handler(base_handler):
    """
    만족도 조사 데이터 전용 핸들러
    
    처리 범위:
    - 교육과정 만족도 (course_satisfaction.csv)
    - 교과목 만족도 (subject_satisfaction.csv)
    - 통합 만족도 분석 및 순위 정보
    - 정량적 점수 + 정성적 의견 통합 제공
    """
    
    def __init__(self):
        super().__init__(
            domain="satisfaction",
            index_name="satisfaction_index", 
            confidence_threshold=0.68
        )
        
        logger.info("📊 satisfaction_handler 초기화 완료 (θ=0.68)")
    
    def get_system_prompt(self) -> str:
        """만족도 전용 시스템 프롬프트 (기존 코랩 검증 버전)"""
        return """당신은 "벼리(영문명: Byeoli)"입니다. 경상남도인재개발원의 교육과정 및 교과목 만족도 조사 데이터를 분석하여 사용자 질문에 정확하고 친절하게 답변하는 전문 챗봇입니다.

제공된 만족도 데이터를 기반으로 다음 지침을 엄격히 따르십시오:

1. **데이터 기반 답변**: 반드시 제공된 컨텍스트 내의 정보를 기반으로 답변해야 합니다. 추측하거나 없는 정보를 만들어내지 마세요.

2. **정확성**: 만족도 조사 결과(점수, 의견, 순위 등)는 정확하게 제시하세요.
   - 전반만족도, 역량향상도, 현업적용도, 교과편성 만족도, 강의만족도 구분
   - 교육과정별/교과목별 순위 정보 포함
   - 점수는 소수점 둘째 자리까지 정확히 표기

3. **정보 부족 시 대처**: 만약 제공된 데이터만으로는 질문에 답변할 수 없다면, 솔직하게 "해당 정보는 제가 가지고 있는 만족도 조사 데이터에서 찾을 수 없습니다."라고 답하고 추가적인 질문을 요청하세요.

4. **친절하고 간결한 어조**: 항상 친절하고 명확하며 간결하게 답변하세요.

5. **불필요한 서론/결론 제거**: 핵심 정보에 집중하여 군더더기 없는 답변을 제공하세요.

6. **정량적, 정성적 정보 혼합**: 답변에 점수와 같은 정량적 정보와, 교육생의 의견과 같은 정성적 정보를 함께 제시하여 풍부한 답변을 제공하세요.

7. **교육과정 vs 교과목 구분**: 
   - 교육과정 만족도: 전체 과정에 대한 종합적 평가
   - 교과목 만족도: 개별 강의/과목에 대한 평가
   명확히 구분하여 답변하세요.

8. **순위 정보 활용**: 해당 연도 전체 교육과정/교과목 중 몇 위인지 순위 정보를 포함하여 상대적 성과를 제시하세요."""

    def format_context(self, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """만족도 데이터를 컨텍스트로 포맷"""
        if not search_results:
            return "관련 만족도 데이터를 찾을 수 없습니다."
        
        context_parts = []
        
        for i, (text, score, metadata) in enumerate(search_results[:5], 1):
            # 메타데이터에서 추가 정보 추출
            source_info = ""
            if metadata.get('source_file'):
                source_info = f"[출처: {metadata['source_file']}]"
            
            if metadata.get('satisfaction_type'):
                source_info += f" [{metadata['satisfaction_type']}]"
                
            context_part = f"""=== 만족도 데이터 {i} ===
{source_info}
유사도 점수: {score:.3f}

{text}

"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)

    def _generate_prompt(self, query: str, retrieved_docs: List[Tuple[TextChunk, float]]) -> str:
        """
        만족도 도메인에 특화된 최종 프롬프트 생성
        """
        system_prompt = self.get_system_prompt()
        context = self.format_context([(doc.text, score, doc.metadata) for doc, score in retrieved_docs])
        
        prompt = f"""
        {system_prompt}

        ---
        참고 자료 (만족도 데이터):
        {context}
        ---

        사용자 질문:
        {query}

        답변:
        """
        return prompt
        
    
    def handle(self, request: QueryRequest) -> HandlerResponse:
        """
        만족도 질의 처리 (follow_up 완화 로직 포함)
        """
        # QueryRequest에서 필요한 정보 추출
        query = getattr(request, 'query', None) or getattr(request, 'text', '')
        follow_up = getattr(request, 'follow_up', False)
        
        # follow_up인 경우 컨피던스 임계값 완화
        original_threshold = self.confidence_threshold
        if follow_up:
            self.confidence_threshold = max(0.0, original_threshold - 0.02)
            logger.info(f"🔄 Follow-up 질의: 임계값 완화 {original_threshold} → {self.confidence_threshold}")
        
        try:
            # base_handler의 표준 처리 로직 사용
            response = super().handle(request)
            
            # 만족도 특화 후처리
            if response.confidence >= self.confidence_threshold:
                # 응답에 만족도 도메인 힌트 추가
                if "점" in response.answer and any(keyword in query for keyword in ["만족도", "점수", "평가"]):
                    # 만족도 점수가 포함된 답변인 경우 단위 표준화
                    response.answer = self._standardize_satisfaction_scores(response.answer)
                
                logger.info(f"✅ 만족도 답변 생성 완료 (confidence={response.confidence:.3f})")
            else:
                # 낮은 컨피던스인 경우 재질문 유도
                response.answer = self._generate_reask_response(query, response.confidence)
                logger.warning(f"⚠️ 낮은 컨피던스로 재질문 유도 (confidence={response.confidence:.3f})")
            
            return response
            
        finally:
            # 임계값 복원
            self.confidence_threshold = original_threshold

    
    def _standardize_satisfaction_scores(self, answer: str) -> str:
        """만족도 점수 표기 표준화"""
        import re
        
        # 점수 패턴 정규화 (예: "4.5점" → "4.50점")
        score_pattern = r'(\d+\.\d{1})점'
        standardized = re.sub(score_pattern, r'\g<1>0점', answer)
        
        return standardized
    
    def _generate_reask_response(self, query: str, confidence: float) -> str:
        """낮은 컨피던스 시 재질문 유도 응답"""
        reask_suggestions = []
        
        # 쿼리 분석해서 구체적인 재질문 제안
        if "만족도" in query:
            if "교육과정" not in query and "교과목" not in query:
                reask_suggestions.append("'교육과정 만족도' 또는 '교과목 만족도' 중 어떤 것을 원하시는지")
            
            if not any(year in query for year in ["2024", "2025"]):
                reask_suggestions.append("구체적인 연도(예: 2024년, 2025년)")
                
            if not any(keyword in query for keyword in ["과정명", "교과목명", "강의명"]):
                reask_suggestions.append("특정 교육과정명이나 교과목명")
        
        base_response = f"죄송합니다. 요청하신 만족도 정보를 정확히 찾기 어렵습니다. (신뢰도: {confidence:.2f})"
        
        if reask_suggestions:
            suggestion_text = ", ".join(reask_suggestions)
            base_response += f"\n\n더 정확한 답변을 위해 다음 정보를 추가해서 다시 질문해 주세요:\n- {suggestion_text}"
        else:
            base_response += "\n\n다른 방식으로 질문해 주시거나, 좀 더 구체적인 정보를 포함해서 다시 질문해 주세요."
        
        return base_response


# 편의 함수 (기존 API 호환성)
def handle_satisfaction_query(query: str, temperature: float = 0.1, k: int = 5) -> str:
    """
    기존 코랩 코드 호환을 위한 편의 함수
    
    Args:
        query: 사용자 질문
        temperature: LLM 온도 (사용되지 않음, 호환성을 위해 유지)
        k: 검색 문서 수 (사용되지 않음, 호환성을 위해 유지)
        
    Returns:
        응답 텍스트
    """
    from utils.contracts import QueryRequest
    import uuid
    
    handler = satisfaction_handler()
    request = QueryRequest(
        query=query,
        text=query,
        context=None,
        follow_up=False,
        trace_id=str(uuid.uuid4())
    )
    
    response = handler.handle(request)
    return response.answer


# 테스트용 메인 함수
if __name__ == "__main__":
    # 기본 테스트
    test_queries = [
        "중견리더과정의 만족도와 교육생 의견에 대해 알려줘.",
        "2024년 교육과정 중 만족도가 가장 높은 과정은?",
        "신임공무원 교육과정의 역량향상도 점수는?",
        "교과목 만족도 상위 5개 강의는?"
    ]
    
    handler = satisfaction_handler()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n=== 테스트 {i}: {query} ===")
        
        try:
            from utils.contracts import QueryRequest
            import uuid
            
            request = QueryRequest(
                query=query,
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
    
    print("\n✅ 만족도 핸들러 테스트 완료")
