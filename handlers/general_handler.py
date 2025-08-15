#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - general_handler

일반 도메인 전용 핸들러: 학칙, 전결규정, 운영원칙, 업무담당자 연락처 통합 처리
base_handler를 상속받아 general 도메인 특화 기능 구현

주요 특징:
- 최고 정확도 요구 (θ=0.70)
- 규정 조항의 정확한 인용 필수
- 담당자 연락처 통합 제공
- 학칙, 전결규정, 운영원칙 종합 검색
- 법규/정책 해석 및 담당부서 안내
"""

import logging
from typing import List, Dict, Any, Tuple
from utils.textifier import TextChunk

# 프로젝트 모듈
from handlers.base_handler import base_handler
from utils.contracts import QueryRequest, HandlerResponse


# 로깅 설정
logger = logging.getLogger(__name__)


class general_handler(base_handler):
    """
    일반 도메인 전용 핸들러
    
    처리 범위:
    - hakchik.pdf (학칙+전결규정+운영원칙 통합문서)
    - task_telephone.csv (업무담당자 연락처)
    - operation_test.pdf (운영/평가 계획)
    
    특징:
    - 최고 컨피던스 임계값 (θ=0.70)
    - 정확한 규정 조항 인용
    - 담당자 연락처 자동 제공
    - 법규/정책 해석 및 안내
    """
    
    def __init__(self):
        super().__init__(
            domain="general",
            index_name="general_index", 
            confidence_threshold=0.70
        )
        
        logger.info("📋 general_handler 초기화 완료 (θ=0.70)")
    
    def get_system_prompt(self) -> str:
        """general 전용 시스템 프롬프트"""
        return """당신은 "벼리(영문명: Byeoli)"입니다. 경상남도인재개발원의 학칙, 전결규정, 운영원칙 및 업무담당자 연락처 정보를 기반으로 정확하고 체계적인 답변을 제공하는 전문 챗봇입니다.

제공된 일반 도메인 데이터를 기반으로 다음 지침을 엄격히 따르십시오:

1. **최고 수준의 정확성**: 학칙, 전결규정, 운영원칙 등 공식 문서의 내용은 반드시 원문 그대로 정확하게 인용해야 합니다. 해석이나 추측을 하지 마세요.

2. **구체적인 조항 인용**: 규정 관련 질문에는 해당하는 구체적인 조항 번호, 제목, 내용을 명시하여 답변하세요.
   - 학칙 제○조 (조항명)
   - 전결규정 제○장 제○조
   - 운영원칙 제○항
   
3. **담당자 연락처 적극 제공**: 업무 관련 질문 시 해당 담당부서와 연락처를 함께 안내하세요.
   - 담당부서: ○○○
   - 담당자: ○○○ ○○○
   - 연락처: 055-254-○○○○

4. **단계별 업무 안내**: 절차가 있는 업무는 단계별로 명확하게 설명하세요.

5. **법규 해석의 한계 명시**: 복잡한 법규 해석이 필요한 경우, 기본 정보를 제공한 후 담당부서 문의를 권하세요.

6. **최신성 고려**: 규정 변경 가능성을 언급하고, 최종 확인을 위해 담당부서 문의를 권하세요.

7. **체계적 구조화**: 답변을 다음과 같이 구조화하세요:
   - 핵심 답변
   - 관련 규정 조항
   - 담당부서 및 연락처
   - 추가 안내사항

8. **정보 부족 시 대처**: 제공된 데이터로 완전한 답변이 어려운 경우, 알 수 있는 범위까지 답변하고 담당부서 문의를 안내하세요.

9. **업무 연관성 파악**: 질문과 관련된 다른 규정이나 절차가 있다면 함께 안내하세요.

10. **친절하고 공식적인 어조**: 공무원의 업무를 돕는 전문적이면서도 친근한 어조를 유지하세요."""

    def format_context(self, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """general 데이터를 컨텍스트로 포맷"""
        if not search_results:
            return "관련 일반 도메인 정보를 찾을 수 없습니다."
        
        context_parts = []
        
        # 검색 결과를 문서 타입별로 분류
        regulations = []  # 학칙, 전결규정 등
        contacts = []     # 연락처 정보
        operations = []   # 운영계획 등
        
        for text, score, metadata in search_results:
            doc_type = metadata.get('doc_type', '일반문서')
            category = metadata.get('category', 'general')
            source_file = metadata.get('source_file', 'unknown')
            
            if category == 'contact' or 'telephone' in source_file:
                contacts.append((text, score, metadata))
            elif category == 'regulations' or 'hakchik' in source_file:
                regulations.append((text, score, metadata))
            elif category == 'operations' or 'operation' in source_file:
                operations.append((text, score, metadata))
            else:
                # 기타 일반 문서
                context_parts.append(f"[{doc_type}] {text}")
        
        # 규정 문서 우선 배치
        if regulations:
            context_parts.append("=== 학칙 및 규정 정보 ===")
            for text, score, metadata in regulations[:3]:  # 상위 3개
                doc_type = metadata.get('doc_type', '규정문서')
                page_num = metadata.get('page_number', '')
                page_info = f" (p.{page_num})" if page_num else ""
                context_parts.append(f"[{doc_type}{page_info}] {text}")
        
        # 담당자 연락처 정보
        if contacts:
            context_parts.append("\n=== 담당자 연락처 정보 ===")
            for text, score, metadata in contacts[:3]:  # 상위 3개
                context_parts.append(f"[연락처] {text}")
        
        # 운영계획 등 기타 정보
        if operations:
            context_parts.append("\n=== 운영 및 계획 정보 ===")
            for text, score, metadata in operations[:2]:  # 상위 2개
                doc_type = metadata.get('doc_type', '운영문서')
                context_parts.append(f"[{doc_type}] {text}")
        
        # 기타 일반 문서들은 이미 위에서 추가됨
        
        final_context = "\n\n".join(context_parts)
        
        # 컨텍스트 길이 제한 (너무 긴 경우 후반부 축약)
        max_length = 4000
        if len(final_context) > max_length:
            final_context = final_context[:max_length] + "\n\n[컨텍스트가 길어 일부 생략됨]"
        
        return final_context

    def _generate_prompt(self, query: str, retrieved_docs: List[Tuple[TextChunk, float]]) -> str:
        """
        일반 도메인에 특화된 최종 프롬프트 생성
        """
        # 검색된 문서를 format_context에 맞게 변환
        formatted_search_results = [(doc.text, score, doc.metadata) for doc, score in retrieved_docs]
        context = self.format_context(formatted_search_results)
        system_prompt = self.get_system_prompt()
        
        prompt = f"""
        {system_prompt}

        ---
        참고 자료 (일반 정보):
        {context}
        ---

        사용자 질문:
        {query}

        답변:
        """
        return prompt
    
    def _extract_contact_info(self, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> List[str]:
        """검색 결과에서 연락처 정보 추출"""
        contacts = []
        
        for text, score, metadata in search_results:
            if metadata.get('category') == 'contact':
                # 이미 포맷된 연락처 정보
                contacts.append(text.strip())
            elif 'phone' in metadata:
                # 메타데이터에서 연락처 정보 구성
                dept = metadata.get('department', '')
                position = metadata.get('position', '')
                phone = metadata.get('phone', '')
                task = metadata.get('task_area', '')
                
                if phone and dept:
                    contact_text = f"담당부서: {dept}"
                    if position:
                        contact_text += f" {position}"
                    contact_text += f"\n연락처: {phone}"
                    if task:
                        contact_text += f"\n담당업무: {task}"
                    contacts.append(contact_text)
        
        return contacts[:3]  # 최대 3개 연락처만 반환
    
    def _enhance_response_with_contacts(self, base_response: str, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """기본 응답에 관련 담당자 연락처 정보 추가"""
        contacts = self._extract_contact_info(search_results)
        
        if not contacts:
            return base_response
        
        enhanced_response = base_response
        
        if not any(keyword in base_response for keyword in ['연락처', '담당자', '055-254']):
            enhanced_response += "\n\n**관련 담당부서 연락처:**\n"
            for i, contact in enumerate(contacts, 1):
                enhanced_response += f"\n{i}. {contact}\n"
        
        return enhanced_response
    
    def handle(self, request: QueryRequest) -> HandlerResponse:
        """
        general 도메인 특화 처리
        기본 handle() 호출 후 연락처 정보 자동 추가
        """
        # 기본 핸들러 로직 실행
        response = super().handle(request)
        
        # QueryRequest에서 쿼리 텍스트 추출
        query = getattr(request, 'query', None) or getattr(request, 'text', '')
        
        # general 도메인 특화: 연락처 정보 보강
        if response.confidence >= self.confidence_threshold:
            # 재검색하여 연락처 정보 추가
            search_results = self._hybrid_search(query, k=10)
            # 검색 결과를 _enhance_response_with_contacts에 맞는 형태로 변환
            formatted_search_results = [(doc.text, score, doc.metadata) for doc, score in search_results]
            enhanced_answer = self._enhance_response_with_contacts(response.answer, formatted_search_results)
            response.answer = enhanced_answer
        
        return response

