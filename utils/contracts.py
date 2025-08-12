#!/usr/bin/env python3
"""
벼리톡@경상남도인재개발원 (경상남도인재개발원 RAG 챗봇) - contracts.py (최종 수정)

시스템 전체의 인터페이스 계약을 정의하는 Pydantic 모델 모음
- QueryRequest: 사용자 요청 표준화
- HandlerResponse: 핸들러 응답 표준화  
- ConversationContext: 대화 상태 관리
- Citation: 소스 인용 표준화
- 모든 데이터 교환 시 타입 안전성 보장

🚨 중요: TextChunk는 utils.textifier에서만 정의하고 여기서는 제거함
"""

import logging
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict

# 로깅 설정
logger = logging.getLogger(__name__)


# ================================================================
# 1. 기본 열거형 타입
# ================================================================

class MessageRole(str, Enum):
    """대화 메시지 역할"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class HandlerType(str, Enum):
    """핸들러 도메인 타입"""
    SATISFACTION = "satisfaction"
    GENERAL = "general"
    MENU = "menu"
    CYBER = "cyber"
    PUBLISH = "publish"
    NOTICE = "notice"
    FALLBACK = "fallback"


class ConfidenceLevel(str, Enum):
    """컨피던스 레벨 분류"""
    HIGH = "high"      # θ + 0.1 이상
    MEDIUM = "medium"  # θ ± 0.1
    LOW = "low"        # θ - 0.1 이하


# ================================================================
# 2. 대화 관련 모델
# ================================================================

class ChatTurn(BaseModel):
    """개별 대화 턴"""
    model_config = ConfigDict(
        extra='forbid',
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    role: MessageRole = Field(..., description="메시지 역할 (user/assistant)")
    text: str = Field(..., min_length=1, description="메시지 텍스트")
    ts: datetime = Field(default_factory=datetime.now, description="타임스탬프")


class ConversationContext(BaseModel):
    """대화 컨텍스트 (6턴 윈도우, 4턴마다 요약)"""
    model_config = ConfigDict(extra='forbid')
    
    conversation_id: str = Field(default_factory=lambda: str(uuid4()), description="대화 세션 ID")
    summary: str = Field(default="", description="대화 요약 (1,000토큰 제한)")
    recent_messages: List[ChatTurn] = Field(default_factory=list, description="최근 6턴 메시지")
    entities: List[str] = Field(default_factory=list, description="추출된 핵심 엔티티")
    updated_at: datetime = Field(default_factory=datetime.now, description="최종 갱신 시간")
    
    @field_validator('recent_messages')
    @classmethod
    def validate_message_limit(cls, v):
        """최근 메시지 6턴 제한"""
        if len(v) > 6:
            return v[-6:]  # 최신 6개만 유지
        return v
    
    @field_validator('summary')
    @classmethod
    def validate_summary_length(cls, v):
        """요약 길이 제한 (대략 1,000토큰)"""
        if len(v) > 4000:  # 한글 기준 4,000자 ≈ 1,000토큰
            logger.warning("요약이 너무 깁니다. 자동 압축이 필요합니다.")
        return v
    
    def add_message(self, role: MessageRole, text: str) -> None:
        """새 메시지 추가 (자동으로 6턴 윈도우 유지)"""
        new_turn = ChatTurn(role=role, text=text)
        self.recent_messages.append(new_turn)
        
        # 6턴 제한 유지
        if len(self.recent_messages) > 6:
            self.recent_messages = self.recent_messages[-6:]
        
        self.updated_at = datetime.now()
    
    def should_update_summary(self) -> bool:
        """요약 갱신 필요 여부 (4턴마다 또는 1,000토큰 초과 시)"""
        turn_count = len(self.recent_messages)
        summary_tokens = len(self.summary) // 4  # 대략적 토큰 수
        
        return (turn_count % 4 == 0 and turn_count > 0) or summary_tokens > 1000
    
    def get_context_hash(self) -> str:
        """컨텍스트 해시 (캐시 키 생성용)"""
        import hashlib
        
        # 요약 + 엔티티 기반 해시
        content = f"{self.summary}|{','.join(sorted(self.entities))}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


# ================================================================
# 3. 라우팅 관련 모델
# ================================================================

class HandlerCandidate(BaseModel):
    """핸들러 후보 (라우팅 단계)"""
    model_config = ConfigDict(extra='forbid')
    
    handler_id: HandlerType = Field(..., description="핸들러 ID")
    rule_score: float = Field(default=0.0, ge=0.0, le=1.0, description="규칙 기반 점수")
    llm_score: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM 기반 점수")
    combined_score: float = Field(default=0.0, ge=0.0, le=1.0, description="종합 점수")
    reasoning: str = Field(default="", description="선정 근거")
    
    @field_validator('combined_score', mode='before')
    @classmethod
    def calculate_combined_score(cls, v, info):
        """종합 점수 자동 계산 (rule:llm = 0.3:0.7)"""
        if info.data:
            rule_score = info.data.get('rule_score', 0.0)
            llm_score = info.data.get('llm_score', 0.0)
            return round(rule_score * 0.3 + llm_score * 0.7, 3)
        return v


class RouterResponse(BaseModel):
    """라우터 응답 (Top-2 핸들러 선정 결과)"""
    model_config = ConfigDict(extra='forbid')
    
    selected_handlers: List[HandlerCandidate] = Field(..., max_length=2, description="선정된 핸들러 (최대 2개)")
    selection_time_ms: int = Field(..., ge=0, description="선정 소요 시간")
    routing_strategy: str = Field(default="hybrid", description="사용된 라우팅 전략")
    trace_id: str = Field(..., description="요청 추적 ID")
    
    @field_validator('selected_handlers')
    @classmethod
    def validate_handler_count(cls, v):
        """핸들러 개수 검증"""
        if len(v) == 0:
            raise ValueError("최소 1개의 핸들러가 선정되어야 합니다.")
        if len(v) > 2:
            return v[:2]  # 상위 2개만 유지
        return v


# ================================================================
# 4. 요청/응답 모델
# ================================================================

class QueryRequest(BaseModel):
    """표준화된 사용자 요청"""
    model_config = ConfigDict(extra='forbid')
    
    text: str = Field(..., min_length=1, max_length=2000, description="사용자 질문 텍스트")
    context: Optional[ConversationContext] = Field(default=None, description="대화 컨텍스트")
    follow_up: bool = Field(default=False, description="후속 질문 여부 (θ-0.02 완화)")
    trace_id: str = Field(default_factory=lambda: str(uuid4())[:8], description="요청 추적 ID")
    routing_hints: Dict[str, Any] = Field(default_factory=dict, description="라우팅 힌트 (선택적)")
    timestamp: datetime = Field(default_factory=datetime.now, description="요청 시간")
    
    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v):
        """질문 텍스트 검증"""
        v = v.strip()
        if not v:
            raise ValueError("질문이 비어있습니다.")
        if len(v) < 2:
            raise ValueError("질문이 너무 짧습니다.")
        return v


class Citation(BaseModel):
    """소스 인용 정보"""
    model_config = ConfigDict(extra='forbid')
    
    source_file: str = Field(..., description="소스 파일명")
    source_id: str = Field(..., description="소스 고유 식별자")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="관련성 점수")
    context: str = Field(..., max_length=200, description="인용 맥락 (200자 제한)")
    
    def __str__(self) -> str:
        return f"[{self.source_file}] {self.context[:50]}..."


class HandlerResponse(BaseModel):
    """핸들러 응답 표준화"""
    model_config = ConfigDict(extra='forbid')
    
    content: str = Field(..., min_length=1, description="응답 내용")
    confidence: float = Field(..., ge=0.0, le=1.0, description="컨피던스 점수")
    handler_type: HandlerType = Field(..., description="처리한 핸들러 타입")
    citations: List[Citation] = Field(default_factory=list, description="소스 인용 목록")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")
    processing_time: float = Field(default=0.0, ge=0.0, description="처리 시간 (초)")
    
    @field_validator('citations')
    @classmethod
    def validate_citations_limit(cls, v):
        """인용 개수 제한 (최대 5개)"""
        if len(v) > 5:
            return v[:5]  # 상위 5개만 유지
        return v
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """컨피던스 레벨 분류"""
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def add_citation(self, source_file: str, source_id: str, 
                    relevance_score: float, context: str) -> None:
        """인용 추가 (길이 제한 적용)"""
        # context 길이 제한
        truncated_context = context[:200] if len(context) > 200 else context
        
        citation = Citation(
            source_file=source_file,
            source_id=source_id,
            relevance_score=relevance_score,
            context=truncated_context
        )
        
        self.citations.append(citation)
        
        # 최대 5개 제한
        if len(self.citations) > 5:
            self.citations = self.citations[:5]


# ================================================================
# 5. 캐시 관련 모델
# ================================================================

class CacheEntry(BaseModel):
    """캐시 항목"""
    model_config = ConfigDict(extra='forbid')
    
    key: str = Field(..., description="캐시 키")
    value: Any = Field(..., description="캐시 값")
    ttl_seconds: int = Field(..., gt=0, description="TTL (초)")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    access_count: int = Field(default=0, description="접근 횟수")
    
    @property
    def is_expired(self) -> bool:
        """만료 여부 확인"""
        from datetime import timedelta
        expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry_time
    
    def access(self) -> None:
        """접근 시 카운터 증가"""
        self.access_count += 1


# ================================================================
# 6. 성능/진단 관련 모델
# ================================================================

class ProcessingMetrics(BaseModel):
    """처리 성능 메트릭"""
    model_config = ConfigDict(extra='forbid')
    
    query_hash: str = Field(..., description="쿼리 해시")
    handler_type: HandlerType = Field(..., description="핸들러 타입")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="컨피던스")
    processing_time: float = Field(..., ge=0.0, description="처리 시간")
    cache_hit: bool = Field(default=False, description="캐시 히트 여부")
    retrieval_count: int = Field(default=0, ge=0, description="검색된 문서 수")
    timestamp: datetime = Field(default_factory=datetime.now, description="처리 시간")


class PerformanceMetrics(BaseModel):
    """성능 메트릭"""
    model_config = ConfigDict(extra='forbid')
    
    total_time_ms: int = Field(..., ge=0, description="총 처리 시간")
    router_time_ms: int = Field(default=0, ge=0, description="라우터 시간")
    handler_time_ms: int = Field(default=0, ge=0, description="핸들러 시간")
    retrieval_time_ms: int = Field(default=0, ge=0, description="검색 시간")
    generation_time_ms: int = Field(default=0, ge=0, description="생성 시간")
    cache_hits: int = Field(default=0, ge=0, description="캐시 히트 수")
    cache_misses: int = Field(default=0, ge=0, description="캐시 미스 수")
    
    @property
    def cache_hit_rate(self) -> float:
        """캐시 히트율"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def within_timebox(self) -> bool:
        """15.0초 타임박스 준수 여부"""
        return 2000 <= self.total_time_ms <= 15000


class ErrorLog(BaseModel):
    """오류 로깅 모델"""
    model_config = ConfigDict(extra='forbid')
    
    error_type: str = Field(..., description="오류 타입")
    error_message: str = Field(..., description="오류 메시지")
    handler_type: Optional[HandlerType] = Field(default=None, description="핸들러 타입")
    query_text: str = Field(default="", description="쿼리 텍스트")
    trace_id: str = Field(..., description="추적 ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="오류 시간")


# ================================================================
# 7. 에러 처리 모델
# ================================================================

class ErrorResponse(BaseModel):
    """에러 응답"""
    model_config = ConfigDict(
        extra='forbid',
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    error_code: str = Field(..., description="에러 코드")
    error_message: str = Field(..., description="에러 메시지")
    handler_id: Optional[HandlerType] = Field(default=None, description="에러 발생 핸들러")
    trace_id: str = Field(..., description="추적 ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="오류 시간")
    recovery_suggestion: Optional[str] = Field(default=None, description="복구 제안")


# ================================================================
# 8. 유틸리티 함수들
# ================================================================

def create_error_response(error_msg: str, handler_type: HandlerType = HandlerType.FALLBACK) -> HandlerResponse:
    """오류 응답 생성"""
    return HandlerResponse(
        content=f"죄송합니다. 처리 중 오류가 발생했습니다: {error_msg}",
        confidence=0.0,
        handler_type=handler_type,
        citations=[],
        metadata={"error": True, "error_message": error_msg}
    )


def create_fallback_response(query_text: str) -> HandlerResponse:
    """폴백 응답 생성"""
    return HandlerResponse(
        content=f"'{query_text}' 관련 정보를 찾지 못했습니다. 다시 질문해 주시거나 더 구체적으로 문의해 주세요.",
        confidence=0.1,
        handler_type=HandlerType.FALLBACK,
        citations=[],
        metadata={"fallback": True}
    )


def create_query_request(
    text: str,
    context: Optional[ConversationContext] = None,
    follow_up: bool = False,
    **kwargs
) -> QueryRequest:
    """QueryRequest 생성 헬퍼"""
    return QueryRequest(
        text=text,
        context=context,
        follow_up=follow_up,
        **kwargs
    )


def create_error_response_model(
    error_code: str,
    error_message: str,
    trace_id: str,
    handler_id: Optional[HandlerType] = None,
    recovery_suggestion: Optional[str] = None
) -> ErrorResponse:
    """ErrorResponse 생성 헬퍼"""
    return ErrorResponse(
        error_code=error_code,
        error_message=error_message,
        trace_id=trace_id,
        handler_id=handler_id,
        recovery_suggestion=recovery_suggestion
    )


def normalize_query(text: str) -> str:
    """쿼리 정규화 (캐시 키 생성용)"""
    import re
    
    # 공백 정리 및 소문자 변환
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    
    # 특수문자 제거 (한글, 영문, 숫자, 기본 문장부호만 유지)
    normalized = re.sub(r'[^\w\s가-힣.,?!]', '', normalized)
    
    return normalized


def truncate_text(text: str, max_length: int = 200) -> str:
    """텍스트 길이 제한 (Citation context용)"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


# ================================================================
# 9. 기본 내보내기
# ================================================================

__all__ = [
    # 열거형
    'MessageRole', 'HandlerType', 'ConfidenceLevel',
    
    # 핵심 모델 (TextChunk는 utils.textifier에서 import)
    'ChatTurn', 'ConversationContext',
    'QueryRequest', 'HandlerResponse', 'Citation',
    
    # 라우팅 모델
    'HandlerCandidate', 'RouterResponse',
    
    # 캐시 & 성능
    'CacheEntry', 'ProcessingMetrics', 'PerformanceMetrics',
    
    # 모니터링 & 오류
    'ErrorLog', 'ErrorResponse',
    
    # 유틸리티
    'create_error_response', 'create_fallback_response', 
    'create_query_request', 'create_error_response_model',
    'normalize_query', 'truncate_text'
]
