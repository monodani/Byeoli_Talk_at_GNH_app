#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - contracts.py

시스템 전체의 인터페이스 계약을 정의하는 Pydantic 모델 모음
- QueryRequest: 사용자 요청 표준화
- HandlerResponse: 핸들러 응답 표준화  
- ConversationContext: 대화 상태 관리
- Citation: 소스 인용 표준화
- 모든 데이터 교환 시 타입 안전성 보장
"""

import logging
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator

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
    role: MessageRole = Field(..., description="메시지 역할 (user/assistant)")
    text: str = Field(..., min_length=1, description="메시지 텍스트")
    ts: datetime = Field(default_factory=datetime.now, description="타임스탬프")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationContext(BaseModel):
    """대화 컨텍스트 (6턴 윈도우, 4턴마다 요약)"""
    conversation_id: str = Field(default_factory=lambda: str(uuid4()), description="대화 세션 ID")
    summary: str = Field(default="", description="대화 요약 (1,000토큰 제한)")
    recent_messages: List[ChatTurn] = Field(default_factory=list, description="최근 6턴 메시지")
    entities: List[str] = Field(default_factory=list, description="추출된 핵심 엔티티")
    updated_at: datetime = Field(default_factory=datetime.now, description="최종 갱신 시간")
    
    @validator('recent_messages')
    def validate_message_limit(cls, v):
        """최근 메시지 6턴 제한"""
        if len(v) > 6:
            return v[-6:]  # 최신 6개만 유지
        return v
    
    @validator('summary')
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
# 3. 요청/응답 모델
# ================================================================

class QueryRequest(BaseModel):
    """표준화된 사용자 요청"""
    text: str = Field(..., min_length=1, max_length=2000, description="사용자 질문 텍스트")
    context: Optional[ConversationContext] = Field(default=None, description="대화 컨텍스트")
    follow_up: bool = Field(default=False, description="후속 질문 여부 (θ-0.02 완화)")
    trace_id: str = Field(default_factory=lambda: str(uuid4())[:8], description="요청 추적 ID")
    routing_hints: Dict[str, Any] = Field(default_factory=dict, description="라우팅 힌트 (선택적)")
    timestamp: datetime = Field(default_factory=datetime.now, description="요청 시간")
    
    @validator('text')
    def validate_text_content(cls, v):
        """질문 텍스트 검증"""
        v = v.strip()
        if not v:
            raise ValueError("질문이 비어있습니다.")
        if len(v) < 2:
            raise ValueError("질문이 너무 짧습니다.")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Citation(BaseModel):
    """소스 인용 정보 (2-3건 필수)"""
    source_id: str = Field(..., description="소스 식별자 (예: publish/2025plan.pdf#p12)")
    snippet: Optional[str] = Field(default=None, max_length=200, description="관련 텍스트 발췌 (200자 제한)")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="관련성 점수")
    page_number: Optional[int] = Field(default=None, description="페이지 번호 (PDF용)")
    section_title: Optional[str] = Field(default=None, description="섹션 제목")
    
    @validator('snippet')
    def validate_snippet_length(cls, v):
        """snippet 길이 제한"""
        if v and len(v) > 200:
            return v[:197] + "..."
        return v


class HandlerResponse(BaseModel):
    """표준화된 핸들러 응답"""
    answer: str = Field(..., min_length=1, description="생성된 답변")
    citations: List[Citation] = Field(default_factory=list, description="소스 인용 목록 (2-3건 권장)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="컨피던스 점수")
    handler_id: HandlerType = Field(..., description="처리한 핸들러 ID")
    elapsed_ms: int = Field(..., ge=0, description="처리 시간 (밀리초)")
    used_context: Optional[Dict[str, Any]] = Field(default=None, description="사용된 컨텍스트 정보")
    reask: Optional[str] = Field(default=None, description="재질문 제안 (컨피던스 부족 시)")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="디버깅/분석 정보")
    timestamp: datetime = Field(default_factory=datetime.now, description="응답 시간")
    
    @validator('citations')
    def validate_citation_count(cls, v):
        """Citation 개수 검증 (2-3건 권장)"""
        if len(v) == 0:
            logger.warning("Citation이 없습니다. 소스 인용이 필요합니다.")
        elif len(v) > 5:
            logger.warning(f"Citation이 너무 많습니다 ({len(v)}개). 핵심만 선별 권장.")
        return v
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """컨피던스 레벨 자동 분류"""
        # 핸들러별 θ 값은 외부에서 주입받아야 하므로, 일반적 기준 사용
        if self.confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.60:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    @property
    def is_reliable(self) -> bool:
        """신뢰할만한 응답인지 여부"""
        return self.confidence >= 0.65 and len(self.citations) >= 1
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ================================================================
# 4. 라우팅 관련 모델
# ================================================================

class HandlerCandidate(BaseModel):
    """핸들러 후보 (라우팅 단계)"""
    handler_id: HandlerType = Field(..., description="핸들러 ID")
    rule_score: float = Field(default=0.0, ge=0.0, le=1.0, description="규칙 기반 점수")
    llm_score: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM 기반 점수")
    combined_score: float = Field(default=0.0, ge=0.0, le=1.0, description="종합 점수")
    reasoning: str = Field(default="", description="선정 근거")
    
    @validator('combined_score', always=True)
    def calculate_combined_score(cls, v, values):
        """종합 점수 자동 계산 (rule:llm = 0.3:0.7)"""
        rule_score = values.get('rule_score', 0.0)
        llm_score = values.get('llm_score', 0.0)
        return round(rule_score * 0.3 + llm_score * 0.7, 3)


class RouterResponse(BaseModel):
    """라우터 응답 (Top-2 핸들러 선정 결과)"""
    selected_handlers: List[HandlerCandidate] = Field(..., max_items=2, description="선정된 핸들러 (최대 2개)")
    selection_time_ms: int = Field(..., ge=0, description="선정 소요 시간")
    routing_strategy: str = Field(default="hybrid", description="사용된 라우팅 전략")
    trace_id: str = Field(..., description="요청 추적 ID")
    
    @validator('selected_handlers')
    def validate_handler_count(cls, v):
        """핸들러 개수 검증"""
        if len(v) == 0:
            raise ValueError("최소 1개의 핸들러가 선정되어야 합니다.")
        if len(v) > 2:
            return v[:2]  # 상위 2개만 유지
        return v


# ================================================================
# 5. 캐시 관련 모델
# ================================================================

class CacheEntry(BaseModel):
    """캐시 항목"""
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

class PerformanceMetrics(BaseModel):
    """성능 메트릭"""
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
        """1.5초 타임박스 준수 여부"""
        return self.total_time_ms <= 1500


# ================================================================
# 7. 에러 처리 모델
# ================================================================

class ErrorResponse(BaseModel):
    """에러 응답"""
    error_code: str = Field(..., description="에러 코드")
    error_message: str = Field(..., description="에러 메시지")
    handler_id: Optional[HandlerType] = Field(default=None, description="에러 발생 핸들러")
    trace_id: str = Field(..., description="추적 ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="에러 시간")
    recovery_suggestion: Optional[str] = Field(default=None, description="복구 제안")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ================================================================
# 8. 유틸리티 함수들
# ================================================================

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


def create_error_response(
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


# ================================================================
# 9. 타입 힌트 별칭
# ================================================================

# 자주 사용되는 타입 조합들
QueryDict = Dict[str, Any]
ResponseDict = Dict[str, Any]
MetadataDict = Dict[str, Any]
ConfigDict = Dict[str, Any]

# Streamlit 상태 타입
StreamlitState = Dict[str, Any]

# 벡터스토어 관련 타입
DocumentScore = tuple[str, float, MetadataDict]  # (text, score, metadata)
SearchResults = List[DocumentScore]

# ================================================================
# 모듈 로드 완료 로그
# ================================================================

logger.info("✅ contracts.py 모듈 로드 완료 - 시스템 인터페이스 계약 정의됨")
