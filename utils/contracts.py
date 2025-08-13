# utils/contracts.py
"""
벼리톡 API 계약 정의 (Pydantic v2 - 최소 필수 버전)
실제로 필요한 클래스만 포함
"""

from typing import Optional, List, Dict, Any, Union, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator 
import json

# ===== 기본 Enum =====
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class HandlerType(str, Enum):
    SATISFACTION = "satisfaction"
    GENERAL = "general"
    PUBLISH = "publish"
    CYBER = "cyber"
    MENU = "menu"
    NOTICE = "notice"
    FALLBACK = "fallback"

# ===== 핵심 모델 =====
class ChatTurn(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConversationContext(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    session_id: str
    turns: List[ChatTurn] = Field(default_factory=list)
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    current_topic: Optional[str] = None
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_message(self, role, content, **kwargs):
        """메시지 추가 메소드"""
        new_turn = ChatTurn(role=role, content=content, timestamp=datetime.now())
        self.turns.append(new_turn)
        self.updated_at = datetime.now()
        if len(self.turns) > 6:
            self.turns = self.turns[-6:]



class QueryRequest(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    query: str
    domain: Optional[str] = None
    context: Optional[Union[ConversationContext, Dict[str, Any]]] = None # ✅ 수정: Union과 Optional 사용
    metadata: Dict[str, Any] = Field(default_factory=dict)
    max_results: int = Field(default=5, ge=1, le=20)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    @model_validator(mode="before")
    @classmethod
    def _coerce_payload(cls, data):
        if isinstance(data, dict):
            # ✅ text -> query 치환
            if "query" not in data and "text" in data:
                data["query"] = data.pop("text")
            # ✅ trace_id는 metadata로 흡수
            if "trace_id" in data:
                md = dict(data.get("metadata") or {})
                md["trace_id"] = data.pop("trace_id")
                data["metadata"] = md
        return data
        

class Citation(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    source_id: str
    text: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HandlerResponse(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    success: bool
    domain: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: List[Citation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    processing_time: Optional[float] = None
    
    @classmethod
    def create_error(cls, domain: str, error: str) -> "HandlerResponse":
        return cls(
            success=False,
            domain=domain,
            answer="",
            confidence=0.0,
            error=error
        )
    
    @classmethod
    def create_success(
        cls,
        domain: str,
        answer: str,
        confidence: float,
        citations: List[Citation] = None,
        metadata: Dict[str, Any] = None,
        processing_time: float = None
    ) -> "HandlerResponse":
        return cls(
            success=True,
            domain=domain,
            answer=answer,
            confidence=confidence,
            citations=citations or [],
            metadata=metadata or {},
            processing_time=processing_time
        )

# ===== 라우터 관련 =====
class HandlerCandidate(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    domain: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    is_rule_based: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RouterDecision(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    primary_domain: str
    secondary_domain: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    candidates: List[HandlerCandidate] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RouterResponse(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    query: str
    decision: RouterDecision
    responses: List[HandlerResponse] = Field(default_factory=list)
    final_answer: str = ""
    final_confidence: float = 0.0
    total_processing_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ===== 성능/시스템 =====
class PerformanceMetrics(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    total_time: float = 0.0
    routing_time: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    documents_retrieved: int = 0
    confidence_score: float = 0.0
    cache_hit: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def within_timebox(self) -> bool:
        """15초 타임박스 준수 여부"""
        return self.total_time_ms <= 15000

# ===== 오류 처리 =====
class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    error_type: str
    error_message: str
    domain: Optional[str] = None
    query: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    suggestions: List[str] = Field(default_factory=list)

# ===== 기타 필요 클래스 =====
class RoutingResult(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    decision: RouterDecision
    responses: List[HandlerResponse]
    selected_response: Optional[HandlerResponse] = None
    processing_time: float = 0.0

class SearchResult(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    text: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_id: Optional[str] = None

class DomainConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    name: str
    description: str
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results: int = Field(default=5, ge=1, le=20)

class CacheEntry(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    key: str
    value: Any
    created_at: datetime = Field(default_factory=datetime.now)
    ttl_seconds: Optional[int] = 3600

class StreamingResponse(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    chunk_id: int
    content: str
    is_final: bool = False
    domain: Optional[str] = None

class IndexMetadata(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    domain: str
    document_count: int
    last_updated: datetime
    version: str = "1.0.0"

class SystemStatus(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    status: Literal["healthy", "degraded", "error"]
    domains: Dict[str, bool]
    total_documents: int
    memory_usage_mb: float

# ===== 헬퍼 함수 =====
def create_error_response(
    domain: str,
    error: Exception,
    query: Optional[str] = None,
    include_traceback: bool = False
) -> HandlerResponse:
    """에러 응답 생성 헬퍼"""
    return HandlerResponse(
        success=False,
        domain=domain,
        answer=f"오류가 발생했습니다: {str(error)}",
        confidence=0.0,
        error=str(error),
        metadata={"error_type": type(error).__name__}
    )
