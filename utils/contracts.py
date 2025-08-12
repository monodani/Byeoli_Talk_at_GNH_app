# utils/contracts.py
"""
API 계약 정의 (Pydantic v2 호환)
요청/응답 모델 정의
"""

from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator
import json

class MessageRole(str, Enum):
    """메시지 역할"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class HandlerType(str, Enum):
    """핸들러 타입"""
    SATISFACTION = "satisfaction"
    GENERAL = "general"
    PUBLISH = "publish"
    CYBER = "cyber"
    MENU = "menu"
    NOTICE = "notice"
    FALLBACK = "fallback"
    
    @classmethod
    def get_all_domains(cls) -> List[str]:
        """모든 도메인 리스트 반환 (fallback 제외)"""
        return [h.value for h in cls if h != cls.FALLBACK]
    
    @classmethod
    def is_valid_domain(cls, domain: str) -> bool:
        """유효한 도메인인지 확인"""
        return domain in [h.value for h in cls]

class ChatTurn(BaseModel):
    """대화 턴"""
    model_config = ConfigDict(extra='allow')
    
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('timestamp', mode='before')
    @classmethod
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

class ConversationContext(BaseModel):
    """대화 컨텍스트"""
    model_config = ConfigDict(extra='allow')
    
    session_id: str
    turns: List[ChatTurn] = Field(default_factory=list)
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    current_topic: Optional[str] = None
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_turn(self, role: MessageRole, content: str, metadata: Dict[str, Any] = None):
        """대화 턴 추가"""
        turn = ChatTurn(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.turns.append(turn)
        self.updated_at = datetime.now()
    
    def get_recent_turns(self, n: int = 6) -> List[ChatTurn]:
        """최근 n개 턴 반환"""
        return self.turns[-n:] if self.turns else []
    
    def to_messages(self) -> List[Dict[str, str]]:
        """OpenAI 메시지 형식으로 변환"""
        messages = []
        for turn in self.turns:
            messages.append({
                "role": turn.role.value,
                "content": turn.content
            })
        return messages

class QueryRequest(BaseModel):
    """쿼리 요청"""
    model_config = ConfigDict(extra='allow')
    
    query: str
    domain: Optional[str] = None
    context: Optional[ConversationContext] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    max_results: int = Field(default=5, ge=1, le=20)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class Citation(BaseModel):
    """인용 정보"""
    model_config = ConfigDict(extra='allow')
    
    source_id: str
    text: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_markdown(self) -> str:
        """마크다운 형식으로 변환"""
        source = self.metadata.get('source', self.source_id)
        return f"[{source}]({self.source_id}): {self.text[:100]}..."

class HandlerResponse(BaseModel):
    """핸들러 응답"""
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
        """에러 응답 생성"""
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
        """성공 응답 생성"""
        return cls(
            success=True,
            domain=domain,
            answer=answer,
            confidence=confidence,
            citations=citations or [],
            metadata=metadata or {},
            processing_time=processing_time
        )
    
    def to_streamlit_format(self) -> Dict[str, Any]:
        """Streamlit 표시용 형식"""
        return {
            "answer": self.answer,
            "confidence": f"{self.confidence:.1%}",
            "domain": self.domain,
            "citations": len(self.citations),
            "processing_time": f"{self.processing_time:.2f}s" if self.processing_time else "N/A"
        }

class HandlerCandidate(BaseModel):
    """핸들러 후보"""
    model_config = ConfigDict(extra='allow')
    
    domain: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    is_rule_based: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __lt__(self, other: "HandlerCandidate") -> bool:
        """정렬을 위한 비교 연산자"""
        return self.confidence < other.confidence
    
    def __eq__(self, other: "HandlerCandidate") -> bool:
        """동등성 비교"""
        return self.domain == other.domain and self.confidence == other.confidence

class RouterDecision(BaseModel):
    """라우터 결정"""
    model_config = ConfigDict(extra='allow')
    
    primary_domain: str
    secondary_domain: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    candidates: List["HandlerCandidate"] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_candidates(cls, candidates: List["HandlerCandidate"]) -> "RouterDecision":
        """후보 리스트로부터 결정 생성"""
        if not candidates:
            return cls(
                primary_domain="fallback",
                confidence=0.0,
                candidates=[]
            )
        
        # 신뢰도 기준 정렬
        sorted_candidates = sorted(candidates, key=lambda x: x.confidence, reverse=True)
        
        primary = sorted_candidates[0]
        secondary = sorted_candidates[1] if len(sorted_candidates) > 1 else None
        
        return cls(
            primary_domain=primary.domain,
            secondary_domain=secondary.domain if secondary else None,
            confidence=primary.confidence,
            reasoning=primary.reasoning,
            candidates=sorted_candidates
        )

class IndexMetadata(BaseModel):
    """인덱스 메타데이터"""
    model_config = ConfigDict(extra='allow')
    
    domain: str
    document_count: int
    last_updated: datetime
    index_size_mb: float
    version: str = "1.0.0"
    schema_version: str = "1.0.0"
    
    @field_validator('last_updated', mode='before')
    @classmethod
    def parse_last_updated(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

class RouterResponse(BaseModel):
    """라우터 응답 (라우팅 결정 + 핸들러 실행 결과)"""
    model_config = ConfigDict(extra='allow')
    
    query: str
    decision: RouterDecision
    responses: List[HandlerResponse] = Field(default_factory=list)
    final_answer: str = ""
    final_confidence: float = 0.0
    total_processing_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_routing_result(
        cls, 
        query: str,
        decision: RouterDecision,
        responses: List[HandlerResponse],
        processing_time: float = 0.0
    ) -> "RouterResponse":
        """라우팅 결과로부터 응답 생성"""
        # 최적 응답 선택
        valid_responses = [r for r in responses if r.success]
        
        if valid_responses:
            best = max(valid_responses, key=lambda x: x.confidence)
            final_answer = best.answer
            final_confidence = best.confidence
        else:
            # 모든 응답이 실패한 경우
            final_answer = "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다."
            final_confidence = 0.0
        
        return cls(
            query=query,
            decision=decision,
            responses=responses,
            final_answer=final_answer,
            final_confidence=final_confidence,
            total_processing_time=processing_time
        )
    
    def to_streamlit_message(self) -> Dict[str, Any]:
        """Streamlit 메시지 형식으로 변환"""
        return {
            "role": "assistant",
            "content": self.final_answer,
            "metadata": {
                "domain": self.decision.primary_domain,
                "confidence": self.final_confidence,
                "processing_time": self.total_processing_time,
                "citations": sum(len(r.citations) for r in self.responses)
            }
        }

class RoutingResult(BaseModel):
    """라우팅 결과"""
    model_config = ConfigDict(extra='allow')
    
    decision: RouterDecision
    responses: List[HandlerResponse]
    selected_response: Optional[HandlerResponse] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_best_response(self) -> Optional[HandlerResponse]:
        """최적 응답 선택"""
        if self.selected_response:
            return self.selected_response
        
        if not self.responses:
            return None
        
        # 신뢰도 기준 정렬
        valid_responses = [r for r in self.responses if r.success]
        if not valid_responses:
            return self.responses[0]  # 모두 실패시 첫 번째 반환
        
        return max(valid_responses, key=lambda x: x.confidence)

class SystemStatus(BaseModel):
    """시스템 상태"""
    model_config = ConfigDict(extra='allow')
    
    status: Literal["healthy", "degraded", "error"]
    domains: Dict[str, bool]  # domain -> is_loaded
    total_documents: int
    memory_usage_mb: float
    uptime_seconds: float
    last_check: datetime = Field(default_factory=datetime.now)
    errors: List[str] = Field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"
    
    @property
    def loaded_domains(self) -> List[str]:
        return [d for d, loaded in self.domains.items() if loaded]

class SearchResult(BaseModel):
    """검색 결과"""
    model_config = ConfigDict(extra='allow')
    
    text: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_id: Optional[str] = None
    
    def to_citation(self) -> Citation:
        """Citation으로 변환"""
        return Citation(
            source_id=self.source_id or "unknown",
            text=self.text,
            relevance_score=self.score,
            metadata=self.metadata
        )

class PerformanceMetrics(BaseModel):
    """성능 메트릭"""
    model_config = ConfigDict(extra='allow')
    
    # 시간 메트릭 (초 단위)
    total_time: float = 0.0
    routing_time: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    first_token_time: Optional[float] = None
    
    # 검색 메트릭
    documents_retrieved: int = 0
    documents_used: int = 0
    avg_relevance_score: float = 0.0
    
    # 품질 메트릭
    confidence_score: float = 0.0
    domains_checked: List[str] = Field(default_factory=list)
    cache_hit: bool = False
    
    # 리소스 메트릭
    tokens_used: int = 0
    memory_mb: Optional[float] = None
    
    # 타임스탬프
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def get_summary(self) -> str:
        """요약 문자열 반환"""
        return (
            f"총 {self.total_time:.2f}초 | "
            f"신뢰도 {self.confidence_score:.1%} | "
            f"문서 {self.documents_used}/{self.documents_retrieved}개 | "
            f"캐시 {'HIT' if self.cache_hit else 'MISS'}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "times": {
                "total": round(self.total_time, 3),
                "routing": round(self.routing_time, 3),
                "retrieval": round(self.retrieval_time, 3),
                "generation": round(self.generation_time, 3),
                "first_token": round(self.first_token_time, 3) if self.first_token_time else None
            },
            "search": {
                "retrieved": self.documents_retrieved,
                "used": self.documents_used,
                "avg_score": round(self.avg_relevance_score, 3)
            },
            "quality": {
                "confidence": round(self.confidence_score, 3),
                "domains": self.domains_checked,
                "cache_hit": self.cache_hit
            },
            "resources": {
                "tokens": self.tokens_used,
                "memory_mb": round(self.memory_mb, 2) if self.memory_mb else None
            }
        }

class ErrorResponse(BaseModel):
    """에러 응답"""
    model_config = ConfigDict(extra='allow')
    
    error_type: str
    error_message: str
    domain: Optional[str] = None
    query: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    traceback: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        domain: Optional[str] = None,
        query: Optional[str] = None,
        include_traceback: bool = False
    ) -> "ErrorResponse":
        """예외로부터 에러 응답 생성"""
        import traceback as tb
        
        return cls(
            error_type=type(exception).__name__,
            error_message=str(exception),
            domain=domain,
            query=query,
            traceback=tb.format_exc() if include_traceback else None,
            suggestions=cls._get_suggestions(exception)
        )
    
    @staticmethod
    def _get_suggestions(exception: Exception) -> List[str]:
        """예외 타입에 따른 제안사항 생성"""
        suggestions = []
        error_type = type(exception).__name__
        
        if "Index" in error_type or "Vector" in error_type:
            suggestions.append("인덱스를 재구축해보세요: make build-index")
        if "API" in error_type or "OpenAI" in error_type:
            suggestions.append("API 키를 확인해주세요")
            suggestions.append("API 할당량을 확인해주세요")
        if "Timeout" in error_type:
            suggestions.append("네트워크 연결을 확인해주세요")
            suggestions.append("요청을 다시 시도해주세요")
        if "Memory" in error_type:
            suggestions.append("시스템 메모리를 확인해주세요")
            suggestions.append("캐시를 정리해보세요: make clean-cache")
        
        if not suggestions:
            suggestions.append("문제가 지속되면 관리자에게 문의하세요")
        
        return suggestions
    
    def to_user_message(self) -> str:
        """사용자 친화적 메시지 생성"""
        message = f"죄송합니다. 요청을 처리하는 중 문제가 발생했습니다.\n\n"
        
        if self.domain:
            message += f"**도메인**: {self.domain}\n"
        
        message += f"**오류**: {self.error_message}\n\n"
        
        if self.suggestions:
            message += "**제안사항**:\n"
            for suggestion in self.suggestions:
                message += f"• {suggestion}\n"
        
        return message

class StreamingResponse(BaseModel):
    """스트리밍 응답"""
    model_config = ConfigDict(extra='allow')
    
    chunk_id: int
    content: str
    is_final: bool = False
    domain: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_sse_format(self) -> str:
        """Server-Sent Events 형식으로 변환"""
        import json
        data = {
            "id": self.chunk_id,
            "content": self.content,
            "final": self.is_final
        }
        if self.domain:
            data["domain"] = self.domain
        if self.confidence is not None:
            data["confidence"] = self.confidence
        
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

class CacheEntry(BaseModel):
    """캐시 엔트리"""
    model_config = ConfigDict(extra='allow')
    
    key: str
    value: Any
    created_at: datetime = Field(default_factory=datetime.now)
    accessed_at: datetime = Field(default_factory=datetime.now)
    access_count: int = 1
    ttl_seconds: Optional[int] = 3600  # 기본 1시간
    
    def is_expired(self) -> bool:
        """만료 여부 확인"""
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds
    
    def touch(self):
        """접근 시간 업데이트"""
        self.accessed_at = datetime.now()
        self.access_count += 1

class DomainConfig(BaseModel):
    """도메인 설정"""
    model_config = ConfigDict(extra='allow')
    
    name: str
    description: str
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results: int = Field(default=5, ge=1, le=20)
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    overlap: int = Field(default=100, ge=0, le=500)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# 유틸리티 함수
def validate_schema(data: Dict[str, Any], model_class: type[BaseModel]) -> tuple[bool, Optional[str]]:
    """스키마 검증"""
    try:
        model_class(**data)
        return True, None
    except Exception as e:
        return False, str(e)

def serialize_model(model: BaseModel) -> str:
    """모델을 JSON 문자열로 직렬화"""
    return model.model_dump_json(indent=2)

def deserialize_model(json_str: str, model_class: type[BaseModel]) -> BaseModel:
    """JSON 문자열을 모델로 역직렬화"""
    data = json.loads(json_str)
    return model_class(**data)
