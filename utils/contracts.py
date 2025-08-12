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

class RouterDecision(BaseModel):
    """라우터 결정"""
    model_config = ConfigDict(extra='allow')
    
    primary_domain: str
    secondary_domain: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
