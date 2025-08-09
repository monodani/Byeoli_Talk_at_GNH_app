"""
Data contracts and Pydantic models for BYEOLI_TALK_AT_GNH_app.

Defines standardized request/response interfaces between components
to ensure type safety and API consistency across the system.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator


class HandlerType(str, Enum):
    """Available handler types."""
    GENERAL = "general"
    PUBLISH = "publish" 
    SATISFACTION = "satisfaction"
    CYBER = "cyber"
    MENU = "menu"
    NOTICE = "notice"
    FALLBACK = "fallback"


class QueryType(str, Enum):
    """Query classification types."""
    FACTUAL = "factual"           # 사실 정보 조회
    ANALYTICAL = "analytical"     # 분석/비교 요청
    PROCEDURAL = "procedural"     # 절차/방법 문의
    TEMPORAL = "temporal"         # 일정/시간 관련
    QUANTITATIVE = "quantitative" # 수치/통계 조회
    CONVERSATIONAL = "conversational"  # 대화형/후속 질문


class ConfidenceLevel(str, Enum):
    """Response confidence levels."""
    HIGH = "high"        # θ + 0.1 이상
    MEDIUM = "medium"    # θ ~ θ + 0.1
    LOW = "low"         # θ - 0.05 ~ θ
    VERY_LOW = "very_low"  # θ 미만


# ============================================================================
# Core Data Models
# ============================================================================

class ChatTurn(BaseModel):
    """Single conversation turn."""
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    text: str = Field(..., min_length=1, description="Message content")
    ts: datetime = Field(default_factory=datetime.now, description="Timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationContext(BaseModel):
    """Conversation context for maintaining dialogue state."""
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), 
                                description="Unique conversation identifier")
    summary: str = Field(default="", description="Conversation summary")
    recent_messages: List[ChatTurn] = Field(default_factory=list, 
                                          description="Recent message history")
    entities: List[str] = Field(default_factory=list,
                               description="Extracted entities from conversation")
    updated_at: datetime = Field(default_factory=datetime.now,
                               description="Last update timestamp")
    
    @validator('recent_messages')
    def validate_recent_messages_length(cls, v):
        """Validate recent messages don't exceed window size."""
        if len(v) > 12:  # 6 turns * 2 roles = 12 messages max
            return v[-12:]  # Keep only last 12
        return v
    
    @validator('entities')
    def validate_entities_length(cls, v):
        """Validate entity list doesn't exceed max size."""
        if len(v) > 10:  # Max 10 entities
            return v[:10]  # Keep first 10 (most important)
        return v
    
    def add_turn(self, role: str, text: str) -> None:
        """Add a new conversation turn."""
        self.recent_messages.append(ChatTurn(role=role, text=text))
        if len(self.recent_messages) > 12:
            self.recent_messages = self.recent_messages[-12:]
        self.updated_at = datetime.now()
    
    def get_conversation_history(self, max_turns: int = 6) -> str:
        """Get formatted conversation history."""
        recent = self.recent_messages[-(max_turns * 2):]  # Each turn has user+assistant
        history_parts = []
        
        for msg in recent:
            prefix = "사용자: " if msg.role == "user" else "챗봇: "
            history_parts.append(f"{prefix}{msg.text}")
        
        return "\n".join(history_parts)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Citation(BaseModel):
    """Source citation for response."""
    source_id: str = Field(..., description="Source identifier (e.g., 'publish/2025plan.pdf#p12')")
    snippet: Optional[str] = Field(None, description="Relevant text snippet")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    section: Optional[str] = Field(None, description="Section or subsection name")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, 
                                            description="Relevance confidence")
    
    @validator('source_id')
    def validate_source_id_format(cls, v):
        """Validate source ID follows expected format."""
        if not v or len(v) < 3:
            raise ValueError("source_id must be non-empty and at least 3 characters")
        return v
    
    def get_display_name(self) -> str:
        """Get human-readable source name."""
        if '#' in self.source_id:
            source_file, fragment = self.source_id.split('#', 1)
            source_name = source_file.split('/')[-1]  # Get filename only
            if self.page_number:
                return f"{source_name} (p.{self.page_number})"
            elif fragment.startswith('p'):
                page_num = fragment[1:]
                return f"{source_name} (p.{page_num})"
            else:
                return source_name
        else:
            return self.source_id.split('/')[-1]


class QueryRequest(BaseModel):
    """Standardized query request."""
    text: str = Field(..., min_length=1, max_length=2000, 
                     description="User query text")
    context: ConversationContext = Field(default_factory=ConversationContext,
                                       description="Conversation context")
    follow_up: bool = Field(default=False, 
                           description="Whether this is a follow-up question")
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()),
                         description="Request tracing identifier")
    routing_hints: Dict[str, Any] = Field(default_factory=dict,
                                        description="Optional routing hints")
    query_type: Optional[QueryType] = Field(None, description="Classified query type")
    priority: int = Field(default=1, ge=1, le=5, description="Query priority (1=highest)")
    
    @validator('text')
    def validate_query_text(cls, v):
        """Clean and validate query text."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query text cannot be empty or whitespace only")
        return cleaned
    
    def normalize_query(self) -> str:
        """Get normalized query for caching."""
        import re
        # Remove extra whitespace and normalize
        normalized = re.sub(r'\s+', ' ', self.text.strip().lower())
        # Remove common stop patterns for caching
        normalized = re.sub(r'^(음|어|그|저|이|그런데|그리고)\s*', '', normalized)
        return normalized


class HandlerCandidate(BaseModel):
    """Handler selection candidate."""
    handler_id: HandlerType = Field(..., description="Handler identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Selection confidence")
    reasoning: str = Field(..., description="Selection reasoning")
    estimated_response_time: Optional[float] = Field(None, ge=0.0, 
                                                   description="Estimated response time in seconds")


class UsedContext(BaseModel):
    """Information about context usage in response generation."""
    turns: int = Field(default=0, description="Number of conversation turns used")
    summary_hash: Optional[str] = Field(None, description="Hash of summary used")
    entities_used: List[str] = Field(default_factory=list, 
                                   description="Entities referenced in response")
    context_relevance: Optional[float] = Field(None, ge=0.0, le=1.0,
                                             description="How relevant context was")


class Diagnostics(BaseModel):
    """Detailed diagnostic information."""
    retrieval_time_ms: Optional[float] = Field(None, description="Retrieval time")
    generation_time_ms: Optional[float] = Field(None, description="Generation time")
    total_documents_searched: Optional[int] = Field(None, description="Documents searched")
    cache_hit: Optional[bool] = Field(None, description="Whether cache was hit")
    confidence_components: Optional[Dict[str, float]] = Field(None, 
                                                            description="Confidence breakdown")
    error_details: Optional[str] = Field(None, description="Error information if any")


class HandlerResponse(BaseModel):
    """Standardized handler response."""
    answer: str = Field(..., min_length=1, description="Generated answer")
    citations: List[Citation] = Field(..., min_items=0, max_items=5,
                                    description="Source citations")
    confidence: float = Field(..., ge=0.0, le=1.0, 
                             description="Response confidence score")
    handler_id: HandlerType = Field(..., description="Handler that generated response")
    elapsed_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence categorization")
    used_context: Optional[UsedContext] = Field(None, 
                                              description="Context usage information")
    reask: Optional[str] = Field(None, description="Suggested re-ask if confidence low")
    diagnostics: Optional[Diagnostics] = Field(None, 
                                             description="Detailed diagnostic info")
    
    @validator('citations')
    def validate_citations_count(cls, v):
        """Validate citation requirements."""
        if len(v) > 5:
            # Keep top 5 by confidence if available, otherwise first 5
            if all(c.confidence_score is not None for c in v):
                v.sort(key=lambda x: x.confidence_score, reverse=True)
            return v[:5]
        return v
    
    @root_validator
    def validate_confidence_level_consistency(cls, values):
        """Ensure confidence level matches confidence score."""
        confidence = values.get('confidence', 0.0)
        confidence_level = values.get('confidence_level')
        
        if confidence >= 0.8:
            expected_level = ConfidenceLevel.HIGH
        elif confidence >= 0.6:
            expected_level = ConfidenceLevel.MEDIUM
        elif confidence >= 0.4:
            expected_level = ConfidenceLevel.LOW
        else:
            expected_level = ConfidenceLevel.VERY_LOW
            
        if confidence_level != expected_level:
            values['confidence_level'] = expected_level
            
        return values
    
    def requires_citation_improvement(self) -> bool:
        """Check if response needs more citations."""
        if self.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]:
            return len(self.citations) < 2
        return len(self.citations) < 1
    
    def get_citation_summary(self) -> str:
        """Get formatted citation summary."""
        if not self.citations:
            return "출처 없음"
        
        sources = [cite.get_display_name() for cite in self.citations]
        if len(sources) == 1:
            return f"출처: {sources[0]}"
        elif len(sources) == 2:
            return f"출처: {sources[0]}, {sources[1]}"
        else:
            return f"출처: {sources[0]} 외 {len(sources)-1}건"


# ============================================================================
# Routing and Selection Models  
# ============================================================================

class RoutingResult(BaseModel):
    """Result of query routing process."""
    selected_handlers: List[HandlerCandidate] = Field(..., min_items=1, max_items=2,
                                                     description="Selected handler candidates")
    routing_confidence: float = Field(..., ge=0.0, le=1.0,
                                     description="Routing decision confidence")
    routing_time_ms: float = Field(..., ge=0.0, description="Routing time in milliseconds")
    routing_method: str = Field(..., description="Method used for routing")
    fallback_reason: Optional[str] = Field(None, description="Reason for fallback if any")


class ParallelExecutionResult(BaseModel):
    """Result of parallel handler execution."""
    responses: List[HandlerResponse] = Field(..., description="Handler responses")
    execution_time_ms: float = Field(..., ge=0.0, description="Total execution time")
    timed_out: bool = Field(default=False, description="Whether execution timed out")
    selected_response: Optional[HandlerResponse] = Field(None, 
                                                       description="Finally selected response")


# ============================================================================
# Cache Models
# ============================================================================

class CacheKey(BaseModel):
    """Cache key structure."""
    query_hash: str = Field(..., description="Normalized query hash")
    context_hash: Optional[str] = Field(None, description="Context hash")
    handler_id: Optional[str] = Field(None, description="Handler identifier")
    
    def to_string(self) -> str:
        """Convert to cache key string."""
        parts = [self.query_hash]
        if self.context_hash:
            parts.append(self.context_hash)
        if self.handler_id:
            parts.append(self.handler_id)
        return ":".join(parts)


class CacheEntry(BaseModel):
    """Cache entry structure."""
    key: CacheKey = Field(..., description="Cache key")
    value: Union[HandlerResponse, List[Citation], Dict[str, Any]] = Field(..., 
                                                                         description="Cached value")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    expires_at: datetime = Field(..., description="Expiration time")
    hit_count: int = Field(default=0, description="Cache hit count")
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > self.expires_at
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# Error Models
# ============================================================================

class ErrorType(str, Enum):
    """Error type classification."""
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    API_ERROR = "api_error"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"


class ErrorDetails(BaseModel):
    """Detailed error information."""
    error_type: ErrorType = Field(..., description="Error classification")
    message: str = Field(..., description="Error message")
    handler_id: Optional[str] = Field(None, description="Handler where error occurred")
    trace_id: Optional[str] = Field(None, description="Request trace ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional error context")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# Health Check and Status Models
# ============================================================================

class ComponentStatus(str, Enum):
    """Component status options."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheck(BaseModel):
    """System health check result."""
    overall_status: ComponentStatus = Field(..., description="Overall system status")
    components: Dict[str, ComponentStatus] = Field(..., description="Individual component status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# Utility Functions
# ============================================================================

def create_error_response(
    error_type: ErrorType,
    message: str,
    handler_id: str = "fallback",
    trace_id: Optional[str] = None
) -> HandlerResponse:
    """Create standardized error response."""
    return HandlerResponse(
        answer=f"죄송합니다. 처리 중 오류가 발생했습니다: {message}",
        citations=[],
        confidence=0.0,
        confidence_level=ConfidenceLevel.VERY_LOW,
        handler_id=HandlerType.FALLBACK,
        elapsed_ms=0.0,
        diagnostics=Diagnostics(
            error_details=f"{error_type}: {message}"
        )
    )


def create_timeout_response(
    handler_id: str,
    elapsed_ms: float,
    partial_answer: Optional[str] = None
) -> HandlerResponse:
    """Create timeout response with partial results."""
    answer = partial_answer or "처리 시간이 초과되어 완전한 응답을 제공할 수 없습니다."
    
    return HandlerResponse(
        answer=answer,
        citations=[],
        confidence=0.3,  # Low confidence for timeout
        confidence_level=ConfidenceLevel.LOW,
        handler_id=HandlerType(handler_id),
        elapsed_ms=elapsed_ms,
        diagnostics=Diagnostics(
            error_details="Request timed out"
        )
    )


def normalize_query_for_cache(query: str) -> str:
    """Normalize query text for consistent caching."""
    import re
    import unicodedata
    
    # Unicode normalization
    normalized = unicodedata.normalize('NFKC', query)
    
    # Convert to lowercase
    normalized = normalized.lower().strip()
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove common conversation starters
    patterns = [
        r'^(음|어|그|저|이|그런데|그리고|그러면|그럼|자|잠깐|잠시)\s*',
        r'\s*(좀|좀더|조금|약간)\s*',
        r'[?!。．]+$'  # Remove trailing punctuation
    ]
    
    for pattern in patterns:
        normalized = re.sub(pattern, '', normalized)
    
    return normalized.strip()
