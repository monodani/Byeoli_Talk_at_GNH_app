"""
Global configuration management for BYEOLI_TALK_AT_GNH_app.

Loads and validates environment variables, provides centralized access 
to all configuration settings across modules.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    """LLM and embedding model configuration."""
    # Primary LLM
    primary_llm_provider: str = "openai"
    primary_llm_model: str = "gpt-4o-mini"
    primary_llm_temperature: float = 0.3
    primary_llm_max_tokens: int = 2048
    
    # Lightweight LLM for routing
    routing_llm_provider: str = "openai" 
    routing_llm_model: str = "gpt-4o-mini"
    routing_llm_temperature: float = 0.1
    routing_llm_max_tokens: int = 256
    
    # Embedding model
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # API keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None


@dataclass 
class RetrievalConfig:
    """Retrieval and search configuration."""
    # Hybrid search parameters
    faiss_k: int = 5
    faiss_k_expanded: int = 12
    bm25_k: int = 5
    
    # RRF parameters
    rrf_k: int = 60
    rrf_weights: Dict[str, float] = field(default_factory=lambda: {
        "faiss": 0.7,
        "bm25": 0.3
    })
    
    # MMR parameters
    mmr_diversity_threshold: float = 0.5
    mmr_fetch_k: int = 20
    
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 100
    
    # Confidence thresholds per handler
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "general": 0.70,
        "publish": 0.74, 
        "satisfaction": 0.68,
        "cyber": 0.66,
        "menu": 0.64,
        "notice": 0.62,
        "fallback": 0.00
    })


@dataclass
class PerformanceConfig:
    """Performance and timing configuration."""
    # Timeout settings (seconds)
    total_timeout: float = 1.5
    candidate_selection_timeout: float = 0.4
    parallel_execution_timeout: float = 1.1
    
    # First token goal
    first_token_goal: float = 1.0
    total_response_goal_min: float = 2.0
    total_response_goal_max: float = 4.0
    
    # Cache TTL (seconds)
    cache_ttl: Dict[str, int] = field(default_factory=lambda: {
        "notice": 6 * 3600,  # 6 hours
        "menu": 6 * 3600,    # 6 hours
        "general": 30 * 24 * 3600,     # 30 days
        "publish": 30 * 24 * 3600,     # 30 days
        "satisfaction": 30 * 24 * 3600, # 30 days
        "cyber": 30 * 24 * 3600        # 30 days
    })
    
    # Parallel processing
    max_concurrent_handlers: int = 2
    handler_timeout: float = 1.0


@dataclass
class ConversationConfig:
    """Conversation and context management configuration."""
    # Context window
    recent_messages_window: int = 6
    summary_update_frequency: int = 4  # turns
    max_summary_tokens: int = 1000
    
    # Follow-up handling
    followup_confidence_adjustment: float = -0.02
    
    # Entity extraction
    max_entities: int = 10
    entity_relevance_threshold: float = 0.3


@dataclass
class CacheConfig:
    """Caching configuration."""
    # Cache types
    enable_query_cache: bool = True
    enable_retrieval_cache: bool = True
    enable_embedding_cache: bool = True
    
    # Cache directories  
    cache_dir: str = "cache"
    embedding_cache_dir: str = "cache/embeddings"
    query_cache_dir: str = "cache/queries"
    retrieval_cache_dir: str = "cache/retrievals"
    
    # Cache sizes (items)
    max_query_cache_size: int = 1000
    max_retrieval_cache_size: int = 5000
    max_embedding_cache_size: int = 10000


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Structured logging
    enable_json_logging: bool = True
    enable_masking: bool = True
    
    # Log files
    log_dir: str = "logs"
    app_log_file: str = "logs/app.log"
    performance_log_file: str = "logs/performance.log"
    error_log_file: str = "logs/error.log"
    
    # Sensitive fields to mask
    sensitive_fields: set = field(default_factory=lambda: {
        "api_key", "token", "password", "secret"
    })


@dataclass
class PathConfig:
    """File system path configuration."""
    # Root directory
    root_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.absolute())
    
    # Data directories
    data_dir: str = "data"
    schemas_dir: str = "schemas"
    vectorstores_dir: str = "vectorstores" 
    cache_dir: str = "cache"
    logs_dir: str = "logs"
    
    # Specific data subdirs
    data_general_dir: str = "data/general"
    data_publish_dir: str = "data/publish"
    data_satisfaction_dir: str = "data/satisfaction"
    data_cyber_dir: str = "data/cyber"
    data_menu_dir: str = "data/menu"
    data_notice_dir: str = "data/notice"
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert relative path to absolute path from root."""
        return self.root_dir / relative_path


class Config:
    """Main configuration class - singleton pattern."""
    
    _instance: Optional['Config'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._load_config()
            Config._initialized = True
    
    def _load_config(self):
        """Load configuration from environment variables."""
        # Load .env file if exists
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        
        # Initialize configurations
        self.model = self._load_model_config()
        self.retrieval = self._load_retrieval_config()
        self.performance = self._load_performance_config()
        self.conversation = self._load_conversation_config()
        self.cache = self._load_cache_config()
        self.logging = self._load_logging_config()
        self.paths = self._load_path_config()
        
        # Validate configuration
        self._validate_config()
        
        # Create necessary directories
        self._create_directories()
    
    def _load_model_config(self) -> ModelConfig:
        """Load model configuration from environment."""
        return ModelConfig(
            primary_llm_provider=os.getenv("PRIMARY_LLM_PROVIDER", "openai"),
            primary_llm_model=os.getenv("PRIMARY_LLM_MODEL", "gpt-4o-mini"),
            primary_llm_temperature=float(os.getenv("PRIMARY_LLM_TEMPERATURE", "0.3")),
            primary_llm_max_tokens=int(os.getenv("PRIMARY_LLM_MAX_TOKENS", "2048")),
            
            routing_llm_provider=os.getenv("ROUTING_LLM_PROVIDER", "openai"),
            routing_llm_model=os.getenv("ROUTING_LLM_MODEL", "gpt-4o-mini"),
            routing_llm_temperature=float(os.getenv("ROUTING_LLM_TEMPERATURE", "0.1")),
            routing_llm_max_tokens=int(os.getenv("ROUTING_LLM_MAX_TOKENS", "256")),
            
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
            
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    
    def _load_retrieval_config(self) -> RetrievalConfig:
        """Load retrieval configuration from environment."""
        config = RetrievalConfig()
        
        # Override with env vars if present
        config.faiss_k = int(os.getenv("FAISS_K", str(config.faiss_k)))
        config.faiss_k_expanded = int(os.getenv("FAISS_K_EXPANDED", str(config.faiss_k_expanded)))
        config.chunk_size = int(os.getenv("CHUNK_SIZE", str(config.chunk_size)))
        config.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", str(config.chunk_overlap)))
        
        # Load confidence thresholds
        for handler in config.confidence_thresholds:
            env_key = f"CONFIDENCE_THRESHOLD_{handler.upper()}"
            if os.getenv(env_key):
                config.confidence_thresholds[handler] = float(os.getenv(env_key))
        
        return config
    
    def _load_performance_config(self) -> PerformanceConfig:
        """Load performance configuration from environment."""
        config = PerformanceConfig()
        
        config.total_timeout = float(os.getenv("TOTAL_TIMEOUT", str(config.total_timeout)))
        config.candidate_selection_timeout = float(os.getenv("CANDIDATE_SELECTION_TIMEOUT", 
                                                           str(config.candidate_selection_timeout)))
        config.parallel_execution_timeout = float(os.getenv("PARALLEL_EXECUTION_TIMEOUT",
                                                           str(config.parallel_execution_timeout)))
        
        return config
    
    def _load_conversation_config(self) -> ConversationConfig:
        """Load conversation configuration from environment."""
        config = ConversationConfig()
        
        config.recent_messages_window = int(os.getenv("RECENT_MESSAGES_WINDOW", 
                                                    str(config.recent_messages_window)))
        config.summary_update_frequency = int(os.getenv("SUMMARY_UPDATE_FREQUENCY",
                                                       str(config.summary_update_frequency)))
        
        return config
    
    def _load_cache_config(self) -> CacheConfig:
        """Load cache configuration from environment."""
        config = CacheConfig()
        
        config.enable_query_cache = os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true"
        config.enable_retrieval_cache = os.getenv("ENABLE_RETRIEVAL_CACHE", "true").lower() == "true"
        config.enable_embedding_cache = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
        
        return config
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration from environment."""
        config = LoggingConfig()
        
        config.level = os.getenv("LOG_LEVEL", config.level)
        config.enable_json_logging = os.getenv("ENABLE_JSON_LOGGING", "true").lower() == "true"
        config.enable_masking = os.getenv("ENABLE_LOG_MASKING", "true").lower() == "true"
        
        return config
    
    def _load_path_config(self) -> PathConfig:
        """Load path configuration."""
        return PathConfig()
    
    def _validate_config(self):
        """Validate loaded configuration."""
        # Validate required API keys
        if self.model.primary_llm_provider == "openai" and not self.model.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI as primary LLM provider")
        
        if self.model.embedding_provider == "openai" and not self.model.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI embeddings")
        
        # Validate timeout consistency
        if (self.performance.candidate_selection_timeout + 
            self.performance.parallel_execution_timeout > 
            self.performance.total_timeout):
            raise ValueError("Sum of individual timeouts exceeds total timeout")
        
        # Validate confidence thresholds
        for handler, threshold in self.retrieval.confidence_thresholds.items():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Confidence threshold for {handler} must be between 0.0 and 1.0")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        dirs_to_create = [
            self.paths.get_absolute_path(self.cache.cache_dir),
            self.paths.get_absolute_path(self.cache.embedding_cache_dir),
            self.paths.get_absolute_path(self.cache.query_cache_dir),
            self.paths.get_absolute_path(self.cache.retrieval_cache_dir),
            self.paths.get_absolute_path(self.logging.log_dir),
            self.paths.get_absolute_path(self.paths.vectorstores_dir)
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_api_key(self, provider: str) -> str:
        """Get API key for specified provider."""
        if provider.lower() == "openai":
            if not self.model.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            return self.model.openai_api_key
        elif provider.lower() == "anthropic":
            if not self.model.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")  
            return self.model.anthropic_api_key
        elif provider.lower() == "google":
            if not self.model.google_api_key:
                raise ValueError("Google API key not configured")
            return self.model.google_api_key
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def get_confidence_threshold(self, handler_id: str) -> float:
        """Get confidence threshold for specific handler."""
        return self.retrieval.confidence_thresholds.get(handler_id, 0.5)
    
    def get_cache_ttl(self, cache_type: str) -> int:
        """Get cache TTL for specific type."""
        return self.performance.cache_ttl.get(cache_type, 3600)  # 1 hour default
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return os.getenv("ENVIRONMENT", "development").lower() == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config():
    """Reload configuration (useful for testing)."""
    Config._instance = None
    Config._initialized = False
    global config
    config = Config()
