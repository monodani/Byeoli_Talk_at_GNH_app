#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - logging_utils.py

구조화된 로깅 유틸리티:
- JSON 형식 로깅 지원
- 성능 측정용 타이머 컨텍스트 매니저
- 민감정보 마스킹 기능
- context_manager.py 호환성 보장

핵심 기능:
- get_logger(): 표준 로거 반환
- log_timer(): 성능 측정 컨텍스트 매니저
- 구조화된 JSON 로깅 (선택사항)
- 민감정보 자동 마스킹
"""

import logging
import time
import json
import re
from contextlib import contextmanager
from typing import Dict, Any, Optional
from datetime import datetime

# 민감정보 패턴 (API 키, 개인정보 등)
SENSITIVE_PATTERNS = [
    (r'(api[_-]?key["\s]*[:=]["\s]*)[^"\s]+', r'\1***'),
    (r'(password["\s]*[:=]["\s]*)[^"\s]+', r'\1***'),
    (r'(token["\s]*[:=]["\s]*)[^"\s]+', r'\1***'),
    (r'(\d{3}-\d{4}-\d{4})', r'***-****-****'),  # 전화번호
    (r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', r'***@***.***'),  # 이메일
]


def mask_sensitive_info(text: str) -> str:
    """민감정보 마스킹"""
    if not isinstance(text, str):
        return text
    
    masked_text = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        masked_text = re.sub(pattern, replacement, masked_text, flags=re.IGNORECASE)
    
    return masked_text


def get_logger(name: str) -> logging.Logger:
    """
    표준 로거 반환 (context_manager.py 호환)
    
    Args:
        name: 로거 이름 (일반적으로 __name__)
    
    Returns:
        logging.Logger: 설정된 로거 인스턴스
    """
    logger = logging.getLogger(name)
    
    # 로거가 아직 설정되지 않았다면 기본 설정 적용
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


@contextmanager
def log_timer(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    성능 측정용 타이머 컨텍스트 매니저 (context_manager.py 호환)
    
    Args:
        operation_name: 작업 이름
        logger: 로거 인스턴스 (없으면 기본 로거 사용)
    
    Usage:
        with log_timer("데이터 로드"):
            # 시간 측정할 코드
            load_data()
    """
    if logger is None:
        logger = get_logger(__name__)
    
    start_time = time.time()
    start_perf = time.perf_counter()
    
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        elapsed_perf = time.perf_counter() - start_perf
        elapsed_ms = elapsed_perf * 1000
        
        logger.debug(f"{operation_name}: {elapsed_ms:.1f}ms ({elapsed_time:.3f}s)")


class StructuredLogger:
    """
    구조화된 JSON 로깅을 위한 고급 로거
    (향후 ELK 스택 연동 시 사용)
    """
    
    def __init__(self, name: str, mask_sensitive: bool = True):
        self.logger = get_logger(name)
        self.mask_sensitive = mask_sensitive
    
    def log_structured(self, level: str, message: str, **kwargs):
        """구조화된 로그 출력"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level.upper(),
            "message": message,
            "module": self.logger.name,
            **kwargs
        }
        
        # 민감정보 마스킹
        if self.mask_sensitive:
            log_data = self._mask_log_data(log_data)
        
        json_log = json.dumps(log_data, ensure_ascii=False, separators=(',', ':'))
        
        # 레벨에 따른 로깅
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(json_log)
    
    def _mask_log_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """로그 데이터 내 민감정보 마스킹"""
        if isinstance(data, dict):
            return {k: self._mask_log_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._mask_log_data(item) for item in data]
        elif isinstance(data, str):
            return mask_sensitive_info(data)
        else:
            return data
    
    def info(self, message: str, **kwargs):
        """INFO 레벨 구조화 로깅"""
        self.log_structured("info", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """ERROR 레벨 구조화 로깅"""
        self.log_structured("error", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """DEBUG 레벨 구조화 로깅"""
        self.log_structured("debug", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """WARNING 레벨 구조화 로깅"""
        self.log_structured("warning", message, **kwargs)


@contextmanager
def performance_timer(operation: str, 
                     logger: Optional[logging.Logger] = None,
                     threshold_ms: float = 1000.0):
    """
    성능 임계값 기반 타이머
    임계값을 초과하면 WARNING, 그렇지 않으면 DEBUG 레벨로 로깅
    
    Args:
        operation: 작업 이름
        logger: 로거 인스턴스
        threshold_ms: 경고 임계값 (밀리초)
    """
    if logger is None:
        logger = get_logger(__name__)
    
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        if elapsed_ms > threshold_ms:
            logger.warning(f"🐌 {operation}: {elapsed_ms:.1f}ms (임계값 {threshold_ms}ms 초과)")
        else:
            logger.debug(f"⚡ {operation}: {elapsed_ms:.1f}ms")


def setup_logging(level: str = "INFO", 
                 format_style: str = "standard",
                 enable_json: bool = False):
    """
    애플리케이션 전역 로깅 설정
    
    Args:
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
        format_style: 포맷 스타일 (standard, detailed)
        enable_json: JSON 형식 로깅 활성화
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # 기존 핸들러 제거
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 새 핸들러 설정
    handler = logging.StreamHandler()
    
    if enable_json:
        # JSON 포맷터 (향후 ELK 연동용)
        formatter = logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","message":"%(message)s"}',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
    elif format_style == "detailed":
        # 상세 포맷터
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        # 표준 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    
    # 외부 라이브러리 로깅 레벨 조정
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    get_logger(__name__).info(f"✅ 로깅 시스템 초기화 완료 (레벨: {level}, 스타일: {format_style})")


# ================================================================
# 테스트 및 예제 코드
# ================================================================

def test_logging_utils():
    """logging_utils 기능 테스트"""
    print("🧪 logging_utils 테스트 시작")
    
    # 1. 기본 로거 테스트
    logger = get_logger("test_module")
    logger.info("기본 로거 테스트")
    
    # 2. 타이머 테스트
    with log_timer("테스트 작업", logger):
        time.sleep(0.1)  # 100ms 지연
    
    # 3. 성능 타이머 테스트
    with performance_timer("성능 테스트", logger, threshold_ms=50.0):
        time.sleep(0.02)  # 20ms 지연 (임계값 미만)
    
    with performance_timer("느린 작업", logger, threshold_ms=50.0):
        time.sleep(0.08)  # 80ms 지연 (임계값 초과)
    
    # 4. 구조화된 로거 테스트
    struct_logger = StructuredLogger("structured_test")
    struct_logger.info("구조화된 로그 테스트", 
                      user_id="test123", 
                      operation="login",
                      api_key="sk-1234567890abcdef")  # 마스킹 테스트
    
    # 5. 민감정보 마스킹 테스트
    sensitive_text = "API Key: sk-1234567890, 연락처: 010-1234-5678, 이메일: test@example.com"
    masked_text = mask_sensitive_info(sensitive_text)
    logger.info(f"마스킹 테스트 - 원본: {sensitive_text}")
    logger.info(f"마스킹 테스트 - 결과: {masked_text}")
    
    print("✅ logging_utils 테스트 완료")


if __name__ == "__main__":
    """개발/테스트용 메인 함수"""
    # 로깅 시스템 초기화
    setup_logging(level="DEBUG", format_style="detailed")
    
    # 테스트 실행
    test_logging_utils()
    
    print("\n🎉 logging_utils.py 구현 및 테스트 완료!")
    print("이제 context_manager.py와 다른 모듈들이 정상 작동할 수 있습니다.")
