"""
Configuration Module: 환경변수 로드/검증 및 전역 설정

주요 기능:
1. .env 파일에서 환경변수 로드
2. 설정값 검증 및 기본값 적용
3. 라우팅용 키워드 매칭 규칙 정의
4. 핸들러별 컨피던스 임계값 및 캐시 설정
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 프로젝트 루트 디렉터리 설정
ROOT_DIR = Path(__file__).parent.parent.absolute()

# .env 파일 로드
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # .env가 없으면 .env.example 참조 경고
    example_path = ROOT_DIR / ".env.example"
    if example_path.exists():
        print(f"Warning: .env not found. Please copy {example_path} to {env_path} and configure it.")


@dataclass
class AppConfig:
    """애플리케이션 전역 설정"""
    
    # API 키
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    
    # 모델 설정
    OPENAI_MODEL_MAIN: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL_MAIN", "gpt-4o"))
    OPENAI_MODEL_ROUTER: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL_ROUTER", "gpt-4o-mini"))
    
    # 임베딩 설정
    EMBEDDING_MODEL: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    EMBEDDING_DIMENSION: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "1536")))
    
    # 검색 설정
    FAISS_K_DEFAULT: int = field(default_factory=lambda: int(os.getenv("FAISS_K_DEFAULT", "5")))
    FAISS_K_EXPANDED: int = field(default_factory=lambda: int(os.getenv("FAISS_K_EXPANDED", "12")))
    CHUNK_SIZE: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    CHUNK_OVERLAP: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))
    
    # 라우팅 설정
    ROUTER_CANDIDATE_SELECTION_TIMEOUT: float = field(default_factory=lambda: float(os.getenv("ROUTER_CANDIDATE_SELECTION_TIMEOUT", "0.4")))
    ROUTER_HANDLER_EXECUTION_TIMEOUT: float = field(default_factory=lambda: float(os.getenv("ROUTER_HANDLER_EXECUTION_TIMEOUT", "1.1")))
    ROUTER_TOTAL_TIMEOUT: float = field(default_factory=lambda: float(os.getenv("ROUTER_TOTAL_TIMEOUT", "1.5")))
    
    # 컨피던스 임계값 설정
    CONFIDENCE_THRESHOLD_GENERAL: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD_GENERAL", "0.70")))
    CONFIDENCE_THRESHOLD_PUBLISH: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD_PUBLISH", "0.74")))
    CONFIDENCE_THRESHOLD_SATISFACTION: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD_SATISFACTION", "0.68")))
    CONFIDENCE_THRESHOLD_CYBER: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD_CYBER", "0.66")))
    CONFIDENCE_THRESHOLD_MENU: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD_MENU", "0.64")))
    CONFIDENCE_THRESHOLD_NOTICE: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD_NOTICE", "0.62")))
    CONFIDENCE_THRESHOLD_FALLBACK: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD_FALLBACK", "0.00")))
    
    # 캐시 설정 (초 단위)
    CACHE_TTL_NOTICE: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_NOTICE", "21600")))  # 6시간
    CACHE_TTL_MENU: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_MENU", "21600")))     # 6시간
    CACHE_TTL_DEFAULT: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_DEFAULT", "2592000")))  # 30일
    
    # 대화형 RAG 설정
    CONVERSATION_RECENT_MESSAGES_WINDOW: int = field(default_factory=lambda: int(os.getenv("CONVERSATION_RECENT_MESSAGES_WINDOW", "6")))
    CONVERSATION_SUMMARY_UPDATE_INTERVAL: int = field(default_factory=lambda: int(os.getenv("CONVERSATION_SUMMARY_UPDATE_INTERVAL", "4")))
    CONVERSATION_SUMMARY_TOKEN_THRESHOLD: int = field(default_factory=lambda: int(os.getenv("CONVERSATION_SUMMARY_TOKEN_THRESHOLD", "1000")))
    
    # 로깅 설정
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    LOG_FORMAT: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "json"))
    
    # 디렉터리 경로
    ROOT_DIR: Path = field(default=ROOT_DIR)
    DATA_DIR: Path = field(default_factory=lambda: ROOT_DIR / "data")
    VECTORSTORE_DIR: Path = field(default_factory=lambda: ROOT_DIR / "vectorstores")
    CACHE_DIR: Path = field(default_factory=lambda: ROOT_DIR / "cache")
    LOGS_DIR: Path = field(default_factory=lambda: ROOT_DIR / "logs")
    SCHEMAS_DIR: Path = field(default_factory=lambda: ROOT_DIR / "schemas")
    
    def __post_init__(self):
        """설정 검증 및 디렉터리 생성"""
        self._validate_api_keys()
        self._create_directories()
        self._setup_logging()
    
    def _validate_api_keys(self):
        """필수 API 키 검증"""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in .env file.")
    
    def _create_directories(self):
        """필요한 디렉터리 생성"""
        for dir_path in [self.CACHE_DIR, self.LOGS_DIR, self.VECTORSTORE_DIR]:
            dir_path.mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """로깅 기본 설정"""
        log_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @property
    def confidence_thresholds(self) -> Dict[str, float]:
        """핸들러별 컨피던스 임계값 딕셔너리"""
        return {
            'general': self.CONFIDENCE_THRESHOLD_GENERAL,
            'publish': self.CONFIDENCE_THRESHOLD_PUBLISH,
            'satisfaction': self.CONFIDENCE_THRESHOLD_SATISFACTION,
            'cyber': self.CONFIDENCE_THRESHOLD_CYBER,
            'menu': self.CONFIDENCE_THRESHOLD_MENU,
            'notice': self.CONFIDENCE_THRESHOLD_NOTICE,
            'fallback': self.CONFIDENCE_THRESHOLD_FALLBACK
        }
    
    @property
    def cache_ttl_config(self) -> Dict[str, int]:
        """핸들러별 캐시 TTL 설정"""
        return {
            'notice': self.CACHE_TTL_NOTICE,
            'menu': self.CACHE_TTL_MENU,
            'general': self.CACHE_TTL_DEFAULT,
            'publish': self.CACHE_TTL_DEFAULT,
            'satisfaction': self.CACHE_TTL_DEFAULT,
            'cyber': self.CACHE_TTL_DEFAULT,
            'fallback': self.CACHE_TTL_DEFAULT
        }


# 라우팅용 키워드 매칭 규칙
KEYWORD_MATCHING_RULES: Dict[str, Dict[str, float]] = {
    'general': {
        # 규정 및 학칙
        '학칙': 0.95,
        '규정': 0.90,
        '전결': 0.85,
        '전결규정': 0.90,
        '운영원칙': 0.85,
        '운영규칙': 0.80,
        '내규': 0.75,
        
        # 담당자 및 연락처
        '연락처': 0.90,
        '전화번호': 0.90,
        '담당자': 0.85,
        '업무담당': 0.85,
        '부서': 0.75,
        '조직': 0.70,
        
        # 일반 업무
        '업무': 0.70,
        '사무': 0.65,
        '절차': 0.70,
        '과정': 0.60,  # satisfaction과 겹칠 수 있어 낮게 설정
        
        # 평가 관련 (operation_test.pdf)
        '운영평가': 0.85,
        '평가계획': 0.80,
        '시험': 0.70,
        
        # 기타
        '문의': 0.65,
        '안내': 0.60,
        '정보': 0.55
    },
    
    'publish': {
        # 2025 교육훈련계획서
        '2025계획': 0.98,
        '2025년계획': 0.98,
        '교육훈련계획': 0.95,
        '계획서': 0.90,
        '훈련계획': 0.85,
        
        # 2024 종합평가서
        '2024평가': 0.98,
        '2024년평가': 0.98,
        '종합평가': 0.95,
        '평가서': 0.90,
        '성과평가': 0.85,
        
        # 발행물 일반
        '발행물': 0.85,
        '공식문서': 0.80,
        '보고서': 0.75,
        '백서': 0.75,
        '연보': 0.75,
        
        # 계획 관련
        '계획': 0.70,
        '방침': 0.70,
        '정책': 0.75,
        '전략': 0.70,
        
        # 평가 관련
        '평가': 0.65,  # satisfaction과 구분을 위해 낮게
        '성과': 0.70,
        '결과': 0.60
    },
    
    'satisfaction': {
        # 만족도 조사
        '만족도': 0.98,
        '만족도조사': 0.98,
        '수강만족도': 0.95,
        '교육만족도': 0.95,
        
        # 교육과정 만족도
        '교육과정만족도': 0.98,
        '과정만족도': 0.95,
        '코스만족도': 0.90,
        
        # 교과목 만족도
        '교과목만족도': 0.98,
        '과목만족도': 0.95,
        '강의만족도': 0.90,
        '수업만족도': 0.90,
        
        # 평가 관련
        '교육평가': 0.85,
        '강의평가': 0.85,
        '수강평가': 0.80,
        '평가점수': 0.80,
        '점수': 0.75,
        '성적': 0.70,
        
        # 수강생 관련
        '수강생': 0.75,
        '교육생': 0.75,
        '참여자': 0.70,
        '학습자': 0.70,
        
        # 교육 관련
        '교육과정': 0.70,
        '교과목': 0.70,
        '강의': 0.65,
        '수업': 0.65,
        '강좌': 0.65,
        
        # 조사 관련
        '설문': 0.75,
        '조사': 0.70,
        '피드백': 0.70,
        '의견': 0.65
    },
    
    'cyber': {
        # 사이버 교육
        '사이버교육': 0.98,
        '사이버': 0.95,
        '온라인교육': 0.95,
        '온라인': 0.90,
        '인터넷교육': 0.85,
        '웹교육': 0.80,
        '이러닝': 0.90,
        'e-learning': 0.90,
        
        # 민간위탁
        '민간위탁': 0.95,
        '민간사이버': 0.95,
        '위탁교육': 0.85,
        '민간교육': 0.80,
        
        # 나라배움터
        '나라배움터': 0.98,
        'ncs': 0.85,
        '국가직무능력표준': 0.85,
        
        # 교육일정
        '교육일정': 0.90,
        '수강일정': 0.85,
        '개설일정': 0.85,
        '일정': 0.70,
        '스케줄': 0.75,
        
        # 수강 관련
        '수강신청': 0.90,
        '신청': 0.70,
        '등록': 0.70,
        '수강': 0.75,
        
        # 기타
        '원격교육': 0.85,
        '디지털교육': 0.80,
        '비대면교육': 0.80
    },
    
    'menu': {
        # 식단 관련
        '식단': 0.98,
        '식단표': 0.98,
        '메뉴': 0.95,
        '메뉴표': 0.95,
        '주간식단': 0.90,
        
        # 식당 관련
        '구내식당': 0.95,
        '식당': 0.90,
        '급식': 0.85,
        '급식소': 0.85,
        '식당운영': 0.80,
        
        # 식사 관련
        '점심': 0.85,
        '중식': 0.85,
        '식사': 0.80,
        '밥': 0.75,
        '음식': 0.75,
        '먹을것': 0.70,
        
        # 요일별
        '월요일': 0.70,
        '화요일': 0.70,
        '수요일': 0.70,
        '목요일': 0.70,
        '금요일': 0.70,
        '오늘': 0.75,
        '내일': 0.75,
        
        # 기타
        '주차별': 0.70,
        '이번주': 0.75,
        '다음주': 0.75
    },
    
    'notice': {
        # 공지사항
        '공지': 0.95,
        '공지사항': 0.98,
        '공고': 0.90,
        '안내': 0.85,
        '안내사항': 0.90,
        '알림': 0.90,
        '소식': 0.80,
        
        # 최신/새로운
        '최신': 0.85,
        '새로운': 0.80,
        '신규': 0.80,
        '업데이트': 0.75,
        '변경': 0.75,
        '수정': 0.70,
        
        # 시간 관련
        '오늘': 0.75,
        '최근': 0.80,
        '이번달': 0.70,
        '이번주': 0.70,
        
        # 중요도
        '중요': 0.80,
        '긴급': 0.85,
        '필수': 0.80,
        '주의': 0.75,
        
        # 기타
        '뉴스': 0.70,
        '소개': 0.65,
        '발표': 0.70
    }
}

# 키워드 매칭에서 제외할 불용어 (false positive 방지)
KEYWORD_STOP_WORDS = {
    '교육',  # 너무 일반적
    '과정',  # 모든 핸들러에서 사용 가능
    '정보',  # 너무 일반적
    '문서',  # 너무 일반적
    '자료',  # 너무 일반적
    '내용',  # 너무 일반적
}

# 핸들러별 동의어 그룹 (확장 검색용)
HANDLER_SYNONYMS = {
    'satisfaction': {
        '만족도': ['만족', '평가', '점수', '성적'],
        '교육과정': ['과정', '코스', '프로그램'],
        '교과목': ['과목', '강의', '수업', '강좌']
    },
    'cyber': {
        '사이버': ['온라인', '인터넷', '웹', '디지털'],
        '민간위탁': ['위탁', '민간'],
        '나라배움터': ['ncs', '국가직무능력표준']
    },
    'menu': {
        '식단': ['메뉴', '식사', '급식'],
        '구내식당': ['식당', '급식소']
    }
}


# 전역 설정 인스턴스
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """전역 설정 인스턴스 반환 (싱글톤)"""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def get_keyword_rules() -> Dict[str, Dict[str, float]]:
    """키워드 매칭 규칙 반환"""
    return KEYWORD_MATCHING_RULES.copy()


def get_handler_synonyms() -> Dict[str, Dict[str, list]]:
    """핸들러별 동의어 그룹 반환"""
    return HANDLER_SYNONYMS.copy()


def get_stop_words() -> set:
    """키워드 불용어 세트 반환"""
    return KEYWORD_STOP_WORDS.copy()


# 개발/디버그용 함수
def validate_keyword_rules():
    """키워드 매칭 규칙 검증"""
    issues = []
    
    for handler_id, keywords in KEYWORD_MATCHING_RULES.items():
        for keyword, score in keywords.items():
            if not (0.0 <= score <= 1.0):
                issues.append(f"{handler_id}.{keyword}: score {score} out of range [0.0, 1.0]")
            
            if keyword in KEYWORD_STOP_WORDS:
                issues.append(f"{handler_id}.{keyword}: keyword is in stop words")
    
    if issues:
        print("Keyword rules validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Keyword rules validation passed")
    
    return len(issues) == 0


if __name__ == "__main__":
    # 설정 테스트 및 검증
    config = get_config()
    print(f"Config loaded from: {config.ROOT_DIR}")
    print(f"Confidence thresholds: {config.confidence_thresholds}")
    print(f"Cache TTL config: {config.cache_ttl_config}")
    
    print("\nValidating keyword rules...")
    validate_keyword_rules()
    
    print(f"\nTotal keywords by handler:")
    for handler_id, keywords in KEYWORD_MATCHING_RULES.items():
        print(f"  {handler_id}: {len(keywords)} keywords")
