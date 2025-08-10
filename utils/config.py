"""
Configuration Module: 환경변수 로드/검증 및 전역 설정

주요 기능:
1. .env 파일에서 환경변수 로드
2. 설정값 검증 및 기본값 적용
3. 라우팅용 키워드 매칭 규칙 정의
4. 핸들러별 컨피던스 임계값 및 캐시 설정

최소 수정 사항:
- .env.example과 변수명 통일
- context_manager.py 호환성 보장
- 환경변수 로드 로직 개선
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 프로젝트 루트 디렉터리 설정
ROOT_DIR = Path(__file__).parent.parent.absolute()

# .env 파일 로드 (개선된 로직)
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ 환경설정 로드: {env_path}")
else:
    # .env가 없으면 .env.example 참조 경고
    example_path = ROOT_DIR / ".env.example"
    if example_path.exists():
        print(f"⚠️ .env 파일이 없습니다. {example_path}를 {env_path}로 복사하여 설정하세요.")
        # 개발 환경에서는 .env.example을 임시로 로드
        load_dotenv(example_path)
        print(f"🔧 개발용으로 .env.example 임시 로드")
    else:
        print("❌ .env 및 .env.example 파일을 찾을 수 없습니다.")


@dataclass
class AppConfig:
    """애플리케이션 전역 설정 (context_manager.py 호환성 보장)"""
    
    # API 키
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    
    # 모델 설정 (.env.example과 통일)
    OPENAI_MODEL_MAIN: str = field(default_factory=lambda: os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL_MAIN", "gpt-4o")))
    OPENAI_MODEL_ROUTER: str = field(default_factory=lambda: os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL_ROUTER", "gpt-4o-mini")))
    
    # 임베딩 설정
    EMBEDDING_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")))
    EMBEDDING_DIMENSION: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "1536")))
    
    # 검색 설정
    FAISS_K_DEFAULT: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_K", os.getenv("FAISS_K_DEFAULT", "5"))))
    FAISS_K_EXPANDED: int = field(default_factory=lambda: int(os.getenv("FAISS_K_EXPANDED", "12")))
    CHUNK_SIZE: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    CHUNK_OVERLAP: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))
    
    # 라우팅 설정 (.env.example과 통일)
    ROUTER_CANDIDATE_SELECTION_TIMEOUT: float = field(default_factory=lambda: float(os.getenv("ROUTER_CANDIDATE_SELECTION_TIMEOUT", "0.4")))
    ROUTER_HANDLER_EXECUTION_TIMEOUT: float = field(default_factory=lambda: float(os.getenv("ROUTER_HANDLER_EXECUTION_TIMEOUT", "1.1")))
    ROUTER_TOTAL_TIMEOUT: float = field(default_factory=lambda: float(os.getenv("TIMEBOX_S", os.getenv("ROUTER_TOTAL_TIMEOUT", "1.5"))))
    
    # 컨피던스 임계값 설정 (.env.example과 통일)
    CONFIDENCE_THRESHOLD_GENERAL: float = field(default_factory=lambda: float(os.getenv("TH_GENERAL", os.getenv("CONFIDENCE_THRESHOLD_GENERAL", "0.70"))))
    CONFIDENCE_THRESHOLD_PUBLISH: float = field(default_factory=lambda: float(os.getenv("TH_PUBLISH", os.getenv("CONFIDENCE_THRESHOLD_PUBLISH", "0.74"))))
    CONFIDENCE_THRESHOLD_SATISFACTION: float = field(default_factory=lambda: float(os.getenv("TH_SATIS", os.getenv("CONFIDENCE_THRESHOLD_SATISFACTION", "0.68"))))
    CONFIDENCE_THRESHOLD_CYBER: float = field(default_factory=lambda: float(os.getenv("TH_CYBER", os.getenv("CONFIDENCE_THRESHOLD_CYBER", "0.66"))))
    CONFIDENCE_THRESHOLD_MENU: float = field(default_factory=lambda: float(os.getenv("TH_MENU", os.getenv("CONFIDENCE_THRESHOLD_MENU", "0.64"))))
    CONFIDENCE_THRESHOLD_NOTICE: float = field(default_factory=lambda: float(os.getenv("TH_NOTICE", os.getenv("CONFIDENCE_THRESHOLD_NOTICE", "0.62"))))
    CONFIDENCE_THRESHOLD_FALLBACK: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD_FALLBACK", "0.00")))
    
    # 캐시 설정 (.env.example과 통일)
    CACHE_TTL_NOTICE: int = field(default_factory=lambda: int(os.getenv("TTL_NOTICE_HOURS", "6")) * 3600)  # 시간을 초로 변환
    CACHE_TTL_MENU: int = field(default_factory=lambda: int(os.getenv("TTL_MENU_HOURS", "6")) * 3600)
    CACHE_TTL_DEFAULT: int = field(default_factory=lambda: int(os.getenv("TTL_OTHERS_DAYS", "30")) * 86400)  # 일을 초로 변환
    
    # 대화형 RAG 설정 (.env.example과 통일)
    CONVERSATION_RECENT_MESSAGES_WINDOW: int = field(default_factory=lambda: int(os.getenv("RECENT_TURNS", os.getenv("CONVERSATION_RECENT_MESSAGES_WINDOW", "6"))))
    CONVERSATION_SUMMARY_UPDATE_INTERVAL: int = field(default_factory=lambda: int(os.getenv("SUMMARY_EVERY_TURNS", os.getenv("CONVERSATION_SUMMARY_UPDATE_INTERVAL", "4"))))
    CONVERSATION_SUMMARY_TOKEN_THRESHOLD: int = field(default_factory=lambda: int(os.getenv("CONVERSATION_SUMMARY_TOKEN_THRESHOLD", "1000")))
    
    # 로깅 설정
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    LOG_FORMAT: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "json"))
    
    # 디렉터리 경로 (.env.example과 통일)
    ROOT_DIR: Path = field(default=ROOT_DIR)
    DATA_DIR: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data"))))
    VECTORSTORE_DIR: Path = field(default_factory=lambda: Path(os.getenv("VECTOR_DIR", str(ROOT_DIR / "vectorstores"))))
    CACHE_DIR: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", str(ROOT_DIR / "cache"))))
    LOGS_DIR: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", str(ROOT_DIR / "logs"))))
    SCHEMAS_DIR: Path = field(default_factory=lambda: ROOT_DIR / "schemas")
    
    # 앱 모드 (.env.example 추가)
    APP_MODE: str = field(default_factory=lambda: os.getenv("APP_MODE", "dev"))
    
    def __post_init__(self):
        """설정 검증 및 디렉터리 생성"""
        self._validate_api_keys()
        self._create_directories()
        self._setup_logging()
        self._validate_settings()
    
    def _validate_api_keys(self):
        """필수 API 키 검증 (개선된 로직)"""
        if not self.OPENAI_API_KEY:
            if self.APP_MODE == "prod":
                raise ValueError("❌ OPENAI_API_KEY는 운영환경에서 필수입니다. .env 파일에 설정해주세요.")
            else:
                print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다. 개발 모드에서는 일부 기능이 제한됩니다.")
    
    def _create_directories(self):
        """필요한 디렉터리 생성"""
        directories = [self.CACHE_DIR, self.LOGS_DIR, self.VECTORSTORE_DIR, self.DATA_DIR]
        
        for dir_path in directories:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"⚠️ 디렉터리 생성 실패: {dir_path} - {e}")
    
    def _setup_logging(self):
        """로깅 기본 설정 (개선된 로직)"""
        try:
            log_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
            
            # 기본 포맷터
            if self.LOG_FORMAT == "json":
                # JSON 형태 로깅 (운영환경)
                formatter = logging.Formatter(
                    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
                )
            else:
                # 일반 텍스트 로깅 (개발환경)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            # 루트 로거 설정
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if self.LOG_FORMAT != "json" else None,
                handlers=[logging.StreamHandler()]
            )
            
            # 서드파티 라이브러리 로그 레벨 조정
            logging.getLogger("openai").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("anthropic").setLevel(logging.WARNING)
            
        except Exception as e:
            print(f"⚠️ 로깅 설정 실패: {e}")
    
    def _validate_settings(self):
        """설정값 검증 (추가)"""
        # 컨피던스 임계값 검증
        thresholds = self.confidence_thresholds
        for handler, threshold in thresholds.items():
            if not (0.0 <= threshold <= 1.0):
                print(f"⚠️ 잘못된 컨피던스 임계값: {handler}={threshold}")
        
        # 타임아웃 검증
        if self.ROUTER_TOTAL_TIMEOUT <= 0:
            print(f"⚠️ 잘못된 타임아웃 설정: {self.ROUTER_TOTAL_TIMEOUT}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        속성을 딕셔너리처럼 접근할 수 있는 메서드
        예: config.get('OPENAI_API_KEY')
        """
        return getattr(self, key, default)

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


# ================================================================
# 라우팅용 키워드 매칭 규칙 (기존 유지)
# ================================================================

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
        '담당부서': 0.85,
        '책임자': 0.80,
        '055-254': 0.95,  # 기관 대표번호
        
        # 업무 관련
        '업무': 0.75,
        '처리': 0.70,
        '절차': 0.80,
        '방법': 0.65,
        '과정': 0.60,  # 교육과정과 혼동 방지
        
        # 시설 관련
        '시설': 0.80,
        '장소': 0.70,
        '위치': 0.75,
        '건물': 0.75
    },
    
    'satisfaction': {
        # 만족도 관련
        '만족도': 0.95,
        '만족': 0.85,
        '평가': 0.90,
        '점수': 0.85,
        '성적': 0.80,
        '결과': 0.75,
        
        # 교육과정/교과목
        '교육과정': 0.85,
        '과정': 0.80,
        '교과목': 0.85,
        '과목': 0.80,
        '강의': 0.80,
        '수업': 0.75,
        
        # 평가 지표
        '전반만족도': 0.90,
        '역량향상도': 0.90,
        '현업적용도': 0.90,
        '교과편성': 0.85,
        '강의만족도': 0.90,
        
        # 순위/랭킹
        '순위': 0.80,
        '랭킹': 0.80,
        '등급': 0.75,
        '위': 0.70,
        
        # 통계
        '통계': 0.75,
        '분석': 0.70,
        '데이터': 0.70
    },
    
    'cyber': {
        # 사이버교육 관련
        '사이버': 0.95,
        '온라인': 0.90,
        '인터넷': 0.85,
        '웹': 0.80,
        '디지털': 0.75,
        
        # 플랫폼
        '민간위탁': 0.95,
        '위탁': 0.85,
        '민간': 0.80,
        '나라배움터': 0.95,
        'ncs': 0.80,
        '국가직무능력표준': 0.85,
        
        # 교육 분류
        '직무': 0.80,
        '소양': 0.80,
        '시책': 0.80,
        'Gov-MOOC': 0.90,
        '무크': 0.85,
        
        # 학습 관련
        '학습': 0.75,
        '수강': 0.80,
        '차시': 0.85,
        '인정시간': 0.85,
        '이수': 0.80
    },
    
    'publish': {
        # 공식 발행물
        '계획서': 0.95,
        '평가서': 0.95,
        '보고서': 0.85,
        '교육훈련계획': 0.95,
        '종합평가': 0.95,
        
        # 연도별
        '2024': 0.80,
        '2025': 0.90,
        '작년': 0.75,
        '올해': 0.80,
        '내년': 0.75,
        
        # 계획/성과
        '계획': 0.80,
        '성과': 0.85,
        '실적': 0.85,
        '목표': 0.80,
        '방향': 0.75,
        
        # 정책
        '정책': 0.80,
        '방침': 0.75,
        '전략': 0.80,
        '비전': 0.75
    },
    
    'menu': {
        # 식단 관련
        '식단': 0.95,
        '메뉴': 0.95,
        '식사': 0.85,
        '급식': 0.85,
        '식당': 0.90,
        '구내식당': 0.95,
        
        # 식사 시간
        '조식': 0.90,
        '중식': 0.90,
        '석식': 0.90,
        '아침': 0.85,
        '점심': 0.90,
        '저녁': 0.85,
        
        # 요일
        '월요일': 0.80,
        '화요일': 0.80,
        '수요일': 0.80,
        '목요일': 0.80,
        '금요일': 0.80,
        '주간': 0.75,
        
        # 기타
        '운영': 0.70,
        '시간': 0.65,
        '가격': 0.75
    },
    
    'notice': {
        # 공지 관련
        '공지': 0.95,
        '안내': 0.90,
        '알림': 0.85,
        '소식': 0.80,
        '뉴스': 0.70,
        
        # 업데이트
        '업데이트': 0.80,
        '변경': 0.80,
        '수정': 0.75,
        '개정': 0.75,
        '갱신': 0.70,
        
        # 시간 관련
        '최근': 0.85,
        '새로운': 0.80,
        '오늘': 0.80,
        '이번주': 0.75,
        '이번달': 0.75,
        
        # 중요도
        '중요': 0.85,
        '긴급': 0.90,
        '필수': 0.85,
        '주의': 0.80,
        
        # 기타
        '발표': 0.75,
        '소개': 0.70,
        '홍보': 0.70
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
    '시간',  # 너무 일반적
    '방법',  # 너무 일반적
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
        '구내식당': ['식당', '급식소', '카페테리아']
    },
    'general': {
        '연락처': ['전화번호', '담당자', '담당부서'],
        '규정': ['학칙', '전결규정', '운영원칙', '내규']
    }
}


# ================================================================
# 싱글톤 및 유틸리티 함수들
# ================================================================

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


# ================================================================
# 개발/디버그용 함수들
# ================================================================

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
        print("❌ 키워드 규칙 검증 실패:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ 키워드 규칙 검증 통과")
    
    return len(issues) == 0


def print_config_summary():
    """설정 요약 출력"""
    config = get_config()
    
    print("\n🔧 Byeoli Talk at GNH 설정 요약")
    print("=" * 50)
    print(f"📁 프로젝트 루트: {config.ROOT_DIR}")
    print(f"🔧 앱 모드: {config.APP_MODE}")
    print(f"📝 로그 레벨: {config.LOG_LEVEL}")
    print(f"🤖 메인 모델: {config.OPENAI_MODEL_MAIN}")
    print(f"🔄 라우터 모델: {
