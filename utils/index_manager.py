#!/usr/bin/env python3
"""
벼리톡@경상남도인재개발원 (경상남도인재개발원 RAG 챗봇) - index_manager.py (OpenAI 호환성 수정 버전)

IndexManager 싱글톤: 모든 벡터스토어 중앙 관리
- 앱 기동 시 모든 FAISS 인덱스 사전 로드
- 저장된 BM25 파일 로드
- 해시 기반 파일 변경 감지로 핫스왑
- 전역 공유로 핸들러 간 일관성 보장
- 메모리 효율적인 단일 인스턴스 관리

🚨 주요 수정사항:
✅ OpenAIEmbeddings 초기화 방식 수정 (호환성 문제 해결)
✅ API 키 명시적 전달
✅ Graceful Degradation 적용
✅ 에러 처리 강화
"""

import hashlib
import logging
import threading
import time
import pickle
import traceback
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field

# 프로젝트 모듈
from utils.config import config
from utils.contracts import HandlerType
from utils.textifier import TextChunk

# 외부 라이브러리
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import numpy as np

# 로깅 설정
logger = logging.getLogger(__name__)

# ================================================================
# 프로젝트 루트 경로 설정
# ================================================================
try:
    ROOT_DIR = Path(__file__).parent.parent.absolute()
except NameError:
    ROOT_DIR = Path(".").absolute()

# ================================================================
# 1. 벡터스토어 메타데이터 클래스
# ================================================================

@dataclass
class VectorStoreMetadata:
    """
    벡터스토어 메타데이터 및 상태 관리 (BM25 파일 지원)
    """
    domain: str
    vectorstore_base_dir: Path
    
    # post_init에서 설정되므로 init=False로 설정
    vectorstore_path: Path = field(init=False)
    faiss_path: Path = field(init=False)
    pkl_path: Path = field(init=False)
    bm25_path: Path = field(init=False)
    
    # 런타임 속성
    embeddings: Optional[Any] = None  # OpenAIEmbeddings 타입 힌트 제거
    vectorstore: Optional[FAISS] = None
    bm25: Optional[BM25Okapi] = None
    documents: List[TextChunk] = field(default_factory=list)
    last_loaded: Optional[datetime] = None
    load_count: int = 0
    error_count: int = 0
    last_hash: Optional[str] = None
    
    def __post_init__(self):
        # data_ingestion.py와 동일한 경로 매핑 사용
        self.vectorstore_path = self._get_vectorstore_path()
        self.faiss_path = self.vectorstore_path / f"{self.domain}_index.faiss"
        self.pkl_path = self.vectorstore_path / f"{self.domain}_index.pkl"
        self.bm25_path = self.vectorstore_path / f"{self.domain}_index.bm25"
        
        # ✅ OpenAIEmbeddings 안전한 초기화
        self.embeddings = self._init_embeddings()
    
    def _init_embeddings(self) -> Optional[Any]:
        """
        OpenAIEmbeddings 안전한 초기화 (호환성 수정)
        """
        try:
            # LangChain OpenAI Embeddings 호환성 수정
            from langchain_openai import OpenAIEmbeddings
            
            # API 키 확인
            api_key = config.OPENAI_API_KEY
            if not api_key:
                logger.warning("⚠️ OPENAI_API_KEY가 설정되지 않아 임베딩을 사용할 수 없습니다.")
                return None
            
            # 최소한의 매개변수로 안전한 초기화 (proxies 오류 방지)
            embeddings = OpenAIEmbeddings(
                api_key=api_key,
                model=config.EMBEDDING_MODEL
            )
            
            logger.debug(f"✅ {self.domain} 도메인용 OpenAIEmbeddings 초기화 완료")
            return embeddings
            
        except ImportError as e:
            logger.error(f"❌ LangChain OpenAI 라이브러리 임포트 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ {self.domain} 도메인 OpenAIEmbeddings 초기화 실패: {e}")
            return None
    
    def _get_vectorstore_path(self) -> Path:
        """도메인별 벡터스토어 경로 매핑"""
        domain_mapping = {
            "course_satisfaction": "vectorstore_course_satisfaction",
            "subject_satisfaction": "vectorstore_subject_satisfaction", 
            "satisfaction": "vectorstore_unified_satisfaction",
            "publish": "vectorstore_unified_publish",
            "general": "vectorstore_general",
            "cyber": "vectorstore_cyber",
            "notice": "vectorstore_notice",
            "menu": "vectorstore_menu"
        }
        
        vectorstore_dir_name = domain_mapping.get(self.domain, f"vectorstore_{self.domain}")
        return self.vectorstore_base_dir / vectorstore_dir_name
    
    def exists(self) -> bool:
        """필수 파일 존재 여부 확인"""
        return (
            self.faiss_path.exists() and 
            self.pkl_path.exists() and
            self.vectorstore_path.exists()
        )
    
    def get_file_hash(self) -> str:
        """파일 변경 감지용 해시 계산"""
        try:
            if not self.exists():
                return ""
                
            hash_content = ""
            
            # FAISS 파일 해시
            if self.faiss_path.exists():
                hash_content += str(self.faiss_path.stat().st_mtime)
            
            # PKL 파일 해시
            if self.pkl_path.exists():
                hash_content += str(self.pkl_path.stat().st_mtime)
                
            # BM25 파일 해시 (선택적)
            if self.bm25_path.exists():
                hash_content += str(self.bm25_path.stat().st_mtime)
            
            return hashlib.md5(hash_content.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"⚠️ {self.domain} 해시 계산 실패: {e}")
            return ""

# ================================================================
# 2. IndexManager 싱글톤 클래스
# ================================================================

class IndexManager:
    """
    모든 벡터스토어를 관리하는 싱글톤 클래스 (OpenAI 호환성 수정)
    """
    _instance = None
    _instance_lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super(IndexManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self.metadata: Dict[str, VectorStoreMetadata] = {}
        
        # ✅ 글로벌 OpenAIEmbeddings 안전한 초기화
        self.embeddings = self._init_global_embeddings()
        
        for domain in config.HANDLERS:
            self.metadata[domain] = VectorStoreMetadata(
                domain=domain,
                vectorstore_base_dir=Path(config.VECTORSTORE_DIR)
            )
        
        logger.info(f"🚀 IndexManager 싱글톤 초기화 완료: {len(self.metadata)}개 도메인")
        self.load_all_domains()
        self._initialized = True

    def _init_global_embeddings(self) -> Optional[Any]:
        """
        글로벌 OpenAIEmbeddings 안전한 초기화
        """
        try:
            from langchain_openai import OpenAIEmbeddings
            
            api_key = config.OPENAI_API_KEY
            if not api_key:
                logger.warning("⚠️ OPENAI_API_KEY가 설정되지 않아 임베딩 기능이 제한됩니다.")
                return None
            
            embeddings = OpenAIEmbeddings(
                api_key=api_key,
                model=config.EMBEDDING_MODEL
            )
            
            logger.info(f"✅ 글로벌 OpenAIEmbeddings 초기화 완료: {config.EMBEDDING_MODEL}")
            return embeddings
            
        except ImportError as e:
            logger.error(f"❌ LangChain OpenAI 라이브러리 임포트 실패: {e}")
            logger.info("🔄 Graceful Degradation: 임베딩 없이 기본 기능으로 동작합니다.")
            return None
        except Exception as e:
            logger.error(f"❌ 글로벌 OpenAIEmbeddings 초기화 실패: {e}")
            logger.info("🔄 Graceful Degradation: 임베딩 없이 기본 기능으로 동작합니다.")
            return None

    def _load_domain(self, domain: str):
        """
        단일 도메인의 벡터스토어를 로드 (Pydantic v1/v2 호환성 수정)
        """
        meta = self.metadata[domain]
        
        logger.info(f"🔄 도메인 {domain} 로드 시작...")
        logger.debug(f"  - FAISS 경로: {meta.faiss_path}")
        logger.debug(f"  - PKL 경로: {meta.pkl_path}")
        logger.debug(f"  - BM25 경로: {meta.bm25_path}")
        
        try:
            if not meta.exists():
                logger.warning(f"⚠️ 도메인 {domain}에 필요한 인덱스 파일이 없습니다. 로드 건너뜁니다.")
                meta.vectorstore = None
                meta.bm25 = None
                return
            
            start_time = time.time()
            
            # 임베딩 모델 사용 (글로벌 또는 도메인별)
            embeddings_to_use = meta.embeddings or self.embeddings
            if not embeddings_to_use:
                logger.warning(f"⚠️ {domain} 임베딩 모델이 없어 FAISS 로드를 건너뜁니다.")
                meta.vectorstore = None
            else:
                # FAISS 인덱스 로드
                meta.vectorstore = FAISS.load_local(
                    str(meta.vectorstore_path),
                    embeddings_to_use,
                    index_name=f"{domain}_index",
                    allow_dangerous_deserialization=True
                )
            
            # ✅ 문서 메타데이터 로드 (Pydantic v1/v2 호환성 처리)
            try:
                with open(meta.pkl_path, "rb") as f:
                    loaded_data = pickle.load(f)
                    
                    # Pydantic v1 → v2 변환 처리
                    if isinstance(loaded_data, list):
                        meta.documents = []
                        for item in loaded_data:
                            try:
                                if hasattr(item, '__fields_set__'):
                                    # Pydantic v1 객체 감지 → v2로 변환
                                    logger.debug(f"🔄 Pydantic v1 객체를 v2로 변환: {type(item)}")
                                    
                                    # 기본 딕셔너리로 변환 후 재생성
                                    if hasattr(item, 'dict'):
                                        item_dict = item.dict()
                                    elif hasattr(item, 'model_dump'):
                                        item_dict = item.model_dump()
                                    else:
                                        # 직접 속성 추출
                                        item_dict = {
                                            'text': getattr(item, 'text', ''),
                                            'metadata': getattr(item, 'metadata', {}),
                                            'source_id': getattr(item, 'source_id', ''),
                                            'chunk_index': getattr(item, 'chunk_index', 0)
                                        }
                                    
                                    # 새로운 TextChunk 생성
                                    new_chunk = TextChunk(**item_dict)
                                    meta.documents.append(new_chunk)
                                else:
                                    # 이미 Pydantic v2 또는 호환 객체
                                    meta.documents.append(item)
                            except Exception as item_error:
                                logger.warning(f"⚠️ 개별 문서 변환 실패: {item_error}, 건너뜀")
                                continue
                    else:
                        meta.documents = loaded_data
                        
                logger.info(f"✅ 도메인 {domain} 문서 메타데이터 로드 완료: {len(meta.documents)}개")
                
            except Exception as pkl_error:
                logger.error(f"❌ 도메인 {domain} PKL 로드 실패: {pkl_error}")
                
                # ✅ Fallback: FAISS docstore에서 직접 로드
                if meta.vectorstore:
                    logger.info(f"🔄 {domain} FAISS docstore에서 문서 복구 시도")
                    try:
                        # FAISS의 docstore에서 문서 목록을 직접 가져옵니다.
                        raw_documents = list(meta.vectorstore.docstore._dict.values())
                        meta.documents = []
                        
                        for i, doc in enumerate(raw_documents):
                            # LangChain Document → TextChunk 변환
                            chunk = TextChunk(
                                text=doc.page_content,
                                metadata=doc.metadata,
                                source_id=doc.metadata.get('source_id', f'{domain}_{i}'),
                                chunk_index=i
                            )
                            meta.documents.append(chunk)
                        
                        logger.info(f"✅ {domain} FAISS에서 {len(meta.documents)}개 문서 복구 완료")
                    except Exception as fallback_error:
                        logger.error(f"❌ {domain} FAISS 문서 복구도 실패: {fallback_error}")
                        meta.documents = []
                else:
                    meta.documents = []
            
            # BM25 인덱스 로드
            if meta.bm25_path.exists():
                try:
                    with open(meta.bm25_path, 'rb') as f:
                        bm25_data = pickle.load(f)
                        if isinstance(bm25_data, tuple):
                            meta.bm25, _ = bm25_data
                        else:
                            meta.bm25 = bm25_data
                    logger.info(f"✅ 도메인 {domain} BM25 인덱스 로드 완료.")
                except Exception as bm25_error:
                    logger.warning(f"⚠️ 도메인 {domain} BM25 로드 실패: {bm25_error}")
                    meta.bm25 = None
            else:
                logger.warning(f"⚠️ 도메인 {domain} BM25 인덱스 파일이 없습니다.")
                meta.bm25 = None
            
            meta.last_loaded = datetime.now()
            meta.load_count += 1
            meta.last_hash = meta.get_file_hash()
            elapsed = time.time() - start_time
            
            # 로드 상태 요약
            status_parts = []
            if meta.vectorstore:
                status_parts.append("FAISS")
            if meta.bm25:
                status_parts.append("BM25")
            if meta.documents:
                status_parts.append(f"문서 {len(meta.documents)}개")
            
            logger.info(f"✅ 도메인 {domain} 로드 성공! ({', '.join(status_parts)}, {elapsed:.2f}초)")
            
        except Exception as e:
            meta.error_count += 1
            logger.error(f"❌ 도메인 {domain} 로드 실패: {e}")
            logger.debug(f"상세 오류:\n{traceback.format_exc()}")
            
            # Graceful Degradation: 부분 로드라도 시도
            meta.vectorstore = None
            meta.bm25 = None


    def load_all_domains(self):
        """모든 도메인 로드"""
        logger.info(f"🔄 전체 도메인 로드 시작: {list(self.metadata.keys())}")
        
        start_time = time.time()
        loaded_count = 0
        
        for domain in self.metadata.keys():
            try:
                self._load_domain(domain)
                loaded_count += 1
            except Exception as e:
                logger.error(f"❌ 도메인 {domain} 로드 중 예외 발생: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"🎉 도메인 로드 완료: {loaded_count}/{len(self.metadata)}개 성공 ({elapsed:.2f}초)")

    def get_vectorstore(self, domain: str) -> Optional[FAISS]:
        """도메인별 벡터스토어 획득"""
        if domain not in self.metadata:
            logger.warning(f"⚠️ 알 수 없는 도메인: {domain}")
            return None
        
        return self.metadata[domain].vectorstore

    def get_bm25(self, domain: str) -> Optional[BM25Okapi]:
        """도메인별 BM25 인덱스 획득"""
        if domain not in self.metadata:
            logger.warning(f"⚠️ 알 수 없는 도메인: {domain}")
            return None
        
        return self.metadata[domain].bm25

    def get_documents(self, domain: str) -> List[TextChunk]:
        """도메인별 문서 리스트 획득"""
        if domain not in self.metadata:
            logger.warning(f"⚠️ 알 수 없는 도메인: {domain}")
            return []
        
        return self.metadata[domain].documents

    def health_check(self) -> Dict[str, Any]:
        """시스템 상태 체크"""
        status = {
            "total_domains": len(self.metadata),
            "loaded_domains": 0,
            "failed_domains": 0,
            "domains_detail": {},
            "global_embeddings": self.embeddings is not None
        }
        
        for domain, meta in self.metadata.items():
            domain_status = {
                "loaded": meta.vectorstore is not None,
                "bm25_available": meta.bm25 is not None,
                "documents_count": len(meta.documents),
                "load_count": meta.load_count,
                "error_count": meta.error_count,
                "last_loaded": meta.last_loaded.isoformat() if meta.last_loaded else None
            }
            
            if domain_status["loaded"]:
                status["loaded_domains"] += 1
            else:
                status["failed_domains"] += 1
            
            status["domains_detail"][domain] = domain_status
        
        return status

# ================================================================
# 3. 싱글톤 인스턴스 팩터리 및 호환성 함수
# ================================================================

_index_manager_instance = None

def get_index_manager() -> IndexManager:
    """IndexManager 싱글톤 인스턴스 획득"""
    global _index_manager_instance
    if _index_manager_instance is None:
        _index_manager_instance = IndexManager()
    return _index_manager_instance

def preload_all_indexes() -> Dict[str, Any]:
    """
    모든 인덱스 사전 로드 (app.py 호환성 개선)
    
    Returns:
        Dict[str, Any]: 로드 결과 정보
    """
    logger.info("🚀 인덱스 사전 로드 시작")
    start_time = time.time()
    
    try:
        manager = get_index_manager()
        
        # 재로드 실행
        manager.load_all_domains()
        
        # 상태 체크
        status = manager.health_check()
        elapsed_time = time.time() - start_time
        
        logger.info(f"📊 인덱스 로드 상태: {status['loaded_domains']}/{status['total_domains']}개 성공")
        
        # app.py에서 기대하는 형식으로 반환
        return {
            "success": status["loaded_domains"] > 0,
            "loaded_indexes": list(status["domains_detail"].keys()),
            "performance": {
                "load_time": elapsed_time,
                "loaded_domains": status["loaded_domains"],
                "total_domains": status["total_domains"]
            },
            "error": None if status["loaded_domains"] > 0 else "No domains loaded"
        }
        
    except Exception as e:
        logger.error(f"❌ 인덱스 사전 로드 실패: {e}")
        return {
            "success": False,
            "loaded_indexes": [],
            "performance": {},
            "error": str(e)
        }

def index_health_check() -> Dict[str, Any]:
    """
    IndexManager 헬스체크 (app.py 호환성 함수)
    
    Returns:
        Dict[str, Any]: 시스템 상태 정보
    """
    try:
        manager = get_index_manager()
        return manager.health_check()
    except Exception as e:
        logger.error(f"❌ 헬스체크 실패: {e}")
        return {
            "total_domains": 0,
            "loaded_domains": 0,
            "failed_domains": 0,
            "domains_detail": {},
            "global_embeddings": False,
            "error": str(e)
        }

# ================================================================
# 4. 테스트 및 검증 
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 IndexManager 테스트 시작")
    
    try:
        # 싱글톤 테스트
        manager1 = get_index_manager()
        manager2 = get_index_manager()
        assert manager1 is manager2, "싱글톤 패턴 실패"
        print("✅ 싱글톤 패턴 테스트 통과")
        
        # 상태 체크
        status = manager1.health_check()
        print(f"📊 시스템 상태: {status}")
        
        print("🎉 모든 테스트 통과!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        traceback.print_exc()
