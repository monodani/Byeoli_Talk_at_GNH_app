#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - index_manager.py (BM25 로드 및 import 수정)

IndexManager 싱글톤: 모든 벡터스토어 중앙 관리
- 앱 기동 시 모든 FAISS 인덱스 사전 로드
- 저장된 BM25 파일 로드
- 해시 기반 파일 변경 감지로 핫스왑
- 전역 공유로 핸들러 간 일관성 보장
- 메모리 효율적인 단일 인스턴스 관리

핵심 수정사항:
✅ TextChunk import 누락 오류 수정
✅ 경로 오류 수정: VectorStoreMetadata 초기화 시 절대 경로 사용
✅ preload_all_indexes 함수 추가 (test_integration.py 호환)
✅ traceback import 추가
✅ embeddings 초기화 누락 수정
✅ health_check() 메서드 추가
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
from langchain_community.embeddings import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np

# 로깅 설정
logger = logging.getLogger(__name__)


# ================================================================
# 프로젝트 루트 경로 설정 (기존과 동일)
# ================================================================
try:
    ROOT_DIR = Path(__file__).parent.parent.absolute()
except NameError:
    ROOT_DIR = Path(".").absolute()

# ================================================================
# 1. 벡터스토어 메타데이터 클래스 (BM25 파일 추가)
# ================================================================

@dataclass
class VectorStoreMetadata:
    """
    벡터스토어 메타데이터 및 상태 관리 (BM25 파일 지원)
    """
    domain: str
    vectorstore_dir: Path
    
    # post_init에서 설정되므로 init=False로 설정
    vectorstore_path: Path = field(init=False)
    faiss_path: Path = field(init=False)
    pkl_path: Path = field(init=False)
    bm25_path: Path = field(init=False)
    
    # 런타임 속성
    embeddings: Optional[OpenAIEmbeddings] = None
    vectorstore: Optional[FAISS] = None
    bm25: Optional[BM25Okapi] = None
    documents: List[TextChunk] = field(default_factory=list)
    last_loaded: Optional[datetime] = None
    load_count: int = 0
    error_count: int = 0
    last_hash: Optional[str] = None
    
    def __post_init__(self):
        self.vectorstore_path = ROOT_DIR / self.vectorstores_dir / f"vectorstore_{self.domain}"
        self.faiss_path = self.vectorstore_path / f"{self.domain}_index.faiss"
        self.pkl_path = self.vectorstore_path / f"{self.domain}_index.pkl"
        self.bm25_path = self.vectorstore_path / f"{self.domain}_index.bm25"
        self.embeddings = OpenAIEmbeddings()

    def exists(self) -> bool:
        """
        필요한 인덱스 파일들이 모두 존재하는지 확인
        """
        return self.faiss_path.exists() and self.pkl_path.exists() and self.bm25_path.exists()

    @property
    def needs_reload(self) -> bool:
        """
        파일 변경 여부를 감지하여 리로드 필요성을 판단
        """
        if not self.exists():
            return False
            
        current_hash = self.get_file_hash()
        return current_hash != self.last_hash
        
    def get_file_hash(self) -> str:
        """
        모든 인덱스 파일의 해시를 합쳐서 반환
        """
        hasher = hashlib.sha256()
        try:
            for path in [self.faiss_path, self.pkl_path, self.bm25_path]:
                if path.exists():
                    hasher.update(path.read_bytes())
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"❌ 도메인 {self.domain} 파일 해시 계산 실패: {e}")
            self.error_count += 1
            return ""

# ================================================================
# 2. IndexManager 싱글톤 클래스
# ================================================================

class IndexManager:
    """
    모든 벡터스토어를 관리하는 싱글톤 클래스
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
        
        self.embeddings = OpenAIEmbeddings()
        
        for domain in config.HANDLERS:
            self.metadata[domain] = VectorStoreMetadata(
                domain,
                Path(config.VECTORSTORE_DIR)
            )
        
        logger.info(f"🚀 IndexManager 싱글톤 초기화 완료: {len(self.metadata)}개 도메인")
        self.load_all_domains()
        self._initialized = True

    def _load_domain(self, domain: str):
        """
        단일 도메인의 벡터스토어를 로드
        """
        meta = self.metadata[domain]
        
        logger.info(f"🔄 도메인 {domain} 로드 시작...")
        
        try:
            if not meta.exists():
                logger.warning(f"⚠️ 도메인 {domain}에 필요한 인덱스 파일이 없습니다. 로드 건너뜁니다.")
                meta.vectorstore = None
                meta.bm25 = None
                return
            
            start_time = time.time()
            meta.vectorstore = FAISS.load_local(
                str(meta.vectorstore_path),
                meta.embeddings,
                allow_dangerous_deserialization=True
            )
            
            with open(meta.pkl_path, "rb") as f:
                meta.documents = pickle.load(f)
            
            if meta.bm25_path.exists():
                with open(meta.bm25_path, 'rb') as f:
                    meta.bm25 = pickle.load(f)
                logger.info(f"✅ 도메인 {domain} BM25 인덱스 로드 완료.")
            else:
                logger.warning(f"⚠️ 도메인 {domain} BM25 인덱스 파일이 없습니다. BM25 검색을 사용할 수 없습니다.")
            
            meta.last_loaded = datetime.now()
            meta.load_count += 1
            meta.last_hash = meta.get_file_hash()
            elapsed = time.time() - start_time
            logger.info(f"✅ 도메인 {domain} 로드 성공! ({len(meta.documents):,}개 문서, {elapsed:.2f}초)")

        except Exception as e:
            meta.error_count += 1
            logger.error(f"❌ 도메인 {domain} 로드 실패: {e}")
            logger.debug(traceback.format_exc())
            meta.vectorstore = None
            meta.bm25 = None
            
    def load_all_domains(self):
        """병렬로 모든 도메인을 로드"""
        threads = []
        for domain in self.metadata:
            thread = threading.Thread(target=self._load_domain, args=(domain,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
            
    def check_for_updates_and_reload(self):
        """
        파일 변경을 감지하고, 변경된 도메인만 핫스왑 실행
        """
        for domain, meta in self.metadata.items():
            if meta.needs_reload:
                logger.info(f"🔄 도메인 {domain} 파일 변경 감지, 핫스왑 실행...")
                self._load_domain(domain)
    
    def get_vectorstore(self, domain: str) -> Optional[FAISS]:
        """도메인에 해당하는 FAISS 벡터스토어 반환 (base_handler 호환)"""
        meta = self.metadata.get(domain)
        return meta.vectorstore if meta else None
    
    def get_index(self, domain: str) -> Optional[FAISS]:
        """도메인에 해당하는 FAISS 인덱스 반환 (기존 호환성)"""
        return self.get_vectorstore(domain)
    
    def get_documents(self, domain: str) -> List[TextChunk]:
        """도메인에 해당하는 원본 문서 청크 반환"""
        meta = self.metadata.get(domain)
        return meta.documents if meta else []
        
    def get_bm25(self, domain: str) -> Optional[BM25Okapi]:
        """도메인에 해당하는 BM25 인덱스 반환"""
        meta = self.metadata.get(domain)
        return meta.bm25 if meta else None
        
    def get_status(self) -> Dict[str, Dict]:
        """
        전체 인덱스 상태 정보 반환
        """
        status = {}
        
        for domain, meta in self.metadata.items():
            status[domain] = {
                'loaded': meta.vectorstore is not None,
                'documents_count': len(meta.documents),
                'has_bm25': meta.bm25 is not None,
                'last_loaded': meta.last_loaded.isoformat() if meta.last_loaded else None,
                'load_count': meta.load_count,
                'error_count': meta.error_count,
                'files_exist': meta.exists()
            }
        
        return status

    def health_check(self) -> Dict[str, Dict]:
        """
        test_integration.py 호환성을 위해 추가된 메서드
        모든 도메인의 인덱스 상태를 점검하고 반환합니다.
        
        get_status() 메서드를 활용하여 현재 로드 상태를 반환합니다.
        """
        return self.get_status()

# ================================================================
# 3. 싱글톤 인스턴스 관리
# ================================================================

_index_manager_instance: Optional[IndexManager] = None
_instance_lock = threading.Lock()


def get_index_manager() -> IndexManager:
    """
    IndexManager 싱글톤 인스턴스 반환
    """
    global _index_manager_instance
    
    if _index_manager_instance is None:
        with _instance_lock:
            if _index_manager_instance is None:
                _index_manager_instance = IndexManager()
    
    return _index_manager_instance


# ================================================================
# 4. test_integration.py 호환성을 위한 추가 함수들
# ================================================================

def preload_all_indexes() -> Dict[str, bool]:
    """
    모든 인덱스를 사전 로드하고 결과를 반환
    test_integration.py에서 요구하는 함수
    
    Returns:
        Dict[str, bool]: 도메인별 로드 성공 여부
    """
    logger.info("🚀 모든 인덱스 사전 로드 시작...")
    
    try:
        index_manager = get_index_manager()
        
        results = {}
        status = index_manager.get_status()
        
        for domain, domain_status in status.items():
            is_loaded = domain_status['loaded'] and domain_status['documents_count'] > 0
            results[domain] = is_loaded
            
            if is_loaded:
                logger.info(f"✅ {domain}: {domain_status['documents_count']}개 문서 로드 완료")
            else:
                logger.warning(f"⚠️ {domain}: 로드 실패 또는 문서 없음")
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        logger.info(f"📊 인덱스 사전 로드 완료: {success_count}/{total_count}개 도메인 성공")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 인덱스 사전 로드 실패: {e}")
        return {domain: False for domain in config.HANDLERS}


def index_health_check() -> Dict[str, Any]:
    """
    인덱스 상태 건강 검진
    
    Returns:
        Dict[str, Any]: 상태 정보 및 건강도 지표
    """
    try:
        index_manager = get_index_manager()
        status = index_manager.get_status()
        
        total_domains = len(status)
        loaded_domains = sum(1 for s in status.values() if s['loaded'])
        total_documents = sum(s['documents_count'] for s in status.values())
        total_errors = sum(s['error_count'] for s in status.values())
        
        health_score = int((loaded_domains / total_domains) * 100) if total_domains > 0 else 0
        
        if health_score >= 90:
            health_status = "healthy"
        elif health_score >= 70:
            health_status = "degraded"
        else:
            health_status = "critical"
        
        checks = {
            "all_domains_loaded": loaded_domains == total_domains,
            "no_recent_errors": total_errors == 0,
            "sufficient_documents": total_documents > 100,
            "embeddings_available": hasattr(index_manager, 'metadata')
        }
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "health_status": health_status,
            "health_score": health_score,
            "summary": {
                "total_domains": total_domains,
                "loaded_domains": loaded_domains,
                "total_documents": total_documents,
                "total_errors": total_errors
            },
            "checks": checks,
            "domain_details": status,
            "recommendations": []
        }
        
        if health_score < 100:
            failed_domains = [domain for domain, s in status.items() if not s['loaded']]
            if failed_domains:
                result["recommendations"].append(f"다음 도메인 재구축 필요: {', '.join(failed_domains)}")
        
        if total_errors > 0:
            result["recommendations"].append("에러가 발생한 도메인의 로그를 확인해주세요")
        
        if total_documents < 100:
            result["recommendations"].append("문서 수가 부족합니다. 데이터 소스를 확인해주세요")
        
        logger.info(f"🏥 건강 검진 완료 - 상태: {health_status}, 점수: {health_score}/100")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 건강 검진 실패: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "health_status": "error",
            "health_score": 0,
            "error": str(e),
            "recommendations": ["시스템 재시작이 필요할 수 있습니다"]
        }


# ================================================================
# 5. 모듈 로드 완료 로그
# ================================================================

logger.info("✅ index_manager.py 모듈 로드 완료 (preload_all_indexes 포함)")
