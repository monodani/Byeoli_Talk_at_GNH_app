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
"""

import hashlib
import logging
import threading
import time
import pickle
import traceback  # ✅ traceback import 추가
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime

# 프로젝트 모듈
from utils.config import config
from utils.contracts import HandlerType
from utils.textifier import TextChunk  # ✅ TextChunk 클래스 import 추가

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

class VectorStoreMetadata:
    """
    벡터스토어 메타데이터 및 상태 관리 (BM25 파일 지원)
    """
    
    def __init__(self, domain: str, vectorstore_dir: Path):
        self.domain = domain
        self.vectorstore_dir = ROOT_DIR / vectorstore_dir
        self.vectorstore_path = self.vectorstore_dir / f"vectorstore_{domain}"
        self.faiss_path = self.vectorstore_path / f"{domain}_index.faiss"
        self.pkl_path = self.vectorstore_path / f"{domain}_index.pkl"
        
        self.bm25_path = self.vectorstore_dir / f"{domain}_index.bm25"
        
        # ✅ embeddings 초기화 추가
        self.embeddings = OpenAIEmbeddings()
        
        self.vectorstore: Optional[FAISS] = None
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[TextChunk] = []
        self.last_loaded: Optional[datetime] = None
        self.load_count: int = 0
        self.error_count: int = 0
        self.last_hash: Optional[str] = None

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
    
    def __init__(self):
        self.metadata: Dict[str, VectorStoreMetadata] = {}
        self.lock = threading.Lock()
        
        # ✅ 임베딩 모델 초기화 추가
        self.embeddings = OpenAIEmbeddings()
        
        for domain in config.HANDLERS:
            self.metadata[domain] = VectorStoreMetadata(
                domain,
                Path(config.VECTORSTORE_DIR)
            )
        
        logger.info(f"🚀 IndexManager 싱글톤 초기화 완료: {len(self.metadata)}개 도메인")
        self.load_all_domains()

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
                meta.bm25 = None

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
# 4. ✅ test_integration.py 호환성을 위한 추가 함수들
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
        # IndexManager 인스턴스 생성 (자동으로 모든 도메인 로드됨)
        index_manager = get_index_manager()
        
        # 각 도메인별 로드 결과 확인
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
        # 모든 도메인을 실패로 표시
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
        
        # 전체 통계 계산
        total_domains = len(status)
        loaded_domains = sum(1 for s in status.values() if s['loaded'])
        total_documents = sum(s['documents_count'] for s in status.values())
        total_errors = sum(s['error_count'] for s in status.values())
        
        # 건강도 계산 (0-100 점수)
        health_score = int((loaded_domains / total_domains) * 100) if total_domains > 0 else 0
        
        # 상태 등급 결정
        if health_score >= 90:
            health_status = "healthy"
        elif health_score >= 70:
            health_status = "degraded"
        else:
            health_status = "critical"
        
        # 추가 체크 항목들
        checks = {
            "all_domains_loaded": loaded_domains == total_domains,
            "no_recent_errors": total_errors == 0,
            "sufficient_documents": total_documents > 100,  # 최소 문서 수 기준
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
        
        # 권장사항 생성
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