#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - index_manager.py

IndexManager 싱글톤: 모든 벡터스토어 중앙 관리
- 앱 기동 시 모든 FAISS 인덱스 사전 로드
- 해시 기반 파일 변경 감지로 핫스왑
- 전역 공유로 핸들러 간 일관성 보장
- 메모리 효율적인 단일 인스턴스 관리

핵심 특징:
- 싱글톤 패턴으로 전역 인스턴스 보장
- 지연 로딩 및 캐시 기반 성능 최적화
- 파일 해시 감시로 자동 핫스왑
- 에러 복구 및 안전망 제공
"""

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime

# 프로젝트 모듈
from utils.config import config
from utils.contracts import HandlerType

# 외부 라이브러리
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np

# 로깅 설정
logger = logging.getLogger(__name__)


# ================================================================
# 1. 벡터스토어 메타데이터 클래스
# ================================================================

class VectorStoreMetadata:
    """벡터스토어 메타데이터 및 상태 관리"""
    
    def __init__(self, domain: str, vectorstore_dir: Path):
        self.domain = domain
        self.vectorstore_dir = vectorstore_dir
        self.faiss_file = vectorstore_dir / f"{domain}_index.faiss"
        self.pkl_file = vectorstore_dir / f"{domain}_index.pkl"
        
        # 상태 정보
        self.vectorstore: Optional[FAISS] = None
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.last_loaded: Optional[datetime] = None
        self.file_hash: Optional[str] = None
        self.load_count: int = 0
        self.error_count: int = 0
        
    def exists(self) -> bool:
        """벡터스토어 파일 존재 여부"""
        return self.faiss_file.exists() and self.pkl_file.exists()
    
    def calculate_hash(self) -> str:
        """벡터스토어 파일들의 해시 계산"""
        if not self.exists():
            return ""
        
        try:
            hash_md5 = hashlib.md5()
            
            # FAISS 파일 해시
            with open(self.faiss_file, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            # PKL 파일 해시
            with open(self.pkl_file, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            return hash_md5.hexdigest()
            
        except Exception as e:
            logger.error(f"❌ {self.domain} 해시 계산 실패: {e}")
            return ""
    
    def needs_reload(self) -> bool:
        """재로드 필요 여부 확인"""
        if not self.vectorstore:
            return True
        
        current_hash = self.calculate_hash()
        return current_hash != self.file_hash
    
    def mark_loaded(self, success: bool = True):
        """로드 완료 마킹"""
        self.last_loaded = datetime.now()
        self.load_count += 1
        if not success:
            self.error_count += 1
        self.file_hash = self.calculate_hash()


# ================================================================
# 2. IndexManager 싱글톤 클래스
# ================================================================

class IndexManager:
    """
    벡터스토어 인덱스 중앙 관리자 (싱글톤)
    
    주요 기능:
    - 모든 도메인의 벡터스토어 사전 로드
    - 파일 변경 감지 및 자동 핫스왑
    - 스레드 안전한 액세스 제공
    - 성능 모니터링 및 에러 처리
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # 초기화
        self.embeddings = OpenAIEmbeddings()
        self.domains = self._get_domain_configs()
        self.metadata: Dict[str, VectorStoreMetadata] = {}
        self._access_lock = threading.RLock()
        
        # 도메인별 메타데이터 초기화
        for domain, config_info in self.domains.items():
            self.metadata[domain] = VectorStoreMetadata(
                domain=domain,
                vectorstore_dir=config_info["path"]
            )
        
        self._initialized = True
        logger.info(f"🚀 IndexManager 싱글톤 초기화 완료: {len(self.domains)}개 도메인")
    
    def _get_domain_configs(self) -> Dict[str, Dict[str, Any]]:
        """도메인별 설정 정보 반환"""
        vectorstore_base = config.ROOT_DIR / "vectorstores"
        
        return {
            "satisfaction": {
                "path": vectorstore_base / "vectorstore_unified_satisfaction",
                "handler_type": HandlerType.SATISFACTION
            },
            "general": {
                "path": vectorstore_base / "vectorstore_general",
                "handler_type": HandlerType.GENERAL
            },
            "menu": {
                "path": vectorstore_base / "vectorstore_menu",
                "handler_type": HandlerType.MENU
            },
            "cyber": {
                "path": vectorstore_base / "vectorstore_cyber",
                "handler_type": HandlerType.CYBER
            },
            "publish": {
                "path": vectorstore_base / "vectorstore_unified_publish",
                "handler_type": HandlerType.PUBLISH
            },
            "notice": {
                "path": vectorstore_base / "vectorstore_notice",
                "handler_type": HandlerType.NOTICE
            }
        }
    
    def preload_all(self) -> Dict[str, bool]:
        """
        모든 벡터스토어 사전 로드
        
        Returns:
            Dict[str, bool]: 도메인별 로드 성공 여부
        """
        logger.info("📚 전체 벡터스토어 사전 로드 시작...")
        start_time = time.time()
        
        results = {}
        for domain in self.domains.keys():
            try:
                success = self._load_domain(domain)
                results[domain] = success
                
                if success:
                    logger.info(f"✅ {domain} 로드 성공")
                else:
                    logger.warning(f"⚠️ {domain} 로드 실패")
                    
            except Exception as e:
                logger.error(f"❌ {domain} 로드 중 예외: {e}")
                results[domain] = False
        
        elapsed_time = time.time() - start_time
        success_count = sum(1 for success in results.values() if success)
        
        logger.info(f"📊 사전 로드 완료: {success_count}/{len(results)} 성공 ({elapsed_time:.2f}s)")
        return results
    
    def _load_domain(self, domain: str) -> bool:
        """
        특정 도메인 벡터스토어 로드
        
        Args:
            domain: 도메인 이름
            
        Returns:
            bool: 로드 성공 여부
        """
        if domain not in self.metadata:
            logger.error(f"❌ 알 수 없는 도메인: {domain}")
            return False
        
        meta = self.metadata[domain]
        
        # 파일 존재 확인
        if not meta.exists():
            logger.warning(f"⚠️ {domain} 벡터스토어 파일이 없습니다: {meta.vectorstore_dir}")
            return False
        
        try:
            with self._access_lock:
                # FAISS 벡터스토어 로드
                vectorstore = FAISS.load_local(
                    folder_path=str(meta.vectorstore_dir),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                    index_name=f"{domain}_index"
                )
                
                # 문서 내용 추출 (BM25용)
                documents = []
                docstore = vectorstore.docstore._dict
                
                for doc_id in range(len(docstore)):
                    doc = docstore.get(str(doc_id))
                    if doc and hasattr(doc, 'page_content'):
                        documents.append(doc.page_content)
                
                # BM25 인덱스 구축
                bm25 = None
                if documents:
                    tokenized_docs = [doc.split() for doc in documents]
                    bm25 = BM25Okapi(tokenized_docs)
                
                # 메타데이터 업데이트
                meta.vectorstore = vectorstore
                meta.bm25 = bm25
                meta.documents = documents
                meta.mark_loaded(success=True)
                
                logger.debug(f"📄 {domain}: {len(documents)}개 문서, BM25 {'✓' if bm25 else '✗'}")
                return True
                
        except Exception as e:
            logger.error(f"❌ {domain} 벡터스토어 로드 실패: {e}")
            meta.mark_loaded(success=False)
            return False
    
    def get_vectorstore(self, domain: str) -> Optional[FAISS]:
        """
        도메인별 FAISS 벡터스토어 반환
        
        Args:
            domain: 도메인 이름
            
        Returns:
            Optional[FAISS]: 로드된 벡터스토어 또는 None
        """
        if domain not in self.metadata:
            logger.error(f"❌ 알 수 없는 도메인: {domain}")
            return None
        
        meta = self.metadata[domain]
        
        # 핫스왑 체크
        if meta.needs_reload():
            logger.info(f"🔄 {domain} 파일 변경 감지, 핫스왑 실행...")
            self._load_domain(domain)
        
        return meta.vectorstore
    
    def get_bm25(self, domain: str) -> Optional[BM25Okapi]:
        """
        도메인별 BM25 인덱스 반환
        
        Args:
            domain: 도메인 이름
            
        Returns:
            Optional[BM25Okapi]: BM25 인덱스 또는 None
        """
        if domain not in self.metadata:
            return None
        
        meta = self.metadata[domain]
        
        # 핫스왑 체크
        if meta.needs_reload():
            self._load_domain(domain)
        
        return meta.bm25
    
    def get_documents(self, domain: str) -> List[str]:
        """
        도메인별 문서 목록 반환
        
        Args:
            domain: 도메인 이름
            
        Returns:
            List[str]: 문서 텍스트 목록
        """
        if domain not in self.metadata:
            return []
        
        meta = self.metadata[domain]
        
        # 핫스왑 체크
        if meta.needs_reload():
            self._load_domain(domain)
        
        return meta.documents.copy()
    
    def hybrid_search(
        self, 
        domain: str, 
        query: str, 
        k: int = 10,
        rrf_k: int = 60
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        도메인별 하이브리드 검색 (FAISS + BM25 + RRF)
        
        Args:
            domain: 검색 대상 도메인
            query: 검색 쿼리
            k: 반환할 결과 수
            rrf_k: RRF 파라미터
            
        Returns:
            List[Tuple[str, float, dict]]: (텍스트, 점수, 메타데이터) 튜플 목록
        """
        vectorstore = self.get_vectorstore(domain)
        bm25 = self.get_bm25(domain)
        documents = self.get_documents(domain)
        
        if not vectorstore:
            logger.warning(f"⚠️ {domain} 벡터스토어를 사용할 수 없습니다")
            return []
        
        try:
            # 1. FAISS 검색
            faiss_results = vectorstore.similarity_search_with_score(query, k=k)
            faiss_docs = [(doc.page_content, score, doc.metadata) for doc, score in faiss_results]
            
            # 2. BM25 검색
            bm25_docs = []
            if bm25 and documents:
                tokenized_query = query.split()
                bm25_scores = bm25.get_scores(tokenized_query)
                
                # 상위 k개 선택
                top_indices = np.argsort(bm25_scores)[-k:][::-1]
                for idx in top_indices:
                    if idx < len(documents):
                        # 메타데이터 찾기
                        doc_id = str(idx)
                        metadata = {}
                        if hasattr(vectorstore, 'docstore') and doc_id in vectorstore.docstore._dict:
                            metadata = vectorstore.docstore._dict[doc_id].metadata
                        
                        bm25_docs.append((documents[idx], bm25_scores[idx], metadata))
            
            # 3. RRF 융합
            combined_results = self._rrf_fusion(faiss_docs, bm25_docs, k, rrf_k)
            
            logger.debug(f"🔍 {domain} 하이브리드 검색: FAISS {len(faiss_docs)}, BM25 {len(bm25_docs)}, 융합 {len(combined_results)}")
            return combined_results
            
        except Exception as e:
            logger.error(f"❌ {domain} 하이브리드 검색 실패: {e}")
            return []
    
    def _rrf_fusion(
        self, 
        faiss_results: List[Tuple[str, float, Dict]], 
        bm25_results: List[Tuple[str, float, Dict]], 
        k: int, 
        rrf_k: int = 60
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Reciprocal Rank Fusion (RRF) 알고리즘으로 결과 융합
        
        Args:
            faiss_results: FAISS 검색 결과
            bm25_results: BM25 검색 결과
            k: 최종 반환할 결과 수
            rrf_k: RRF 파라미터
            
        Returns:
            List[Tuple]: 융합된 검색 결과
        """
        try:
            # 문서별 점수 집계
            doc_scores = {}
            
            # FAISS 결과 처리 (정규화된 순위 점수)
            for rank, (text, score, metadata) in enumerate(faiss_results, 1):
                doc_key = text[:100]  # 문서 식별용 키
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {
                        'text': text,
                        'metadata': metadata,
                        'faiss_score': score,
                        'bm25_score': 0.0,
                        'rrf_score': 0.0
                    }
                
                # RRF 점수 계산: 1 / (rank + k)
                doc_scores[doc_key]['rrf_score'] += 1.0 / (rank + rrf_k)
                doc_scores[doc_key]['faiss_score'] = max(doc_scores[doc_key]['faiss_score'], score)
            
            # BM25 결과 처리
            for rank, (text, score, metadata) in enumerate(bm25_results, 1):
                doc_key = text[:100]
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {
                        'text': text,
                        'metadata': metadata,
                        'faiss_score': 0.0,
                        'bm25_score': score,
                        'rrf_score': 0.0
                    }
                
                doc_scores[doc_key]['rrf_score'] += 1.0 / (rank + rrf_k)
                doc_scores[doc_key]['bm25_score'] = max(doc_scores[doc_key]['bm25_score'], score)
            
            # RRF 점수 기준 정렬
            sorted_docs = sorted(
                doc_scores.values(),
                key=lambda x: x['rrf_score'],
                reverse=True
            )
            
            # 상위 k개 반환
            return [
                (doc['text'], doc['rrf_score'], doc['metadata'])
                for doc in sorted_docs[:k]
            ]
            
        except Exception as e:
            logger.error(f"❌ RRF 융합 실패: {e}")
            # 융합 실패 시 FAISS 결과만 반환
            return faiss_results[:k]
    
    def force_reload(self, domain: str) -> bool:
        """
        특정 도메인 강제 재로드
        
        Args:
            domain: 재로드할 도메인
            
        Returns:
            bool: 재로드 성공 여부
        """
        if domain not in self.metadata:
            logger.error(f"❌ 알 수 없는 도메인: {domain}")
            return False
        
        logger.info(f"🔄 {domain} 강제 재로드 실행...")
        
        # 기존 데이터 클리어
        meta = self.metadata[domain]
        meta.vectorstore = None
        meta.bm25 = None
        meta.documents = []
        meta.file_hash = None
        
        # 재로드
        return self._load_domain(domain)
    
    def force_reload_all(self) -> Dict[str, bool]:
        """
        모든 도메인 강제 재로드
        
        Returns:
            Dict[str, bool]: 도메인별 재로드 결과
        """
        logger.info("🔄 전체 도메인 강제 재로드 시작...")
        
        results = {}
        for domain in self.domains.keys():
            results[domain] = self.force_reload(domain)
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"🔄 전체 재로드 완료: {success_count}/{len(results)} 성공")
        
        return results
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """
        전체 인덱스 상태 정보 반환
        
        Returns:
            Dict[str, Dict]: 도메인별 상태 정보
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
                'file_exists': meta.exists(),
                'file_hash': meta.file_hash[:8] if meta.file_hash else None,
                'needs_reload': meta.needs_reload()
            }
        
        return status
    
    def health_check(self) -> Dict[str, Any]:
        """
        IndexManager 헬스체크
        
        Returns:
            Dict[str, Any]: 헬스체크 결과
        """
        status = self.get_status()
        
        total_domains = len(status)
        loaded_domains = sum(1 for s in status.values() if s['loaded'])
        total_documents = sum(s['documents_count'] for s in status.values())
        total_errors = sum(s['error_count'] for s in status.values())
        
        health = {
            'overall_health': 'healthy' if loaded_domains == total_domains else 'degraded',
            'loaded_domains': f"{loaded_domains}/{total_domains}",
            'total_documents': total_documents,
            'total_errors': total_errors,
            'domains': status,
            'timestamp': datetime.now().isoformat()
        }
        
        return health
    
    def cleanup(self):
        """리소스 정리"""
        logger.info("🧹 IndexManager 리소스 정리 시작...")
        
        with self._access_lock:
            for meta in self.metadata.values():
                meta.vectorstore = None
                meta.bm25 = None
                meta.documents = []
        
        logger.info("✅ IndexManager 리소스 정리 완료")


# ================================================================
# 3. 전역 접근 함수들
# ================================================================

def get_index_manager() -> IndexManager:
    """전역 IndexManager 인스턴스 반환"""
    return IndexManager()


def preload_all_indexes() -> Dict[str, bool]:
    """모든 인덱스 사전 로드 (앱 초기화용)"""
    manager = get_index_manager()
    return manager.preload_all()


def get_vectorstore(domain: str) -> Optional[FAISS]:
    """도메인별 벡터스토어 반환 (핸들러용)"""
    manager = get_index_manager()
    return manager.get_vectorstore(domain)


def hybrid_search(domain: str, query: str, k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
    """도메인별 하이브리드 검색 (핸들러용)"""
    manager = get_index_manager()
    return manager.hybrid_search(domain, query, k)


def index_health_check() -> Dict[str, Any]:
    """인덱스 상태 확인 (모니터링용)"""
    manager = get_index_manager()
    return manager.health_check()


# ================================================================
# 4. 개발/테스트용 함수들
# ================================================================

def test_index_manager():
    """IndexManager 기능 테스트"""
    logger.info("🧪 IndexManager 테스트 시작...")
    
    try:
        # 1. 인스턴스 생성 테스트
        manager = get_index_manager()
        logger.info("✅ 싱글톤 인스턴스 생성 성공")
        
        # 2. 헬스체크
        health = manager.health_check()
        logger.info(f"📊 초기 상태: {health['overall_health']}")
        
        # 3. 사전 로드 테스트
        results = manager.preload_all()
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"📚 사전 로드 결과: {success_count}/{len(results)} 성공")
        
        # 4. 검색 테스트
        test_queries = {
            "satisfaction": "교육과정 만족도",
            "general": "학칙 규정",
            "menu": "식단 메뉴",
            "cyber": "사이버교육",
            "publish": "교육계획",
            "notice": "공지사항"
        }
        
        for domain, query in test_queries.items():
            try:
                results = manager.hybrid_search(domain, query, k=3)
                logger.info(f"🔍 {domain} 검색 테스트: {len(results)}건 반환")
            except Exception as e:
                logger.error(f"❌ {domain} 검색 실패: {e}")
        
        # 5. 최종 헬스체크
        final_health = manager.health_check()
        logger.info(f"🏁 최종 상태: {final_health['overall_health']}")
        logger.info(f"📄 총 문서 수: {final_health['total_documents']}")
        
        return final_health
        
    except Exception as e:
        logger.error(f"❌ IndexManager 테스트 실패: {e}")
        return None


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 간단한 테스트 실행
    test_index_manager()
