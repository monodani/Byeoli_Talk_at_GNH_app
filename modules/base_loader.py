#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - BaseLoader 기본 클래스 (개선버전)

주요 개선사항:
- config.py에서 임베딩 모델 설정 가져오기
- 차원 검증 추가
- 더 강화된 에러 처리
- 마이그레이션 지원
"""

import logging
import hashlib
import time
import pickle
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 프로젝트 모듈 임포트
from utils.textifier import TextChunk

# Config 임포트 (동적 임포트로 에러 방지)
try:
    from config import config
    CONFIG_AVAILABLE = True
    EMBEDDING_MODEL = config.EMBEDDING_MODEL
    EMBEDDING_DIMENSION = config.EMBEDDING_DIMENSION
except ImportError:
    CONFIG_AVAILABLE = False
    EMBEDDING_MODEL = "text-embedding-3-large"  # 폴백
    EMBEDDING_DIMENSION = 3072
    logging.warning("config.py를 찾을 수 없어 기본값 사용")

# 외부 라이브러리 (선택적)
try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    FAISS_AVAILABLE = True
except ImportError:
    try:
        from langchain.vectorstores import FAISS
        from langchain.embeddings import OpenAIEmbeddings
        FAISS_AVAILABLE = True
    except ImportError:
        FAISS_AVAILABLE = False

# BM25 라이브러리 (선택적)
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# 로깅 설정
logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """
    모든 도메인 로더의 기본 클래스 (개선버전)
    
    개선사항:
    - config 기반 임베딩 모델 사용
    - 차원 검증 추가
    - 마이그레이션 지원
    """
    
    def __init__(self, domain=None, loader_id=None, source_dir=None, vectorstore_dir=None, target_dir=None, index_name=None, schema_dir=None, **kwargs):
        """
        BaseLoader 초기화
        """
        # 호환성 처리
        self.domain = domain or loader_id
        self.source_dir = Path(source_dir or ".")
    
        # target_dir도 받기 (vectorstore_dir 대신)
        if target_dir:
            self.vectorstore_dir = Path(target_dir)
        else:
            self.vectorstore_dir = Path(vectorstore_dir or ".")
    
        # index_name 기본값 설정  
        self.index_name = index_name or f"{self.domain}_index"
        
        # 임베딩 설정 초기화
        self.embedding_model = EMBEDDING_MODEL
        self.embedding_dimension = EMBEDDING_DIMENSION
        self.embeddings = None
        
        # 디렉터리 생성
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        
        # 임베딩 모델 초기화
        self._initialize_embeddings()
    
        logger.info(f"✨ {self.domain.upper()} BaseLoader 초기화 완료")
        logger.info(f"   - 임베딩 모델: {self.embedding_model}")
        logger.info(f"   - 예상 차원: {self.embedding_dimension}")
    
    def _initialize_embeddings(self):
        """임베딩 모델 초기화"""
        try:
            if not FAISS_AVAILABLE:
                logger.warning("FAISS 라이브러리가 없어 임베딩 초기화 건너뜀")
                return
            
            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # 차원 검증 (테스트 임베딩)
            try:
                test_vector = self.embeddings.embed_query("test")
                actual_dimension = len(test_vector)
                
                if actual_dimension != self.embedding_dimension:
                    logger.warning(
                        f"⚠️ {self.domain} 차원 불일치: "
                        f"예상={self.embedding_dimension}, 실제={actual_dimension}"
                    )
                    self.embedding_dimension = actual_dimension  # 실제 차원으로 업데이트
                
                logger.info(f"✅ {self.domain} 임베딩 초기화 성공 ({actual_dimension}차원)")
                
            except Exception as e:
                logger.warning(f"⚠️ {self.domain} 차원 검증 실패: {e}")
                
        except Exception as e:
            logger.error(f"❌ {self.domain} 임베딩 초기화 실패: {e}")
            self.embeddings = None
    
    @abstractmethod
    def process_domain_data(self) -> List[TextChunk]:
        """
        도메인별 데이터 처리 (각 로더에서 구현)
        
        Returns:
            List[TextChunk]: 처리된 텍스트 청크들
        """
        pass
    
    def get_supported_extensions(self) -> List[str]:
        """지원하는 파일 확장자 (선택적 오버라이드)"""
        return ['.pdf', '.csv', '.txt', '.png', '.jpg']
    
    def validate_schema(self, file_path: Path, schema_path: Path) -> bool:
        """
        스키마 검증 (선택적)
        
        Args:
            file_path: 검증할 파일 경로
            schema_path: 스키마 파일 경로
            
        Returns:
            bool: 검증 성공 여부
        """
        try:
            # 간단한 파일 존재 검증
            return file_path.exists()
        except Exception as e:
            logger.warning(f"스키마 검증 실패: {e}")
            return True  # 검증 실패해도 처리 계속
    
    def calculate_source_hash(self) -> str:
        """소스 데이터 해시 계산 (증분 빌드용)"""
        try:
            hash_md5 = hashlib.md5()
            
            # 임베딩 모델도 해시에 포함 (모델 변경 시 재빌드)
            hash_md5.update(self.embedding_model.encode())
            
            # 소스 디렉터리의 모든 파일 해시 계산
            for file_path in self.source_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix in self.get_supported_extensions():
                    hash_md5.update(str(file_path.stat().st_mtime).encode())
                    hash_md5.update(str(file_path.stat().st_size).encode())
            
            return hash_md5.hexdigest()[:16]
        except Exception as e:
            logger.warning(f"해시 계산 실패: {e}")
            return str(int(time.time()))  # 폴백: 타임스탬프
    
    def check_existing_dimension(self) -> Optional[int]:
        """기존 벡터스토어의 차원 확인"""
        try:
            # 차원 정보 파일 확인
            dimension_file = self.vectorstore_dir / f"{self.index_name}_dimension_info.json"
            if dimension_file.exists():
                import json
                with open(dimension_file, 'r') as f:
                    info = json.load(f)
                return info.get('dimension')
            
            # FAISS 파일에서 직접 확인
            faiss_file = self.vectorstore_dir / f"{self.index_name}.faiss"
            if faiss_file.exists() and self.embeddings:
                try:
                    vectorstore = FAISS.load_local(
                        folder_path=str(self.vectorstore_dir),
                        embeddings=self.embeddings,
                        index_name=self.index_name,
                        allow_dangerous_deserialization=True
                    )
                    if hasattr(vectorstore, 'index'):
                        return vectorstore.index.d
                except Exception:
                    pass
            
            return None
            
        except Exception as e:
            logger.warning(f"기존 차원 확인 실패: {e}")
            return None
    
    def needs_rebuild(self) -> bool:
        """재빌드 필요 여부 확인 (차원 검증 포함)"""
        try:
            # FAISS 파일 확인
            faiss_file = self.vectorstore_dir / f"{self.index_name}.faiss"
            pkl_file = self.vectorstore_dir / f"{self.index_name}.pkl"
            bm25_file = self.vectorstore_dir / f"{self.index_name}.bm25"
            
            # FAISS 파일이 없으면 빌드 필요
            if not (faiss_file.exists() and pkl_file.exists()):
                logger.info(f"🔨 {self.domain}: FAISS 파일이 없어서 새로 빌드")
                return True
                
            # BM25 파일이 없으면 빌드 필요
            if not bm25_file.exists():
                logger.info(f"🔨 {self.domain}: BM25 파일이 없어서 새로 빌드")
                return True
            
            # 차원 호환성 확인
            existing_dimension = self.check_existing_dimension()
            if existing_dimension and existing_dimension != self.embedding_dimension:
                logger.info(
                    f"🔨 {self.domain}: 차원 불일치로 재빌드 필요 "
                    f"(기존={existing_dimension}, 현재={self.embedding_dimension})"
                )
                return True
            
            # 해시 파일 확인
            hash_file = self.vectorstore_dir / ".source_hash"
            if not hash_file.exists():
                logger.info(f"🔨 {self.domain}: 해시 파일이 없어서 새로 빌드")
                return True
            
            # 해시 비교
            current_hash = self.calculate_source_hash()
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            
            if current_hash != stored_hash:
                logger.info(f"🔨 {self.domain}: 소스 데이터 변경으로 재빌드")
                return True
            
            logger.info(f"✅ {self.domain}: FAISS + BM25가 최신 상태")
            return False
            
        except Exception as e:
            logger.warning(f"재빌드 확인 실패: {e}")
            return True  # 확인 실패 시 안전하게 재빌드
    
    def save_source_hash(self):
        """현재 소스 해시 저장"""
        try:
            current_hash = self.calculate_source_hash()
            hash_file = self.vectorstore_dir / ".source_hash"
            with open(hash_file, 'w') as f:
                f.write(current_hash)
            logger.debug(f"📝 {self.domain}: 소스 해시 저장됨")
        except Exception as e:
            logger.warning(f"해시 저장 실패: {e}")
    
    def save_dimension_info(self, vector_count: int):
        """차원 정보 저장 (디버깅용)"""
        try:
            dimension_info = {
                'model': self.embedding_model,
                'dimension': self.embedding_dimension,
                'vector_count': vector_count,
                'created_at': datetime.now().isoformat(),
                'domain': self.domain
            }
            
            dimension_file = self.vectorstore_dir / f"{self.index_name}_dimension_info.json"
            import json
            with open(dimension_file, 'w') as f:
                json.dump(dimension_info, f, indent=2)
                
            logger.debug(f"📝 {self.domain}: 차원 정보 저장됨")
            
        except Exception as e:
            logger.warning(f"차원 정보 저장 실패: {e}")
    
    def build_vectorstore(self, force_rebuild: bool = False) -> bool:
        """
        벡터스토어 빌드 (증분 빌드 지원)
        
        Args:
            force_rebuild: 강제 재빌드 여부
            
        Returns:
            bool: 빌드 성공 여부
        """
        try:
            # 임베딩 모델 확인
            if not self.embeddings:
                logger.error(f"❌ {self.domain}: 임베딩 모델이 초기화되지 않음")
                return False
            
            # 재빌드 필요성 확인
            if not force_rebuild and not self.needs_rebuild():
                logger.info(f"⏭️ {self.domain}: 이미 최신 벡터스토어 존재")
                return True
            
            logger.info(f"🔨 {self.domain}: 벡터스토어 빌드 시작...")
            logger.info(f"   - 모델: {self.embedding_model}")
            logger.info(f"   - 차원: {self.embedding_dimension}")
            start_time = time.time()
            
            # 1. 도메인 데이터 처리
            chunks = self.process_domain_data()
            
            if not chunks:
                logger.warning(f"⚠️ {self.domain}: 처리할 데이터가 없습니다")
                return False
            
            logger.info(f"📄 {self.domain}: {len(chunks)}개 청크 생성됨")
            
            # 2. FAISS 벡터스토어 생성
            if not FAISS_AVAILABLE:
                logger.error(f"❌ {self.domain}: FAISS 라이브러리가 설치되지 않음")
                return False
            
            success = self._create_faiss_vectorstore(chunks)
            
            if success:
                # 3. 해시 및 차원 정보 저장
                self.save_source_hash()
                self.save_dimension_info(len(chunks))
                
                elapsed_time = time.time() - start_time
                logger.info(f"✅ {self.domain}: 벡터스토어 빌드 완료 ({elapsed_time:.2f}s)")
                return True
            else:
                logger.error(f"❌ {self.domain}: 벡터스토어 생성 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ {self.domain}: 벡터스토어 빌드 중 예외: {e}")
            return False
    
    def _create_faiss_vectorstore(self, chunks: List[TextChunk]) -> bool:
        """FAISS 벡터스토어 + BM25 인덱스 통합 생성"""
        try:
            # 텍스트와 메타데이터 추출
            texts = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            logger.info(f"🔄 {self.domain}: FAISS 벡터스토어 생성 중...")
            
            # FAISS 벡터스토어 생성
            vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # 차원 검증
            if hasattr(vectorstore, 'index'):
                actual_dim = vectorstore.index.d
                if actual_dim != self.embedding_dimension:
                    logger.warning(
                        f"⚠️ {self.domain} 생성된 벡터스토어 차원 불일치: "
                        f"예상={self.embedding_dimension}, 실제={actual_dim}"
                    )
                    self.embedding_dimension = actual_dim  # 실제 차원으로 업데이트
            
            # FAISS 저장
            vectorstore.save_local(
                folder_path=str(self.vectorstore_dir),
                index_name=self.index_name
            )
            
            # ✅ BM25 인덱스 생성 및 저장
            bm25_success = self._create_bm25_index(texts, metadatas)
            
            # 생성 확인
            faiss_file = self.vectorstore_dir / f"{self.index_name}.faiss"
            pkl_file = self.vectorstore_dir / f"{self.index_name}.pkl"
            bm25_file = self.vectorstore_dir / f"{self.index_name}.bm25"
            
            if faiss_file.exists() and pkl_file.exists():
                faiss_size = faiss_file.stat().st_size / (1024*1024)
                bm25_size = bm25_file.stat().st_size / (1024*1024) if bm25_file.exists() else 0
                
                logger.info(f"💾 {self.domain}: FAISS 저장됨 ({faiss_size:.1f}MB)")
                if bm25_success:
                    logger.info(f"💾 {self.domain}: BM25 저장됨 ({bm25_size:.1f}MB)")
                else:
                    logger.warning(f"⚠️ {self.domain}: BM25 생성 실패")
                
                return True
            else:
                logger.error(f"❌ {self.domain}: 벡터스토어 파일 생성 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ {self.domain}: FAISS 벡터스토어 생성 실패: {e}")
            return False
    
    def _create_bm25_index(self, texts: List[str], metadatas: List[Dict]) -> bool:
        """BM25 인덱스 생성 및 저장"""
        try:
            if not BM25_AVAILABLE:
                logger.warning(f"⚠️ {self.domain}: rank_bm25 라이브러리가 없어 BM25 인덱스 건너뜀")
                return False
            
            logger.info(f"🔍 {self.domain}: BM25 인덱스 생성 중...")
            
            # 텍스트 토큰화 (간단한 공백 기반)
            tokenized_texts = [text.lower().split() for text in texts]
            
            # BM25 인덱스 생성
            bm25_index = BM25Okapi(tokenized_texts)
            
            # BM25 데이터 패키징
            bm25_data = {
                'bm25_index': bm25_index,
                'texts': texts,
                'metadatas': metadatas,
                'tokenized_texts': tokenized_texts,
                'domain': self.domain,
                'embedding_model': self.embedding_model,  # 모델 정보 추가
                'embedding_dimension': self.embedding_dimension,  # 차원 정보 추가
                'created_at': datetime.now().isoformat(),
                'total_documents': len(texts)
            }
            
            # .bm25 파일로 저장
            bm25_file = self.vectorstore_dir / f"{self.index_name}.bm25"
            with open(bm25_file, 'wb') as f:
                pickle.dump(bm25_data, f)
            
            logger.info(f"✅ {self.domain}: BM25 인덱스 생성 완료 ({len(texts)}개 문서)")
            return True
            
        except Exception as e:
            logger.error(f"❌ {self.domain}: BM25 인덱스 생성 실패: {e}")
            return False
    
    def load_vectorstore(self) -> Optional[FAISS]:
        """생성된 벡터스토어 로드"""
        try:
            if not FAISS_AVAILABLE:
                logger.error("FAISS 라이브러리가 설치되지 않음")
                return None
            
            if not self.embeddings:
                logger.error(f"❌ {self.domain}: 임베딩 모델이 초기화되지 않음")
                return None
            
            vectorstore = FAISS.load_local(
                folder_path=str(self.vectorstore_dir),
                embeddings=self.embeddings,
                index_name=self.index_name,
                allow_dangerous_deserialization=True
            )
            
            # 차원 검증
            if hasattr(vectorstore, 'index'):
                actual_dim = vectorstore.index.d
                if actual_dim != self.embedding_dimension:
                    logger.warning(
                        f"⚠️ {self.domain} 로드된 벡터스토어 차원 불일치: "
                        f"예상={self.embedding_dimension}, 실제={actual_dim}"
                    )
            
            logger.info(f"📚 {self.domain}: 벡터스토어 로드 성공")
            return vectorstore
            
        except Exception as e:
            logger.error(f"❌ {self.domain}: 벡터스토어 로드 실패: {e}")
            return None
    
    def load_bm25_index(self) -> Optional[Dict]:
        """생성된 BM25 인덱스 로드"""
        try:
            bm25_file = self.vectorstore_dir / f"{self.index_name}.bm25"
            
            if not bm25_file.exists():
                logger.warning(f"⚠️ {self.domain}: BM25 파일이 없습니다")
                return None
            
            with open(bm25_file, 'rb') as f:
                bm25_data = pickle.load(f)
            
            # 모델 호환성 확인
            stored_model = bm25_data.get('embedding_model')
            if stored_model and stored_model != self.embedding_model:
                logger.warning(
                    f"⚠️ {self.domain} BM25의 임베딩 모델 불일치: "
                    f"저장됨({stored_model}) vs 현재({self.embedding_model})"
                )
            
            logger.info(f"📚 {self.domain}: BM25 인덱스 로드 성공")
            return bm25_data
            
        except Exception as e:
            logger.error(f"❌ {self.domain}: BM25 인덱스 로드 실패: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """벡터스토어 + BM25 통계 정보"""
        try:
            faiss_file = self.vectorstore_dir / f"{self.index_name}.faiss"
            pkl_file = self.vectorstore_dir / f"{self.index_name}.pkl"
            bm25_file = self.vectorstore_dir / f"{self.index_name}.bm25"
            dimension_file = self.vectorstore_dir / f"{self.index_name}_dimension_info.json"
            
            stats = {
                'domain': self.domain,
                'embedding_model': self.embedding_model,
                'embedding_dimension': self.embedding_dimension,
                'vectorstore_exists': faiss_file.exists() and pkl_file.exists(),
                'bm25_exists': bm25_file.exists(),
                'dimension_info_exists': dimension_file.exists(),
                'faiss_size_mb': faiss_file.stat().st_size / (1024*1024) if faiss_file.exists() else 0,
                'bm25_size_mb': bm25_file.stat().st_size / (1024*1024) if bm25_file.exists() else 0,
                'last_modified': datetime.fromtimestamp(faiss_file.stat().st_mtime).isoformat() if faiss_file.exists() else None,
                'source_dir': str(self.source_dir),
                'vectorstore_dir': str(self.vectorstore_dir)
            }
            
            # 차원 정보 추가
            if dimension_file.exists():
                try:
                    import json
                    with open(dimension_file, 'r') as f:
                        dimension_info = json.load(f)
                    stats['stored_dimension_info'] = dimension_info
                except Exception:
                    pass
            
            return stats
            
        except Exception as e:
            logger.error(f"통계 정보 생성 실패: {e}")
            return {'domain': self.domain, 'error': str(e)}


# ================================================================
# 개발/테스트 함수들
# ================================================================

def test_base_loader():
    """BaseLoader 기본 기능 테스트"""
    
    class TestLoader(BaseLoader):
        """테스트용 로더"""
        
        def process_domain_data(self) -> List[TextChunk]:
            return [
                TextChunk(
                    text="테스트 텍스트입니다.",
                    metadata={'test': True},
                    source_id="test.txt"
                )
            ]
    
    # 테스트 실행
    loader = TestLoader(
        domain="test",
        source_dir=Path("test_data"),
        vectorstore_dir=Path("test_vectorstore"),
        index_name="test_index"
    )
    
    print("✅ BaseLoader 테스트 완료")


if __name__ == "__main__":
    test_base_loader()
