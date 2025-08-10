#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - BaseLoader 기본 클래스

모든 도메인별 로더가 상속받는 기본 클래스:
- 공통 인터페이스 정의
- 해시 기반 증분 빌드
- 벡터스토어 자동 생성
- 에러 처리 및 로깅

주요 메서드:
- process_domain_data(): 각 로더에서 구현
- build_vectorstore(): FAISS 벡터스토어 생성
- validate_schema(): 스키마 검증 (선택적)
"""

import logging
import hashlib
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 프로젝트 모듈 임포트
from utils.textifier import TextChunk

# 외부 라이브러리 (선택적)
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OpenAIEmbeddings
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# 로깅 설정
logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """
    모든 도메인 로더의 기본 클래스
    
    공통 기능:
    - 도메인별 데이터 처리
    - FAISS 벡터스토어 생성
    - 해시 기반 증분 빌드
    - 에러 처리 및 로깅
    """
    def __init__(self, domain=None, loader_id=None, source_dir=None, vectorstore_dir=None, target_dir=None, index_name=None, schema_dir=None, **kwargs):
    """
    BaseLoader 초기화
    
    Args:
        domain: 도메인 이름 (예: "satisfaction") 
        loader_id: 로더 ID (domain과 동일, 호환성용)
        source_dir: 소스 데이터 디렉터리
        vectorstore_dir: 벡터스토어 출력 디렉터리  
        target_dir: 벡터스토어 출력 디렉터리 (vectorstore_dir과 동일, 호환성용)
        index_name: 인덱스 파일명
        schema_dir: 스키마 디렉터리 (선택적)
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
    
    # 디렉터리 생성
    self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✨ {self.domain.upper()} BaseLoader 초기화 완료")
    
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
            
            # 소스 디렉터리의 모든 파일 해시 계산
            for file_path in self.source_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix in self.get_supported_extensions():
                    hash_md5.update(str(file_path.stat().st_mtime).encode())
                    hash_md5.update(str(file_path.stat().st_size).encode())
            
            return hash_md5.hexdigest()[:16]
        except Exception as e:
            logger.warning(f"해시 계산 실패: {e}")
            return str(int(time.time()))  # 폴백: 타임스탬프
    
    def needs_rebuild(self) -> bool:
        """재빌드 필요 여부 확인"""
        try:
            # 벡터스토어 파일 존재 확인
            faiss_file = self.vectorstore_dir / f"{self.index_name}.faiss"
            pkl_file = self.vectorstore_dir / f"{self.index_name}.pkl"
            
            if not (faiss_file.exists() and pkl_file.exists()):
                logger.info(f"🔨 {self.domain}: 벡터스토어 파일이 없어서 새로 빌드")
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
            
            logger.info(f"✅ {self.domain}: 벡터스토어가 최신 상태")
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
    
    def build_vectorstore(self, force_rebuild: bool = False) -> bool:
        """
        벡터스토어 빌드 (증분 빌드 지원)
        
        Args:
            force_rebuild: 강제 재빌드 여부
            
        Returns:
            bool: 빌드 성공 여부
        """
        try:
            # 재빌드 필요성 확인
            if not force_rebuild and not self.needs_rebuild():
                logger.info(f"⏭️ {self.domain}: 이미 최신 벡터스토어 존재")
                return True
            
            logger.info(f"🔨 {self.domain}: 벡터스토어 빌드 시작...")
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
                # 3. 해시 저장
                self.save_source_hash()
                
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
        """FAISS 벡터스토어 생성 (내부 메서드)"""
        try:
            # 임베딩 모델 초기화
            embeddings = OpenAIEmbeddings()
            
            # 텍스트와 메타데이터 추출
            texts = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # FAISS 벡터스토어 생성
            vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas
            )
            
            # 저장
            vectorstore.save_local(
                folder_path=str(self.vectorstore_dir),
                index_name=self.index_name
            )
            
            # 생성 확인
            faiss_file = self.vectorstore_dir / f"{self.index_name}.faiss"
            pkl_file = self.vectorstore_dir / f"{self.index_name}.pkl"
            
            if faiss_file.exists() and pkl_file.exists():
                file_size = faiss_file.stat().st_size / (1024*1024)  # MB
                logger.info(f"💾 {self.domain}: 벡터스토어 저장됨 ({file_size:.1f}MB)")
                return True
            else:
                logger.error(f"❌ {self.domain}: 벡터스토어 파일 생성 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ {self.domain}: FAISS 벡터스토어 생성 실패: {e}")
            return False
    
    def load_vectorstore(self) -> Optional[FAISS]:
        """생성된 벡터스토어 로드"""
        try:
            if not FAISS_AVAILABLE:
                logger.error("FAISS 라이브러리가 설치되지 않음")
                return None
            
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(
                folder_path=str(self.vectorstore_dir),
                embeddings=embeddings,
                index_name=self.index_name,
                allow_dangerous_deserialization=True
            )
            
            logger.info(f"📚 {self.domain}: 벡터스토어 로드 성공")
            return vectorstore
            
        except Exception as e:
            logger.error(f"❌ {self.domain}: 벡터스토어 로드 실패: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """벡터스토어 통계 정보"""
        try:
            faiss_file = self.vectorstore_dir / f"{self.index_name}.faiss"
            pkl_file = self.vectorstore_dir / f"{self.index_name}.pkl"
            
            stats = {
                'domain': self.domain,
                'vectorstore_exists': faiss_file.exists() and pkl_file.exists(),
                'faiss_size_mb': faiss_file.stat().st_size / (1024*1024) if faiss_file.exists() else 0,
                'last_modified': datetime.fromtimestamp(faiss_file.stat().st_mtime).isoformat() if faiss_file.exists() else None,
                'source_dir': str(self.source_dir),
                'vectorstore_dir': str(self.vectorstore_dir)
            }
            
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
