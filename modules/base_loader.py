"""
베이스 로더 클래스
모든 도메인별 로더의 공통 기능 제공 (해시 기반 증분 빌드, 스키마 검증, FAISS 인덱스 생성)
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from utils.config import config
from utils.textifier import textifier, TextChunk
from utils.embedder import embedder
from utils.index_manager import index_manager
from utils.logging_utils import get_logger
from utils.contracts import Citation

logger = get_logger(__name__)

@dataclass
class LoaderMetadata:
    """로더 메타데이터"""
    loader_id: str
    source_dir: str
    target_dir: str
    last_build: datetime
    file_hashes: Dict[str, str]
    total_files: int
    total_chunks: int
    schema_version: str = "1.0"
    
class BaseLoader(ABC):
    """모든 로더의 추상 베이스 클래스"""
    
    def __init__(self, 
                 loader_id: str,
                 source_dir: str, 
                 target_dir: str,
                 schema_dir: Optional[str] = None):
        """
        Args:
            loader_id: 로더 고유 식별자 (예: "cyber", "satisfaction")
            source_dir: 소스 데이터 디렉토리 (data/ 기준 상대경로)
            target_dir: 벡터스토어 저장 디렉토리 (vectorstores/ 기준 상대경로)
            schema_dir: 스키마 디렉토리 (schemas/ 기준 상대경로, 옵션)
        """
        self.loader_id = loader_id
        self.root_dir = Path(config.ROOT_DIR)
        
        # 절대경로 변환
        self.source_dir = self.root_dir / source_dir
        self.target_dir = self.root_dir / target_dir
        self.schema_dir = self.root_dir / schema_dir if schema_dir else None
        
        # 메타데이터 파일 경로
        self.metadata_file = self.target_dir / f"{loader_id}_metadata.json"
        
        # 디렉토리 생성
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__} for {loader_id}")
    
    @abstractmethod
    def get_file_patterns(self) -> List[str]:
        """처리할 파일 패턴 목록 반환"""
        pass
    
    @abstractmethod
    def process_domain_data(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        도메인별 특화 처리 (템플릿 적용, 데이터 변환 등)
        
        Args:
            chunks: textifier로 추출된 원본 청크들
            
        Returns:
            처리된 청크들 (템플릿 적용, 메타데이터 보강 등)
        """
        pass
    
    def get_schema_path(self, filename: str) -> Optional[Path]:
        """파일명에 해당하는 스키마 파일 경로 반환"""
        if not self.schema_dir:
            return None
            
        # CSV 파일의 경우 스키마 확인
        if filename.endswith('.csv'):
            schema_name = f"{Path(filename).stem}.schema.json"
            schema_path = self.schema_dir / schema_name
            if schema_path.exists():
                return schema_path
        return None
    
    def validate_schema(self, file_path: Path, schema_path: Path) -> bool:
        """CSV 파일의 스키마 검증"""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            
            # CSV 파일 헤더 확인
            import pandas as pd
            df = pd.read_csv(file_path, nrows=0)  # 헤더만 읽기
            headers = [col.strip() for col in df.columns]
            
            # 필수 컬럼 검증
            if 'required' in schema:
                missing_cols = set(schema['required']) - set(headers)
                if missing_cols:
                    logger.error(f"Missing required columns in {file_path}: {missing_cols}")
                    return False
            
            # 컬럼 타입 검증 (기본적인 체크)
            if 'properties' in schema:
                for col in headers:
                    if col not in schema['properties']:
                        logger.warning(f"Unknown column '{col}' in {file_path}")
            
            logger.info(f"Schema validation passed for {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Schema validation failed for {file_path}: {e}")
            return False
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def load_metadata(self) -> Optional[LoaderMetadata]:
        """기존 메타데이터 로드"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['last_build'] = datetime.fromisoformat(data['last_build'])
                    return LoaderMetadata(**data)
        except Exception as e:
            logger.warning(f"Failed to load metadata from {self.metadata_file}: {e}")
        return None
    
    def save_metadata(self, metadata: LoaderMetadata):
        """메타데이터 저장"""
        try:
            data = asdict(metadata)
            data['last_build'] = metadata.last_build.isoformat()
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Metadata saved to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def get_changed_files(self) -> tuple[List[Path], Set[str]]:
        """변경된 파일 목록과 삭제된 파일 목록 반환"""
        if not self.source_dir.exists():
            logger.error(f"Source directory not found: {self.source_dir}")
            return [], set()
        
        # 현재 파일들과 해시 계산
        current_files = []
        current_hashes = {}
        
        for pattern in self.get_file_patterns():
            for file_path in self.source_dir.glob(pattern):
                if file_path.is_file():
                    current_files.append(file_path)
                    file_hash = self.calculate_file_hash(file_path)
                    current_hashes[str(file_path.relative_to(self.root_dir))] = file_hash
        
        # 기존 메타데이터와 비교
        metadata = self.load_metadata()
        if not metadata:
            # 첫 빌드: 모든 파일이 변경된 것으로 간주
            logger.info("No existing metadata found. Processing all files.")
            return current_files, set()
        
        changed_files = []
        old_hashes = metadata.file_hashes
        
        # 변경/신규 파일 찾기
        for file_path in current_files:
            rel_path = str(file_path.relative_to(self.root_dir))
            current_hash = current_hashes[rel_path]
            
            if rel_path not in old_hashes or old_hashes[rel_path] != current_hash:
                changed_files.append(file_path)
                logger.info(f"File changed: {rel_path}")
        
        # 삭제된 파일 찾기
        current_rel_paths = set(current_hashes.keys())
        old_rel_paths = set(old_hashes.keys())
        deleted_files = old_rel_paths - current_rel_paths
        
        if deleted_files:
            logger.info(f"Deleted files: {deleted_files}")
        
        return changed_files, deleted_files
    
    def extract_text_chunks(self, file_paths: List[Path]) -> List[TextChunk]:
        """파일들에서 텍스트 청크 추출"""
        all_chunks = []
        
        for file_path in file_paths:
            try:
                # 스키마 검증 (CSV의 경우)
                schema_path = self.get_schema_path(file_path.name)
                if schema_path and not self.validate_schema(file_path, schema_path):
                    logger.error(f"Schema validation failed for {file_path}. Skipping.")
                    continue
                
                # 텍스트 추출
                chunks = textifier.process_file(file_path, schema_path)
                if chunks:
                    all_chunks.extend(chunks)
                    logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
                else:
                    logger.warning(f"No chunks extracted from {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                continue
        
        return all_chunks
    
    def build_index(self, force_rebuild: bool = False) -> bool:
        """인덱스 빌드 (증분 또는 전체)"""
        logger.info(f"Starting index build for {self.loader_id} (force_rebuild={force_rebuild})")
        
        try:
            if force_rebuild:
                # 전체 재빌드
                file_patterns = self.get_file_patterns()
                all_files = []
                for pattern in file_patterns:
                    all_files.extend(self.source_dir.glob(pattern))
                changed_files = [f for f in all_files if f.is_file()]
                deleted_files = set()
                logger.info(f"Force rebuild: processing {len(changed_files)} files")
            else:
                # 증분 빌드
                changed_files, deleted_files = self.get_changed_files()
                if not changed_files and not deleted_files:
                    logger.info("No changes detected. Skipping build.")
                    return True
            
            # 텍스트 추출
            raw_chunks = self.extract_text_chunks(changed_files)
            if not raw_chunks and not deleted_files:
                logger.warning("No chunks to process and no files deleted.")
                return True
            
            # 도메인별 처리
            processed_chunks = self.process_domain_data(raw_chunks)
            
            # 벡터스토어 업데이트
            if processed_chunks or deleted_files:
                success = self._update_vectorstore(processed_chunks, deleted_files)
                if not success:
                    return False
            
            # 메타데이터 업데이트
            self._update_metadata(changed_files)
            
            logger.info(f"Index build completed for {self.loader_id}")
            return True
            
        except Exception as e:
            logger.error(f"Index build failed for {self.loader_id}: {e}")
            return False
    
    def _update_vectorstore(self, chunks: List[TextChunk], deleted_files: Set[str]) -> bool:
        """벡터스토어 업데이트"""
        try:
            # 기존 인덱스 로드 또는 새 인덱스 생성
            index_name = f"{self.loader_id}_index"
            
            if chunks:
                # 텍스트와 메타데이터 분리
                texts = [chunk.content for chunk in chunks]
                metadatas = [chunk.metadata for chunk in chunks]
                
                # 임베딩 생성 및 인덱스 업데이트
                embeddings_array = embedder.embed_texts(texts)
                
                # IndexManager를 통한 저장
                index_manager.save_index(
                    index_name=index_name,
                    embeddings=embeddings_array,
                    texts=texts,
                    metadatas=metadatas,
                    save_path=self.target_dir
                )
                
                logger.info(f"Updated vectorstore with {len(chunks)} chunks")
            
            # 삭제된 파일 처리 (필요시 인덱스에서 제거)
            if deleted_files:
                # TODO: 특정 소스의 청크들을 인덱스에서 제거하는 로직
                logger.warning(f"File deletion handling not implemented yet: {deleted_files}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update vectorstore: {e}")
            return False
    
    def _update_metadata(self, processed_files: List[Path]):
        """메타데이터 업데이트"""
        try:
            # 현재 모든 파일의 해시 계산
            current_hashes = {}
            total_files = 0
            
            for pattern in self.get_file_patterns():
                for file_path in self.source_dir.glob(pattern):
                    if file_path.is_file():
                        rel_path = str(file_path.relative_to(self.root_dir))
                        current_hashes[rel_path] = self.calculate_file_hash(file_path)
                        total_files += 1
            
            # 총 청크 수 계산 (인덱스 매니저에서 가져오기)
            total_chunks = 0  # TODO: IndexManager에서 청크 수 조회
            
            metadata = LoaderMetadata(
                loader_id=self.loader_id,
                source_dir=str(self.source_dir.relative_to(self.root_dir)),
                target_dir=str(self.target_dir.relative_to(self.root_dir)),
                last_build=datetime.now(),
                file_hashes=current_hashes,
                total_files=total_files,
                total_chunks=total_chunks
            )
            
            self.save_metadata(metadata)
            
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """로더 상태 정보 반환"""
        metadata = self.load_metadata()
        
        if not metadata:
            return {
                "loader_id": self.loader_id,
                "status": "not_built",
                "last_build": None,
                "total_files": 0,
                "total_chunks": 0
            }
        
        return {
            "loader_id": self.loader_id,
            "status": "ready",
            "last_build": metadata.last_build.isoformat(),
            "total_files": metadata.total_files,
            "total_chunks": metadata.total_chunks,
            "schema_version": metadata.schema_version
        }
    
    def cleanup(self):
        """리소스 정리"""
        logger.info(f"Cleaning up {self.__class__.__name__}")
        # 필요시 임시 파일, 캐시 등 정리
