# utils/textifier.py
"""
텍스트 변환 유틸리티 (Pydantic v2 호환)
PDF, CSV, 이미지 등을 텍스트로 변환
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import chardet
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

# 로깅 설정
logger = logging.getLogger(__name__)

class TextChunk(BaseModel):
    """
    텍스트 청크 모델 (Pydantic v2 호환)
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow'
    )
    
    text: str = Field(..., description="청크 텍스트")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")
    source_id: str = Field(..., description="소스 ID")
    chunk_index: int = Field(0, description="청크 인덱스")
    
    @property
    def embedding_text(self) -> str:
        """임베딩용 텍스트 생성"""
        meta_str = " ".join([f"{k}:{v}" for k, v in self.metadata.items() if k != 'source'])
        return f"{meta_str} {self.text}".strip()
    
    def to_document(self) -> Dict[str, Any]:
        """LangChain Document 형식으로 변환"""
        return {
            "page_content": self.text,
            "metadata": {
                **self.metadata,
                "source_id": self.source_id,
                "chunk_index": self.chunk_index
            }
        }
    
    @classmethod
    def from_document(cls, doc: Any, source_id: Optional[str] = None, chunk_index: int = 0) -> "TextChunk":
        """LangChain Document에서 TextChunk로 변환"""
        # Document 객체 처리
        if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
            text = doc.page_content
            metadata = doc.metadata if doc.metadata else {}
        # dict 형태 처리
        elif isinstance(doc, dict):
            text = doc.get('page_content', doc.get('text', ''))
            metadata = doc.get('metadata', {})
        # 문자열 처리
        elif isinstance(doc, str):
            text = doc
            metadata = {}
        else:
            text = str(doc)
            metadata = {}
        
        # source_id 결정
        if source_id is None:
            source_id = metadata.get('source_id', f'doc_{chunk_index}')
        
        return cls(
            text=text,
            metadata=metadata,
            source_id=source_id,
            chunk_index=chunk_index
        )

class Textifier:
    """텍스트 변환 클래스"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("cache/textifier")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Textifier 초기화: cache_dir={self.cache_dir}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def process_pdf(self, file_path: Path) -> List[TextChunk]:
        """PDF 파일 처리"""
        try:
            from PyPDF2 import PdfReader
            
            chunks = []
            reader = PdfReader(str(file_path))
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    chunk = TextChunk(
                        text=text,
                        metadata={
                            "source": str(file_path),
                            "page": page_num + 1,
                            "type": "pdf",
                            "filename": file_path.name
                        },
                        source_id=f"{file_path.stem}_page_{page_num + 1}",
                        chunk_index=page_num
                    )
                    chunks.append(chunk)
            
            logger.info(f"✅ PDF 처리 완료: {file_path.name} ({len(chunks)}개 청크)")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ PDF 처리 실패: {file_path.name} - {e}")
            # 에러 시 더미 청크 반환
            return [TextChunk(
                text=f"PDF 파일 '{file_path.name}'을 처리할 수 없습니다.",
                metadata={"source": str(file_path), "type": "pdf", "error": str(e)},
                source_id=f"{file_path.stem}_error",
                chunk_index=0
            )]
    
    def process_csv(self, file_path: Path) -> List[TextChunk]:
        """CSV 파일 처리"""
        try:
            # 인코딩 자동 감지
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
            
            df = pd.read_csv(file_path, encoding=encoding)
            
            chunks = []
            chunk_size = 10  # 10행씩 청크
            
            for i in range(0, len(df), chunk_size):
                chunk_df = df.iloc[i:i + chunk_size]
                text = chunk_df.to_string(index=False)
                
                chunk = TextChunk(
                    text=text,
                    metadata={
                        "source": str(file_path),
                        "rows": f"{i+1}-{min(i+chunk_size, len(df))}",
                        "type": "csv",
                        "filename": file_path.name,
                        "columns": list(df.columns)
                    },
                    source_id=f"{file_path.stem}_rows_{i+1}_{min(i+chunk_size, len(df))}",
                    chunk_index=i // chunk_size
                )
                chunks.append(chunk)
            
            logger.info(f"✅ CSV 처리 완료: {file_path.name} ({len(chunks)}개 청크)")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ CSV 처리 실패: {file_path.name} - {e}")
            return [TextChunk(
                text=f"CSV 파일 '{file_path.name}'을 처리할 수 없습니다.",
                metadata={"source": str(file_path), "type": "csv", "error": str(e)},
                source_id=f"{file_path.stem}_error",
                chunk_index=0
            )]
    
    def process_text(self, file_path: Path) -> List[TextChunk]:
        """텍스트 파일 처리"""
        try:
            # 인코딩 자동 감지
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
            
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # 텍스트를 적절한 크기로 분할
            max_chunk_size = 1000
            chunks = []
            
            if len(content) <= max_chunk_size:
                chunks.append(TextChunk(
                    text=content,
                    metadata={
                        "source": str(file_path),
                        "type": "text",
                        "filename": file_path.name
                    },
                    source_id=f"{file_path.stem}",
                    chunk_index=0
                ))
            else:
                # 문단 단위로 분할
                paragraphs = content.split('\n\n')
                current_chunk = []
                current_size = 0
                chunk_index = 0
                
                for para in paragraphs:
                    if current_size + len(para) > max_chunk_size and current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        chunks.append(TextChunk(
                            text=chunk_text,
                            metadata={
                                "source": str(file_path),
                                "type": "text",
                                "filename": file_path.name,
                                "part": chunk_index + 1
                            },
                            source_id=f"{file_path.stem}_part_{chunk_index + 1}",
                            chunk_index=chunk_index
                        ))
                        current_chunk = [para]
                        current_size = len(para)
                        chunk_index += 1
                    else:
                        current_chunk.append(para)
                        current_size += len(para)
                
                # 마지막 청크
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        metadata={
                            "source": str(file_path),
                            "type": "text",
                            "filename": file_path.name,
                            "part": chunk_index + 1
                        },
                        source_id=f"{file_path.stem}_part_{chunk_index + 1}",
                        chunk_index=chunk_index
                    ))
            
            logger.info(f"✅ 텍스트 처리 완료: {file_path.name} ({len(chunks)}개 청크)")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ 텍스트 처리 실패: {file_path.name} - {e}")
            return [TextChunk(
                text=f"텍스트 파일 '{file_path.name}'을 처리할 수 없습니다.",
                metadata={"source": str(file_path), "type": "text", "error": str(e)},
                source_id=f"{file_path.stem}_error",
                chunk_index=0
            )]
    
    def process_file(self, file_path: Path) -> List[TextChunk]:
        """파일 확장자에 따라 적절한 처리기 선택"""
        if not file_path.exists():
            logger.warning(f"파일이 존재하지 않습니다: {file_path}")
            return []
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.process_pdf(file_path)
        elif suffix == '.csv':
            return self.process_csv(file_path)
        elif suffix in ['.txt', '.md', '.text']:
            return self.process_text(file_path)
        else:
            logger.warning(f"지원하지 않는 파일 형식: {suffix}")
            return []

# 싱글톤 인스턴스
_textifier_instance = None

def get_textifier(cache_dir: Path = None) -> Textifier:
    """Textifier 싱글톤 인스턴스 반환"""
    global _textifier_instance
    if _textifier_instance is None:
        _textifier_instance = Textifier(cache_dir)
    return _textifier_instance
