"""
텍스트 변환 유틸리티
PDF/CSV/이미지를 벡터스토어용 텍스트로 변환 (빌드타임 전용)
"""

import os
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from io import StringIO
import hashlib

# PDF 처리
try:
    import PyPDF2
    import pdfplumber
    HAS_PDF_LIBS = True
except ImportError:
    HAS_PDF_LIBS = False
    logging.warning("PDF libraries not available. Install PyPDF2 and pdfplumber for PDF support")

# 이미지 처리 (OCR)
try:
    from PIL import Image
    import pytesseract
    HAS_OCR_LIBS = True
except ImportError:
    HAS_OCR_LIBS = False
    logging.warning("OCR libraries not available. Install Pillow and pytesseract for image support")

from .config import config
from .logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class TextChunk:
    """텍스트 청크 단위"""
    content: str
    metadata: Dict[str, Any]
    source_id: str
    chunk_index: int
    
class DocumentProcessor:
    """문서별 전용 프로세서"""
    
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.root_dir = Path(config.ROOT_DIR)
        
    def _create_chunks(self, text: str, source_id: str, base_metadata: Dict[str, Any]) -> List[TextChunk]:
        """텍스트를 청크 단위로 분할"""
        if len(text) <= self.chunk_size:
            return [TextChunk(
                content=text.strip(),
                metadata={**base_metadata, "chunk_type": "full"},
                source_id=source_id,
                chunk_index=0
            )]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 단어 경계에서 자르기
            if end < len(text):
                # 뒤에서부터 공백이나 구두점 찾기
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in ' \n\t.,!?;:':
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = {
                    **base_metadata,
                    "chunk_type": "partial",
                    "char_start": start,
                    "char_end": end
                }
                
                chunks.append(TextChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    source_id=source_id,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
            
            # 오버랩 적용
            start = max(start + 1, end - self.overlap_size)
            
        return chunks

class PDFProcessor(DocumentProcessor):
    """PDF 문서 처리"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not HAS_PDF_LIBS:
            raise ImportError("PDF processing requires PyPDF2 and pdfplumber")
    
    def process(self, file_path: Path) -> List[TextChunk]:
        """PDF 파일을 텍스트 청크로 변환"""
        logger.info(f"Processing PDF: {file_path}")
        
        try:
            chunks = []
            source_id = str(file_path.relative_to(self.root_dir))
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        if not text or not text.strip():
                            continue
                            
                        # 페이지별 메타데이터
                        page_metadata = {
                            "source_type": "pdf",
                            "page_number": page_num,
                            "total_pages": len(pdf.pages),
                            "file_path": str(file_path)
                        }
                        
                        # 페이지 단위로 청크화
                        page_source_id = f"{source_id}#p{page_num}"
                        page_chunks = self._create_chunks(text, page_source_id, page_metadata)
                        chunks.extend(page_chunks)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process page {page_num} of {file_path}: {e}")
                        continue
            
            logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {e}")
            return []

class CSVProcessor(DocumentProcessor):
    """CSV 파일 처리"""
    
    def process(self, file_path: Path, schema_path: Optional[Path] = None) -> List[TextChunk]:
        """CSV 파일을 텍스트 청크로 변환"""
        logger.info(f"Processing CSV: {file_path}")
        
        try:
            chunks = []
            source_id = str(file_path.relative_to(self.root_dir))
            
            # 스키마 로드 (있는 경우)
            schema = None
            if schema_path and schema_path.exists():
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                # CSV 방언 자동 감지
                sample = f.read(1024)
                f.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(f, delimiter=delimiter)
                
                # 헤더 정규화
                fieldnames = [name.strip() for name in reader.fieldnames] if reader.fieldnames else []
                
                for row_num, row in enumerate(reader, 1):
                    try:
                        # 행을 텍스트로 변환
                        row_text = self._row_to_text(row, fieldnames, schema)
                        if not row_text.strip():
                            continue
                        
                        # 행별 메타데이터
                        row_metadata = {
                            "source_type": "csv",
                            "row_number": row_num,
                            "fieldnames": fieldnames,
                            "file_path": str(file_path)
                        }
                        
                        # 행 단위로 청크화 (보통 1행 = 1청크)
                        row_source_id = f"{source_id}#row{row_num}"
                        row_chunks = self._create_chunks(row_text, row_source_id, row_metadata)
                        chunks.extend(row_chunks)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process row {row_num} of {file_path}: {e}")
                        continue
            
            logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process CSV {file_path}: {e}")
            return []
    
    def _row_to_text(self, row: Dict[str, str], fieldnames: List[str], schema: Optional[Dict] = None) -> str:
        """CSV 행을 검색 가능한 텍스트로 변환"""
        text_parts = []
        
        for field in fieldnames:
            value = row.get(field, '').strip()
            if not value:
                continue
                
            # 스키마 기반 필드 타입 판단
            field_info = None
            if schema and 'properties' in schema:
                field_info = schema['properties'].get(field, {})
            
            # 필드명과 값을 자연어 형태로 결합
            if field_info and field_info.get('description'):
                text_parts.append(f"{field_info['description']}: {value}")
            else:
                text_parts.append(f"{field}: {value}")
        
        return " | ".join(text_parts)

class ImageProcessor(DocumentProcessor):
    """이미지 OCR 처리"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not HAS_OCR_LIBS:
            raise ImportError("Image processing requires Pillow and pytesseract")
    
    def process(self, file_path: Path, cache_dir: Optional[Path] = None) -> List[TextChunk]:
        """이미지를 OCR로 텍스트 청크 변환"""
        logger.info(f"Processing image with OCR: {file_path}")
        
        try:
            # 캐시 확인
            if cache_dir:
                cache_file = cache_dir / f"{file_path.stem}_ocr.txt"
                if cache_file.exists():
                    logger.info(f"Using cached OCR result: {cache_file}")
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        ocr_text = f.read()
                else:
                    ocr_text = self._extract_text_from_image(file_path)
                    # 캐시 저장
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(ocr_text)
            else:
                ocr_text = self._extract_text_from_image(file_path)
            
            if not ocr_text.strip():
                logger.warning(f"No text extracted from {file_path}")
                return []
            
            # 이미지 메타데이터
            source_id = str(file_path.relative_to(self.root_dir))
            metadata = {
                "source_type": "image_ocr",
                "file_path": str(file_path),
                "ocr_method": "tesseract"
            }
            
            chunks = self._create_chunks(ocr_text, source_id, metadata)
            logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process image {file_path}: {e}")
            return []
    
    def _extract_text_from_image(self, file_path: Path) -> str:
        """이미지에서 텍스트 추출"""
        try:
            with Image.open(file_path) as img:
                # 한국어 OCR 설정
                custom_config = r'--oem 3 --psm 6 -l kor+eng'
                text = pytesseract.image_to_string(img, config=custom_config)
                return text.strip()
        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {e}")
            return ""

class TextProcessor(DocumentProcessor):
    """일반 텍스트 파일 처리"""
    
    def process(self, file_path: Path) -> List[TextChunk]:
        """텍스트 파일을 청크로 변환"""
        logger.info(f"Processing text file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Empty text file: {file_path}")
                return []
            
            source_id = str(file_path.relative_to(self.root_dir))
            metadata = {
                "source_type": "text",
                "file_path": str(file_path)
            }
            
            chunks = self._create_chunks(content, source_id, metadata)
            logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process text file {file_path}: {e}")
            return []

class Textifier:
    """통합 텍스트 변환기"""
    
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.root_dir = Path(config.ROOT_DIR)
        self.cache_dir = self.root_dir / "cache"
        
        # 프로세서 초기화
        self.processors = {
            'pdf': PDFProcessor(chunk_size=chunk_size, overlap_size=overlap_size),
            'csv': CSVProcessor(chunk_size=chunk_size, overlap_size=overlap_size),
            'txt': TextProcessor(chunk_size=chunk_size, overlap_size=overlap_size),
        }
        
        # OCR 프로세서는 선택적 초기화
        if HAS_OCR_LIBS:
            self.processors.update({
                'png': ImageProcessor(chunk_size=chunk_size, overlap_size=overlap_size),
                'jpg': ImageProcessor(chunk_size=chunk_size, overlap_size=overlap_size),
                'jpeg': ImageProcessor(chunk_size=chunk_size, overlap_size=overlap_size),
            })
    
    def process_file(self, file_path: Path, schema_path: Optional[Path] = None) -> List[TextChunk]:
        """파일을 텍스트 청크로 변환"""
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        file_ext = file_path.suffix.lower().lstrip('.')
        processor = self.processors.get(file_ext)
        
        if not processor:
            logger.warning(f"No processor for file type: {file_ext}")
            return []
        
        try:
            # CSV의 경우 스키마 전달
            if file_ext == 'csv' and isinstance(processor, CSVProcessor):
                return processor.process(file_path, schema_path)
            # 이미지의 경우 캐시 디렉토리 전달
            elif file_ext in ['png', 'jpg', 'jpeg'] and isinstance(processor, ImageProcessor):
                return processor.process(file_path, self.cache_dir)
            else:
                return processor.process(file_path)
                
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return []
    
    def process_directory(self, dir_path: Path, file_pattern: str = "*", 
                         schema_dir: Optional[Path] = None) -> Dict[str, List[TextChunk]]:
        """디렉토리 내 파일들을 일괄 처리"""
        if not dir_path.exists():
            logger.error(f"Directory not found: {dir_path}")
            return {}
        
        results = {}
        
        for file_path in dir_path.glob(file_pattern):
            if file_path.is_file():
                # 스키마 파일 찾기
                schema_path = None
                if schema_dir and file_path.suffix.lower() == '.csv':
                    schema_name = f"{file_path.stem}.schema.json"
                    potential_schema = schema_dir / schema_name
                    if potential_schema.exists():
                        schema_path = potential_schema
                
                chunks = self.process_file(file_path, schema_path)
                if chunks:
                    results[str(file_path.relative_to(self.root_dir))] = chunks
        
        return results
    
    def get_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산 (변경 감지용)"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def get_supported_extensions(self) -> List[str]:
        """지원하는 파일 확장자 목록"""
        return list(self.processors.keys())

# 전역 인스턴스
textifier = Textifier(
    chunk_size=config.CHUNK_SIZE,
    overlap_size=config.OVERLAP_SIZE
)
