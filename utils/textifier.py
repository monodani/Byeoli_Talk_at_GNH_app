#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - utils/textifier.py

문서 텍스트화 및 청킹 유틸리티:
- PDF, CSV, 이미지 → 구조화된 텍스트 변환
- BaseLoader 호환성 보장
- 원본 메타데이터 보존
- 검색 최적화된 청킹

주요 클래스:
- TextChunk: 텍스트 청크 표준 모델
- DocumentProcessor: 문서 처리 기본 클래스  
- PDFProcessor: PDF 텍스트 추출
- CSVProcessor: CSV 데이터 처리
- ImageProcessor: OCR 텍스트 추출
"""

import csv
import json
import logging
import hashlib
import re
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field

# 외부 라이브러리
import tiktoken
from pypdf import PdfReader
from PIL import Image
import pytesseract

# 로깅 설정
logger = logging.getLogger(__name__)

# ================================================================
# 1. TextChunk 핵심 모델 (모든 로더/핸들러에서 사용)
# ================================================================

@dataclass
class TextChunk:
    """
    텍스트 청크 표준 모델
    
    모든 문서 처리기에서 공통으로 사용하는 텍스트 단위
    """
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_id: str = ""
    chunk_index: int = 0
    
    def __post_init__(self):
        """후처리: 기본 메타데이터 설정"""
        if not self.source_id and 'source_file' in self.metadata:
            self.source_id = self.metadata['source_file']
        
        # 텍스트 길이 및 해시 자동 계산
        self.metadata.setdefault('text_length', len(self.text))
        self.metadata.setdefault('text_hash', self._calculate_hash())
        self.metadata.setdefault('created_at', datetime.now(timezone.utc).isoformat())
    
    def _calculate_hash(self) -> str:
        """텍스트 내용 해시 계산 (중복 체크용)"""
        return hashlib.md5(self.text.encode('utf-8')).hexdigest()[:16]
    
    def estimate_tokens(self, model: str = "gpt-4o-mini") -> int:
        """토큰 수 추정"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(self.text))
        except Exception:
            # fallback: 대략 4글자 = 1토큰
            return len(self.text) // 4
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환 (직렬화용)"""
        return {
            'text': self.text,
            'metadata': self.metadata,
            'source_id': self.source_id,
            'chunk_index': self.chunk_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextChunk':
        """딕셔너리에서 복원"""
        return cls(
            text=data['text'],
            metadata=data.get('metadata', {}),
            source_id=data.get('source_id', ''),
            chunk_index=data.get('chunk_index', 0)
        )


# ================================================================
# 2. 문서 처리기 기본 클래스
# ================================================================

class DocumentProcessor(ABC):
    """문서 처리기 기본 추상 클래스"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.chunk_size = 1000  # 기본 청크 크기
        self.chunk_overlap = 100  # 기본 오버랩
    
    @abstractmethod
    def process(self, file_path: Path, **kwargs) -> List[TextChunk]:
        """파일을 처리하여 TextChunk 리스트 반환"""
        pass
    
    def _create_chunks(self, text: str, source_id: str, base_metadata: Dict[str, Any]) -> List[TextChunk]:
        """텍스트를 청크로 분할"""
        if not text.strip():
            return []
        
        chunks = []
        
        # 간단한 청크 분할 (문장 단위 고려)
        sentences = self._split_sentences(text)
        
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence) // 4  # 근사치
            
            # 청크 크기 초과 시 새 청크 생성
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_metadata = {
                    **base_metadata,
                    'chunk_index': chunk_index,
                    'token_count': current_tokens
                }
                
                chunk = TextChunk(
                    text=current_chunk.strip(),
                    metadata=chunk_metadata,
                    source_id=f"{source_id}#chunk{chunk_index}",
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                
                # 오버랩 처리
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = overlap_text + " " + sentence
                current_tokens = len(current_chunk) // 4
                chunk_index += 1
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # 마지막 청크 처리
        if current_chunk.strip():
            chunk_metadata = {
                **base_metadata,
                'chunk_index': chunk_index,
                'token_count': current_tokens
            }
            
            chunk = TextChunk(
                text=current_chunk.strip(),
                metadata=chunk_metadata,
                source_id=f"{source_id}#chunk{chunk_index}",
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """문장 단위로 텍스트 분할"""
        # 한국어/영어 문장 분할 패턴
        sentence_pattern = r'[.!?]+\s+|[。！？]+\s*'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]


# ================================================================
# 3. PDF 처리기
# ================================================================

class PDFProcessor(DocumentProcessor):
    """PDF 문서 처리기"""
    
    def process(self, file_path: Path, **kwargs) -> List[TextChunk]:
        """PDF 파일을 처리하여 TextChunk 리스트 반환"""
        logger.info(f"Processing PDF: {file_path}")
        
        try:
            return self.process_file(file_path)
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {e}")
            return []
    
    def process_file(self, pdf_path: Path) -> List[TextChunk]:
        """PDF 파일 처리 (기존 메서드 이름 호환성)"""
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                
                logger.info(f"PDF pages: {len(reader.pages)}")
                
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if not text.strip():
                            logger.warning(f"Empty text on page {page_num}")
                            continue
                        
                        # 페이지별 메타데이터
                        page_metadata = {
                            'source_type': 'pdf',
                            'source_file': pdf_path.name,
                            'file_path': str(pdf_path),
                            'page_number': page_num,
                            'total_pages': len(reader.pages)
                        }
                        
                        # 페이지별 청크 생성
                        source_id = str(pdf_path.relative_to(self.root_dir)) + f"#page{page_num}"
                        page_chunks = self._create_chunks(text, source_id, page_metadata)
                        chunks.extend(page_chunks)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process page {page_num} of {pdf_path}: {e}")
                        continue
                
        except Exception as e:
            logger.error(f"Failed to read PDF {pdf_path}: {e}")
            return []
        
        logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
        return chunks
    
    def extract_text_by_page(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """페이지별 텍스트 추출 (메타데이터 포함)"""
        pages = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    
                    pages.append({
                        'page_number': page_num,
                        'text': text,
                        'char_count': len(text),
                        'word_count': len(text.split()) if text else 0
                    })
        
        except Exception as e:
            logger.error(f"Failed to extract PDF pages {pdf_path}: {e}")
        
        return pages


# ================================================================
# 4. CSV 처리기 
# ================================================================

class CSVProcessor(DocumentProcessor):
    """CSV 파일 처리 - BaseLoader 호환성 강화"""
    
    def process(self, file_path: Path, schema_path: Optional[Path] = None) -> List[TextChunk]:
        """CSV 파일을 텍스트 청크로 변환 (원본 row_data 보존)"""
        logger.info(f"Processing CSV: {file_path}")
        
        try:
            chunks = []
            source_id = str(file_path.relative_to(self.root_dir))
            
            # 스키마 로드 (있는 경우)
            schema = None
            if schema_path and schema_path.exists():
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
            
            # 인코딩 자동 감지 및 다중 시도
            content = self._read_csv_with_encoding_detection(file_path)
            if not content:
                logger.error(f"Failed to read CSV file with any encoding: {file_path}")
                return []
            
            # CSV 파싱
            reader = csv.DictReader(StringIO(content))
            
            # 헤더 정규화
            fieldnames = [name.strip() for name in reader.fieldnames] if reader.fieldnames else []
            if not fieldnames:
                logger.warning(f"No fieldnames found in CSV: {file_path}")
                return []
            
            logger.info(f"CSV fieldnames: {fieldnames}")
            
            for row_num, row in enumerate(reader, 1):
                try:
                    # 원본 row 데이터 정리 (strip 적용)
                    clean_row = {key.strip(): str(value).strip() for key, value in row.items() if key}
                    
                    # 빈 행 스킵
                    if not any(clean_row.values()):
                        continue
                    
                    # 행을 검색 가능한 텍스트로 변환
                    row_text = self._row_to_text(clean_row, fieldnames, schema)
                    if not row_text.strip():
                        logger.warning(f"Empty text generated for row {row_num} in {file_path}")
                        continue
                    
                    # 행별 메타데이터 (★ 핵심: 원본 row_data 저장)
                    row_metadata = {
                        "source_type": "csv",
                        "source_file": file_path.name,
                        "row_number": row_num,
                        "fieldnames": fieldnames,
                        "file_path": str(file_path),
                        "row_data": clean_row,  # ★ 원본 딕셔너리 저장
                        "text_representation": row_text  # 검색용 텍스트도 별도 저장
                    }
                    
                    # 스키마 정보도 메타데이터에 포함
                    if schema:
                        row_metadata["schema_applied"] = True
                        row_metadata["schema_file"] = str(schema_path) if schema_path else None
                    
                    # 행 단위로 청크화 (보통 1행 = 1청크)
                    row_source_id = f"{source_id}#row{row_num}"
                    
                    chunk = TextChunk(
                        text=row_text,
                        metadata=row_metadata,
                        source_id=row_source_id,
                        chunk_index=row_num - 1
                    )
                    chunks.append(chunk)
                    
                except Exception as e:
                    logger.warning(f"Failed to process row {row_num} of {file_path}: {e}")
                    # 개별 행 실패 시에도 계속 진행
                    continue
            
            logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process CSV {file_path}: {e}")
            return []
    
    def _read_csv_with_encoding_detection(self, file_path: Path) -> Optional[str]:
        """다중 인코딩 시도로 CSV 파일 읽기"""
        encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    logger.info(f"Successfully read {file_path} with encoding: {encoding}")
                    return content
            except UnicodeDecodeError:
                logger.debug(f"Failed to read {file_path} with encoding: {encoding}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error reading {file_path} with {encoding}: {e}")
                continue
        
        return None
    
    def _row_to_text(self, row: Dict[str, str], fieldnames: List[str], schema: Optional[Dict] = None) -> str:
        """CSV 행을 검색 가능한 텍스트로 변환 (스키마 활용 강화)"""
        text_parts = []
        
        for field in fieldnames:
            value = row.get(field, '').strip()
            if not value:
                continue
                
            # 스키마 기반 필드 정보 추출
            field_info = None
            if schema and 'properties' in schema:
                field_info = schema['properties'].get(field, {})
            
            # 필드 설명 활용한 자연어 형태 생성
            if field_info and field_info.get('description'):
                # 스키마에 설명이 있으면 더 자연스러운 텍스트 생성
                description = field_info['description']
                text_parts.append(f"{description}: {value}")
            else:
                # 기본 형태
                text_parts.append(f"{field}: {value}")
        
        return " | ".join(text_parts)
    
    def validate_row_against_schema(self, row: Dict[str, str], schema: Dict) -> Tuple[bool, List[str]]:
        """행 데이터의 스키마 유효성 검증"""
        errors = []
        
        if 'required' in schema:
            missing_fields = []
            for required_field in schema['required']:
                if required_field not in row or not row[required_field].strip():
                    missing_fields.append(required_field)
            
            if missing_fields:
                errors.append(f"Missing required fields: {missing_fields}")
        
        # 타입 검증 (기본적인 체크)
        if 'properties' in schema:
            for field, value in row.items():
                if field in schema['properties'] and value.strip():
                    field_schema = schema['properties'][field]
                    
                    # 숫자 타입 검증
                    if field_schema.get('type') == 'number' or (
                        isinstance(field_schema.get('type'), list) and 'number' in field_schema['type']
                    ):
                        try:
                            float(value)
                        except ValueError:
                            errors.append(f"Field '{field}' should be numeric, got: {value}")
        
        return len(errors) == 0, errors


# ================================================================
# 5. 이미지 처리기 (OCR)
# ================================================================

class ImageProcessor(DocumentProcessor):
    """이미지 OCR 처리기"""
    
    def __init__(self, root_dir: Path):
        super().__init__(root_dir)
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    def process(self, file_path: Path, **kwargs) -> List[TextChunk]:
        """이미지 파일을 OCR로 처리하여 TextChunk 리스트 반환"""
        logger.info(f"Processing image with OCR: {file_path}")
        
        try:
            text = self.extract_text_from_image(file_path)
            if not text.strip():
                logger.warning(f"No text extracted from image: {file_path}")
                return []
            
            # 이미지 메타데이터
            image_metadata = {
                'source_type': 'image',
                'source_file': file_path.name,
                'file_path': str(file_path),
                'ocr_engine': 'tesseract'
            }
            
            # 이미지 정보 추가
            try:
                with Image.open(file_path) as img:
                    image_metadata.update({
                        'image_width': img.width,
                        'image_height': img.height,
                        'image_mode': img.mode,
                        'image_format': img.format
                    })
            except Exception as e:
                logger.warning(f"Failed to get image info for {file_path}: {e}")
            
            # 청크 생성
            source_id = str(file_path.relative_to(self.root_dir))
            chunks = self._create_chunks(text, source_id, image_metadata)
            
            logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process image {file_path}: {e}")
            return []
    
    def extract_text_from_image(self, image_path: Path, lang: str = 'kor+eng') -> str:
        """이미지에서 OCR로 텍스트 추출"""
        try:
            with Image.open(image_path) as img:
                # OCR 실행
                text = pytesseract.image_to_string(img, lang=lang)
                
                # 텍스트 정리
                cleaned_text = self._clean_ocr_text(text)
                return cleaned_text
                
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return ""
    
    def _clean_ocr_text(self, text: str) -> str:
        """OCR 결과 텍스트 정리"""
        if not text:
            return ""
        
        # 연속 공백/줄바꿈 정리
        cleaned = re.sub(r'\s+', ' ', text)
        
        # 특수문자 정리 (기본적인 것만)
        cleaned = re.sub(r'[^\w\s가-힣.,!?()/-]', '', cleaned)
        
        return cleaned.strip()


# ================================================================
# 6. 팩토리 클래스
# ================================================================

class ProcessorFactory:
    """문서 처리기 팩토리"""
    
    @staticmethod
    def get_processor(file_path: Path, root_dir: Path) -> Optional[DocumentProcessor]:
        """파일 확장자에 따른 적절한 처리기 반환"""
        ext = file_path.suffix.lower()
        
        if ext == '.pdf':
            return PDFProcessor(root_dir)
        elif ext == '.csv':
            return CSVProcessor(root_dir)
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return ImageProcessor(root_dir)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return None
    
    @staticmethod
    def process_file(file_path: Path, root_dir: Path, **kwargs) -> List[TextChunk]:
        """파일을 자동으로 처리"""
        processor = ProcessorFactory.get_processor(file_path, root_dir)
        if processor:
            return processor.process(file_path, **kwargs)
        return []


# ================================================================
# 7. 유틸리티 함수들
# ================================================================

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """텍스트를 청크로 분할 (간단 버전)"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 문장 경계에서 자르기 시도
        if end < len(text):
            # 다음 문장 끝을 찾기
            sentence_end = max(
                text.rfind('.', start, end),
                text.rfind('!', start, end),
                text.rfind('?', start, end)
            )
            
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        
        chunks.append(text[start:end])
        start = max(end - overlap, start + 1)  # 무한루프 방지
    
    return chunks


def estimate_reading_time(text: str, wpm: int = 200) -> float:
    """텍스트 읽기 시간 추정 (분)"""
    word_count = len(text.split())
    return word_count / wpm


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """핵심 구문 추출 (간단한 버전)"""
    # 한국어 명사 패턴 (간단한 휴리스틱)
    korean_noun_pattern = r'[가-힣]{2,}'
    english_word_pattern = r'[A-Za-z]{3,}'
    
    korean_words = re.findall(korean_noun_pattern, text)
    english_words = re.findall(english_word_pattern, text)
    
    # 빈도 계산
    from collections import Counter
    word_freq = Counter(korean_words + english_words)
    
    # 상위 키워드 반환
    return [word for word, _ in word_freq.most_common(max_phrases)]


# ================================================================
# 8. 테스트 함수들
# ================================================================

def test_textifier():
    """textifier 모듈 테스트"""
    print("🧪 Textifier 모듈 테스트 시작")
    
    # TextChunk 테스트
    chunk = TextChunk(
        text="이것은 테스트 텍스트입니다.",
        metadata={"test": True},
        source_id="test.txt"
    )
    
    print(f"✅ TextChunk 생성: {chunk.source_id}")
    print(f"   토큰 수: {chunk.estimate_tokens()}")
    print(f"   해시: {chunk.metadata['text_hash']}")
    
    # 청킹 테스트
    long_text = "이것은 긴 텍스트입니다. " * 100
    chunks = chunk_text(long_text, chunk_size=200, overlap=50)
    print(f"✅ 텍스트 청킹: {len(chunks)}개 청크 생성")
    
    # 키워드 추출 테스트
    keywords = extract_key_phrases("경상남도인재개발원에서 교육과정 만족도 조사를 실시했습니다.")
    print(f"✅ 키워드 추출: {keywords}")
    
    print("🎉 Textifier 테스트 완료")


if __name__ == "__main__":
    # 모듈 직접 실행 시 테스트
    test_textifier()
    
    print("\n✅ utils/textifier.py 모듈 로드 완료")
    print("📦 사용 가능한 클래스:")
    print("   - TextChunk: 텍스트 청크 모델")
    print("   - DocumentProcessor: 문서 처리기 기본 클래스")
    print("   - PDFProcessor: PDF 처리")
    print("   - CSVProcessor: CSV 처리") 
    print("   - ImageProcessor: 이미지 OCR 처리")
    print("   - ProcessorFactory: 자동 처리기 선택")
