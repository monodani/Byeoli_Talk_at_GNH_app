#!/usr/bin/env python3
"""
벼리톡@경상남도인재개발원 (경상남도인재개발원 RAG 챗봇) - utils/textifier.py (Pydantic v2 호환)

문서 텍스트화 및 청킹 유틸리티:
- PDF, CSV, 이미지 → 구조화된 텍스트 변환
- BaseLoader 호환성 보장
- 원본 메타데이터 보존
- 검색 최적화된 청킹

주요 클래스:
- TextChunk: 텍스트 청크 표준 모델 (Pydantic v2)
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

# 🔧 Pydantic v2 import
from pydantic import BaseModel, Field, ConfigDict

# 외부 라이브러리
import tiktoken
from pypdf import PdfReader
from PIL import Image
import pytesseract

# 로깅 설정
logger = logging.getLogger(__name__)

# ================================================================
# 1. TextChunk 핵심 모델 (Pydantic v2 완전 호환)
# ================================================================

class TextChunk(BaseModel):
    """
    텍스트 청크 표준 모델 (Pydantic v2 호환)
    
    모든 문서 처리기에서 공통으로 사용하는 텍스트 단위
    """
    model_config = ConfigDict(
        extra='forbid',
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    text: str = Field(..., min_length=1, max_length=50000, description="청크 텍스트")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="청크 메타데이터")
    source_id: str = Field(default="", description="소스 식별자")
    chunk_index: int = Field(default=0, ge=0, description="청크 인덱스")
    
    def model_post_init(self, __context: Any) -> None:
        """Pydantic v2 방식 후처리: 기본 메타데이터 설정"""
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
    
    def __hash__(self) -> int:
        """해시 값 계산 (캐시 키 용도)"""
        return hash((self.text, str(sorted(self.metadata.items()))))
    
    def get_source_id(self) -> str:
        """소스 ID 반환"""
        return self.source_id or self.metadata.get('source_id', 'unknown')
    
    def get_cache_ttl(self) -> int:
        """캐시 TTL 반환"""
        return self.metadata.get('cache_ttl', 86400)  # 기본 24시간


# ================================================================
# 2. 문서 처리기 기본 클래스
# ================================================================

class DocumentProcessor(ABC):
    """문서 처리기 기본 추상 클래스"""
    
    def __init__(self, root_dir: Optional[Path] = None):
        """root_dir 매개변수에 None 기본값 제공"""
        self.root_dir = root_dir if root_dir is not None else Path(".")
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
        chunks = []
        
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            
            logger.info(f"📄 PDF 처리 시작: {file_path.name} ({total_pages} 페이지)")
            
            full_text = ""
            page_texts = []
            
            # 페이지별 텍스트 추출
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        page_texts.append(page_text)
                        full_text += f"\n\n=== 페이지 {page_num} ===\n\n{page_text}"
                    
                except Exception as e:
                    logger.warning(f"페이지 {page_num} 추출 실패: {e}")
                    continue
            
            if not full_text.strip():
                logger.warning(f"PDF에서 텍스트를 추출할 수 없습니다: {file_path}")
                return chunks
            
            # 기본 메타데이터
            base_metadata = {
                'source_type': 'pdf',
                'source_file': file_path.name,
                'file_path': str(file_path),
                'total_pages': total_pages,
                'processing_date': datetime.now().isoformat()
            }
            
            # 청킹 처리
            source_id = f"pdf/{file_path.stem}"
            chunks = self._create_chunks(full_text, source_id, base_metadata)
            
            logger.info(f"✅ PDF 처리 완료: {len(chunks)}개 청크 생성")
            
        except Exception as e:
            logger.error(f"❌ PDF 처리 실패 {file_path}: {e}")
        
        return chunks


# ================================================================
# 4. CSV 처리기
# ================================================================

class CSVProcessor(DocumentProcessor):
    """CSV 데이터 처리기"""
    
    def process(self, file_path: Path, **kwargs) -> List[TextChunk]:
        """CSV 파일을 처리하여 TextChunk 리스트 반환"""
        chunks = []
        
        try:
            logger.info(f"📊 CSV 처리 시작: {file_path.name}")
            
            # 인코딩 감지 및 읽기
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding']
            
            # CSV 읽기
            with open(file_path, 'r', encoding=encoding, newline='') as f:
                csv_reader = csv.DictReader(f)
                fieldnames = csv_reader.fieldnames
                
                if not fieldnames:
                    logger.warning(f"CSV 파일에 헤더가 없습니다: {file_path}")
                    return chunks
                
                source_id = f"csv/{file_path.stem}"
                
                for row_num, row in enumerate(csv_reader, 1):
                    try:
                        # 빈 행 건너뛰기
                        if not any(str(v).strip() for v in row.values()):
                            continue
                        
                        # 행을 검색 가능한 텍스트로 변환
                        row_text = self._row_to_text(row, fieldnames)
                        if not row_text.strip():
                            continue
                        
                        # 행별 메타데이터
                        row_metadata = {
                            "source_type": "csv",
                            "source_file": file_path.name,
                            "row_number": row_num,
                            "fieldnames": fieldnames,
                            "file_path": str(file_path),
                            "row_data": dict(row),  # 원본 딕셔너리 저장
                            "processing_date": datetime.now().isoformat()
                        }
                        
                        # 행 단위로 청크화
                        chunk = TextChunk(
                            text=row_text,
                            metadata=row_metadata,
                            source_id=f"{source_id}#row{row_num}",
                            chunk_index=row_num - 1
                        )
                        chunks.append(chunk)
                        
                    except Exception as e:
                        logger.warning(f"행 {row_num} 처리 실패: {e}")
                        continue
            
            logger.info(f"✅ CSV 처리 완료: {len(chunks)}개 청크 생성")
            
        except Exception as e:
            logger.error(f"❌ CSV 처리 실패 {file_path}: {e}")
        
        return chunks
    
    def _row_to_text(self, row: Dict[str, Any], fieldnames: List[str]) -> str:
        """CSV 행을 검색 가능한 텍스트로 변환"""
        text_parts = []
        
        for field in fieldnames:
            value = str(row.get(field, '')).strip()
            if value and value.lower() not in ['', 'nan', 'null', 'none']:
                text_parts.append(f"{field}: {value}")
        
        return " | ".join(text_parts)


# ================================================================
# 5. 이미지 처리기 (OCR)
# ================================================================

class ImageProcessor(DocumentProcessor):
    """이미지 OCR 처리기"""
    
    def process(self, file_path: Path, **kwargs) -> List[TextChunk]:
        """이미지 파일을 OCR 처리하여 TextChunk 리스트 반환"""
        chunks = []
        
        try:
            logger.info(f"🖼️ 이미지 OCR 처리 시작: {file_path.name}")
            
            # 이미지 열기
            image = Image.open(file_path)
            
            # OCR 실행 (한국어 + 영어)
            ocr_text = pytesseract.image_to_string(
                image, 
                lang='kor+eng',
                config='--psm 6'  # 균일한 텍스트 블록 가정
            )
            
            if not ocr_text.strip():
                logger.warning(f"이미지에서 텍스트를 추출할 수 없습니다: {file_path}")
                return chunks
            
            # 기본 메타데이터
            base_metadata = {
                'source_type': 'image',
                'source_file': file_path.name,
                'file_path': str(file_path),
                'image_size': image.size,
                'ocr_confidence': 'estimated',
                'processing_date': datetime.now().isoformat()
            }
            
            # 청킹 처리
            source_id = f"image/{file_path.stem}"
            chunks = self._create_chunks(ocr_text, source_id, base_metadata)
            
            logger.info(f"✅ OCR 처리 완료: {len(chunks)}개 청크 생성")
            
        except Exception as e:
            logger.error(f"❌ 이미지 처리 실패 {file_path}: {e}")
        
        return chunks


# ================================================================
# 6. 처리기 팩토리
# ================================================================

class ProcessorFactory:
    """파일 타입에 따라 적절한 처리기를 선택하는 팩토리"""
    
    _processors = {
        '.pdf': PDFProcessor,
        '.csv': CSVProcessor,
        '.png': ImageProcessor,
        '.jpg': ImageProcessor,
        '.jpeg': ImageProcessor,
        '.bmp': ImageProcessor,
        '.tiff': ImageProcessor
    }
    
    @classmethod
    def get_processor(cls, file_path: Path, root_dir: Optional[Path] = None) -> Optional[DocumentProcessor]:
        """파일 확장자에 따라 적절한 처리기 반환"""
        suffix = file_path.suffix.lower()
        
        if suffix in cls._processors:
            processor_class = cls._processors[suffix]
            return processor_class(root_dir)
        
        logger.warning(f"지원되지 않는 파일 형식: {suffix}")
        return None
    
    @classmethod
    def process_file(cls, file_path: Path, root_dir: Optional[Path] = None, **kwargs) -> List[TextChunk]:
        """파일을 자동으로 처리하여 TextChunk 리스트 반환"""
        processor = cls.get_processor(file_path, root_dir)
        if processor:
            return processor.process(file_path, **kwargs)
        return []


# ================================================================
# 7. 유틸리티 함수들
# ================================================================

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """텍스트를 지정된 크기로 청킹"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 문장 경계에서 자르기 시도
        if end < len(text):
            # 다음 문장 끝 찾기
            next_sentence = text.find('.', end)
            if next_sentence != -1 and next_sentence - end < 100:
                end = next_sentence + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """텍스트에서 핵심 키워드 추출"""
    # 간단한 키워드 추출 (실제로는 더 정교한 NLP 필요)
    words = re.findall(r'\b\w+\b', text)
    
    # 불용어 제거 (간단한 버전)
    stopwords = {'이', '그', '저', '것', '을', '를', '에', '의', '가', '은', '는', '와', '과', '도', '만'}
    keywords = [word for word in words if len(word) > 1 and word not in stopwords]
    
    # 빈도 계산 및 상위 키워드 반환
    from collections import Counter
    word_counts = Counter(keywords)
    
    return [word for word, count in word_counts.most_common(max_phrases)]


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
    print(f"   해시: {chunk.metadata.get('text_hash', 'N/A')}")
    
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
    print("   - TextChunk: 텍스트 청크 모델 (Pydantic v2)")
    print("   - DocumentProcessor: 문서 처리기 기본 클래스")
    print("   - PDFProcessor: PDF 처리")
    print("   - CSVProcessor: CSV 처리") 
    print("   - ImageProcessor: 이미지 OCR 처리")
    print("   - ProcessorFactory: 자동 처리기 선택")
