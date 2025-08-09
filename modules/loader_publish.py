#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - 발행물 PDF 로더

PDF 발간물(2025plan.pdf, 2024pyeongga.pdf)을 처리하여 
vectorstore_unified_publish 인덱스를 구축합니다.

BaseLoader 패턴을 준수하며, 페이지별 메타데이터 생성 및
교육 계획/평가서 전용 템플릿을 적용합니다.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 프로젝트 모듈 임포트
from modules.base_loader import BaseLoader
from utils.textifier import PDFProcessor, TextChunk
from utils.config import config

# 로깅 설정
logger = logging.getLogger(__name__)


class PublishLoader(BaseLoader):
    """
    발행물 PDF 전용 로더
    
    처리 대상:
    - data/publish/2025plan.pdf (2025 교육훈련계획서)
    - data/publish/2024pyeongga.pdf (2024 종합평가서)
    
    특징:
    - BaseLoader 상속으로 표준 패턴 준수
    - 페이지별 청크화 및 메타데이터 생성
    - 발행물 전용 템플릿 적용
    - 해시 기반 증분 빌드
    """
    
    def __init__(self):
        super().__init__(
            domain="publish",
            source_dir=config.ROOT_DIR / "data" / "publish",
            vectorstore_dir=config.ROOT_DIR / "vectorstores" / "vectorstore_unified_publish",
            index_name="publish_index"
        )
        
        # PDF 처리기 초기화
        self.pdf_processor = PDFProcessor()
        
        # 발행물별 템플릿
        self.templates = {
            "2025plan": self._get_plan_template(),
            "2024pyeongga": self._get_evaluation_template(),
            "default": self._get_default_template()
        }
    
    def get_supported_extensions(self) -> List[str]:
        """지원하는 파일 확장자 반환"""
        return ['.pdf']
    
    def process_domain_data(self) -> List[TextChunk]:
        """
        발행물 PDF 파일들을 처리하여 TextChunk 리스트 반환
        
        Returns:
            List[TextChunk]: 처리된 텍스트 청크들
        """
        all_chunks = []
        
        # PDF 파일 검색 및 처리
        pdf_files = list(self.source_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"발간물 디렉토리에서 PDF 파일을 찾을 수 없습니다: {self.source_dir}")
            return all_chunks
        
        logger.info(f"발견된 PDF 파일 {len(pdf_files)}개 처리 시작")
        
        for pdf_path in pdf_files:
            try:
                logger.info(f"PDF 처리 중: {pdf_path.name}")
                
                # PDF 텍스트 청크 추출
                chunks = self.pdf_processor.process_file(pdf_path)
                
                if not chunks:
                    logger.warning(f"PDF에서 텍스트를 추출할 수 없습니다: {pdf_path.name}")
                    continue
                
                # 발행물별 템플릿 적용
                processed_chunks = self._apply_publish_template(chunks, pdf_path.stem)
                all_chunks.extend(processed_chunks)
                
                logger.info(f"{pdf_path.name}: {len(processed_chunks)}개 청크 생성 완료")
                
            except Exception as e:
                logger.error(f"PDF 처리 중 오류 발생 {pdf_path.name}: {e}")
                # 개별 파일 실패 시 전체 프로세스 중단하지 않음
                continue
        
        logger.info(f"발행물 처리 완료: 총 {len(all_chunks)}개 청크")
        return all_chunks
    
    def _apply_publish_template(self, chunks: List[TextChunk], document_name: str) -> List[TextChunk]:
        """
        발행물별 템플릿을 적용하여 청크 내용 보강
        
        Args:
            chunks: 원본 텍스트 청크들
            document_name: 문서명 (파일명에서 확장자 제거)
        
        Returns:
            List[TextChunk]: 템플릿이 적용된 청크들
        """
        processed_chunks = []
        
        # 템플릿 선택
        if "2025plan" in document_name.lower():
            template = self.templates["2025plan"]
            doc_type = "2025 교육훈련계획서"
        elif "2024pyeongga" in document_name.lower():
            template = self.templates["2024pyeongga"]
            doc_type = "2024 종합평가서"
        else:
            template = self.templates["default"]
            doc_type = "발행물"
        
        for chunk in chunks:
            try:
                # 기존 메타데이터에서 정보 추출
                original_metadata = chunk.metadata
                page_num = original_metadata.get('page_number', 1)
                source_file = original_metadata.get('source_file', document_name)
                
                # 템플릿 변수 준비
                template_vars = {
                    'document_type': doc_type,
                    'document_name': document_name,
                    'page_number': page_num,
                    'content': chunk.text,
                    'source_file': source_file,
                    'generation_date': datetime.now().strftime('%Y-%m-%d')
                }
                
                # 템플릿 적용
                enhanced_text = template.format(**template_vars)
                
                # 메타데이터 보강
                enhanced_metadata = {
                    **original_metadata,
                    'document_type': doc_type,
                    'document_category': 'publish',
                    'search_keywords': self._extract_search_keywords(chunk.text, doc_type),
                    'content_summary': self._generate_content_summary(chunk.text),
                    'processing_date': datetime.now().isoformat()
                }
                
                # 새로운 TextChunk 생성
                enhanced_chunk = TextChunk(
                    text=enhanced_text,
                    metadata=enhanced_metadata
                )
                
                processed_chunks.append(enhanced_chunk)
                
            except Exception as e:
                logger.error(f"템플릿 적용 중 오류 발생 (문서: {document_name}, 페이지: {page_num}): {e}")
                # 템플릿 적용 실패 시 원본 청크 유지
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _extract_search_keywords(self, text: str, doc_type: str) -> List[str]:
        """
        텍스트에서 검색 키워드 추출
        
        Args:
            text: 원본 텍스트
            doc_type: 문서 유형
        
        Returns:
            List[str]: 추출된 키워드 목록
        """
        keywords = []
        
        # 기본 키워드
        if "2025" in doc_type:
            keywords.extend(['2025', '교육계획', '교육훈련', '계획서'])
        elif "2024" in doc_type:
            keywords.extend(['2024', '평가', '종합평가', '평가서'])
        
        # 내용 기반 키워드 추출
        common_terms = [
            '교육', '과정', '훈련', '계획', '목표', '대상', '기간', '내용',
            '평가', '결과', '성과', '개선', '방안', '운영', '관리', '정책'
        ]
        
        for term in common_terms:
            if term in text:
                keywords.append(term)
        
        # 중복 제거 및 최대 10개로 제한
        return list(set(keywords))[:10]
    
    def _generate_content_summary(self, text: str) -> str:
        """
        텍스트 내용 요약 생성 (간단한 버전)
        
        Args:
            text: 원본 텍스트
        
        Returns:
            str: 생성된 요약
        """
        # 텍스트 길이에 따른 요약
        if len(text) <= 100:
            return text[:50] + "..." if len(text) > 50 else text
        
        # 첫 문장 추출 시도
        sentences = text.split('.')
        if sentences and len(sentences[0]) < 200:
            return sentences[0].strip() + "..."
        
        # 기본 요약
        return text[:100] + "..."
    
    def _get_plan_template(self) -> str:
        """2025 교육훈련계획서 전용 템플릿"""
        return """
[문서 정보]
- 문서유형: {document_type}
- 문서명: {document_name}
- 페이지: {page_number}
- 생성일: {generation_date}

[내용]
{content}

[검색 최적화]
이 내용은 경상남도인재개발원의 2025년 교육훈련계획서에서 발췌된 것으로, 
교육과정 계획, 운영방안, 교육목표, 교육대상, 교육기간, 교육내용 등과 관련된 정보를 담고 있습니다.

[출처]
파일: {source_file} (페이지 {page_number})
""".strip()
    
    def _get_evaluation_template(self) -> str:
        """2024 종합평가서 전용 템플릿"""
        return """
[문서 정보]
- 문서유형: {document_type}
- 문서명: {document_name}
- 페이지: {page_number}
- 생성일: {generation_date}

[내용]
{content}

[검색 최적화]
이 내용은 경상남도인재개발원의 2024년 종합평가서에서 발췌된 것으로,
교육운영 성과, 평가결과, 만족도 조사, 개선방안, 성취도 분석 등과 관련된 정보를 담고 있습니다.

[출처]
파일: {source_file} (페이지 {page_number})
""".strip()
    
    def _get_default_template(self) -> str:
        """기본 발행물 템플릿"""
        return """
[문서 정보]
- 문서유형: {document_type}
- 문서명: {document_name}
- 페이지: {page_number}
- 생성일: {generation_date}

[내용]
{content}

[검색 최적화]
이 내용은 경상남도인재개발원의 공식 발행물에서 발췌된 것으로,
교육훈련 관련 정책, 계획, 성과, 운영방안 등과 관련된 정보를 담고 있습니다.

[출처]
파일: {source_file} (페이지 {page_number})
""".strip()


def main():
    """메인 실행 함수 - 직접 실행 시 사용"""
    try:
        logger.info("=== 발행물 벡터스토어 구축 시작 ===")
        
        # 로더 인스턴스 생성 및 실행
        loader = PublishLoader()
        success = loader.build_vectorstore()
        
        if success:
            logger.info("=== 발행물 벡터스토어 구축 완료 ===")
        else:
            logger.error("=== 발행물 벡터스토어 구축 실패 ===")
            
    except Exception as e:
        logger.error(f"발행물 로더 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
