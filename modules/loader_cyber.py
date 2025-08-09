"""
사이버 교육 로더
민간위탁 사이버교육(mingan.csv)과 나라배움터(nara.csv) 데이터를 처리하여 벡터스토어 생성
"""

import pandas as pd
from typing import List
from pathlib import Path

from .base_loader import BaseLoader
from utils.textifier import TextChunk
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class CyberLoader(BaseLoader):
    """사이버 교육 데이터 로더"""
    
    # 기존 템플릿 보존 (사용자 요청사항)
    MINGAN_TEMPLATE = """'{교육과정}' 과정은, 2025년 경상남도인재개발원에서 운영하고 있는 민간위탁 사이버교육 과정 중 하나로, {개발연도}년 {개발월}월에 만들어진 교육 콘텐츠로 내용 분류상 {구분}>{대분류}>{중분류}>{소분류}>{세분류}에 해당되고, 학습시간은 {학습시간}시간이며, 학습에 대한 교육 인정시간은 {인정시간}시간입니다.
---
"""

    NARA_TEMPLATE = """'{교육과정}' 과정은, 2025년 경상남도인재개발원 나라배움터에서 운영하는 공동활용 나라콘텐츠를 활용한 교육과정으로, 내용 분류상 {분류}에 해당되며, 학습시간은 {학습차시}이고 학습에 대한 교육 인정시간은 {인정시간}입니다. 참고사항으로, 본 과정은 교육 말미에 진행되는 별도의 평가가 {평가유무}.
---
"""
    
    def __init__(self):
        super().__init__(
            loader_id="cyber",
            source_dir="data/cyber",
            target_dir="vectorstores/vectorstore_cyber",
            schema_dir="schemas"
        )
    
    def get_file_patterns(self) -> List[str]:
        """처리할 파일 패턴 반환"""
        return ["mingan.csv", "nara.csv"]
    
    def process_domain_data(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        사이버 교육 데이터를 도메인별 템플릿으로 처리
        
        Args:
            chunks: textifier로 추출된 CSV 원본 청크들
            
        Returns:
            템플릿이 적용된 처리된 청크들
        """
        processed_chunks = []
        
        for chunk in chunks:
            try:
                # 소스 파일에 따라 다른 처리
                source_file = chunk.metadata.get("file_path", "")
                
                if "mingan.csv" in source_file:
                    processed_chunk = self._process_mingan_chunk(chunk)
                elif "nara.csv" in source_file:
                    processed_chunk = self._process_nara_chunk(chunk)
                else:
                    logger.warning(f"Unknown source file for chunk: {source_file}")
                    continue
                
                if processed_chunk:
                    processed_chunks.append(processed_chunk)
                    
            except Exception as e:
                logger.error(f"Failed to process chunk from {chunk.source_id}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_chunks)} cyber education chunks")
        return processed_chunks
    
    def _process_mingan_chunk(self, chunk: TextChunk) -> TextChunk:
        """민간위탁 사이버교육 청크 처리"""
        try:
            # CSV 행 데이터를 파싱하여 템플릿에 적용
            row_data = self._parse_csv_chunk(chunk.content)
            if not row_data:
                return None
            
            # 필수 필드 확인
            required_fields = ['교육과정', '개발연도', '개발월', '구분', '대분류', 
                             '중분류', '소분류', '세분류', '학습시간', '인정시간']
            
            missing_fields = [field for field in required_fields if field not in row_data]
            if missing_fields:
                logger.warning(f"Missing fields in mingan data: {missing_fields}")
                return None
            
            # 템플릿 적용
            formatted_content = self.MINGAN_TEMPLATE.format(**row_data)
            
            # 새로운 청크 생성
            new_metadata = {
                **chunk.metadata,
                "template_type": "mingan",
                "education_course": row_data.get('교육과정', ''),
                "category_path": f"{row_data.get('구분', '')}>{row_data.get('대분류', '')}>{row_data.get('중분류', '')}>{row_data.get('소분류', '')}>{row_data.get('세분류', '')}",
                "learning_hours": row_data.get('학습시간', ''),
                "recognition_hours": row_data.get('인정시간', '')
            }
            
            return TextChunk(
                content=formatted_content,
                metadata=new_metadata,
                source_id=chunk.source_id,
                chunk_index=chunk.chunk_index
            )
            
        except Exception as e:
            logger.error(f"Failed to process mingan chunk: {e}")
            return None
    
    def _process_nara_chunk(self, chunk: TextChunk) -> TextChunk:
        """나라배움터 청크 처리"""
        try:
            # CSV 행 데이터를 파싱하여 템플릿에 적용
            row_data = self._parse_csv_chunk(chunk.content)
            if not row_data:
                return None
            
            # 필수 필드 확인
            required_fields = ['교육과정', '분류', '학습차시', '인정시간', '평가유무']
            
            missing_fields = [field for field in required_fields if field not in row_data]
            if missing_fields:
                logger.warning(f"Missing fields in nara data: {missing_fields}")
                return None
            
            # 템플릿 적용
            formatted_content = self.NARA_TEMPLATE.format(**row_data)
            
            # 새로운 청크 생성
            new_metadata = {
                **chunk.metadata,
                "template_type": "nara",
                "education_course": row_data.get('교육과정', ''),
                "category": row_data.get('분류', ''),
                "learning_sessions": row_data.get('학습차시', ''),
                "recognition_hours": row_data.get('인정시간', ''),
                "evaluation_required": row_data.get('평가유무', '')
            }
            
            return TextChunk(
                content=formatted_content,
                metadata=new_metadata,
                source_id=chunk.source_id,
                chunk_index=chunk.chunk_index
            )
            
        except Exception as e:
            logger.error(f"Failed to process nara chunk: {e}")
            return None
    
    def _parse_csv_chunk(self, content: str) -> dict:
        """
        CSV 청크 내용을 파싱하여 딕셔너리로 변환
        
        Args:
            content: "필드명: 값 | 필드명: 값" 형태의 텍스트
            
        Returns:
            필드명-값 딕셔너리
        """
        try:
            data = {}
            
            # " | "로 필드 분리
            fields = content.split(" | ")
            
            for field in fields:
                if ":" in field:
                    key, value = field.split(":", 1)
                    data[key.strip()] = value.strip()
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to parse CSV chunk content: {e}")
            return {}
    
    def get_template_info(self) -> dict:
        """템플릿 정보 반환 (디버깅/모니터링용)"""
        return {
            "mingan_template": self.MINGAN_TEMPLATE,
            "nara_template": self.NARA_TEMPLATE,
            "required_fields": {
                "mingan": ['교육과정', '개발연도', '개발월', '구분', '대분류', 
                          '중분류', '소분류', '세분류', '학습시간', '인정시간'],
                "nara": ['교육과정', '분류', '학습차시', '인정시간', '평가유무']
            }
        }

# 편의 함수들
def build_cyber_index(force_rebuild: bool = False) -> bool:
    """사이버 교육 인덱스 빌드"""
    loader = CyberLoader()
    return loader.build_index(force_rebuild=force_rebuild)

def get_cyber_status() -> dict:
    """사이버 교육 로더 상태 조회"""
    loader = CyberLoader()
    return loader.get_status()

# 스크립트 직접 실행용
if __name__ == "__main__":
    import sys
    
    # 명령행 인자 처리
    force_rebuild = "--force" in sys.argv
    
    print(f"Building cyber education index (force_rebuild={force_rebuild})...")
    
    success = build_cyber_index(force_rebuild=force_rebuild)
    
    if success:
        status = get_cyber_status()
        print(f"✅ Build completed successfully!")
        print(f"📊 Status: {status}")
    else:
        print("❌ Build failed!")
        sys.exit(1)
