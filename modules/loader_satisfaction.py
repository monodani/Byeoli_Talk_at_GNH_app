"""
만족도 통합 로더
교육과정 만족도(course_satisfaction.csv)와 교과목 만족도(subject_satisfaction.csv)를 
통합 처리하여 단일 벡터스토어 생성
"""

import pandas as pd
from typing import List
from pathlib import Path

from .base_loader import BaseLoader
from utils.textifier import TextChunk
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class SatisfactionLoader(BaseLoader):
    """만족도 데이터 통합 로더"""
    
    # 기존 템플릿 보존 (코랩에서 검증된 로직)
    COURSE_TEMPLATE = (
        "{교육주차}에 개설된 '제{교육과정_기수}기 {교육과정}'은(는) '{교육과정_유형}'으로 분류되는 교육과정으로 "
        "{교육일자} {교육장소}에서 진행되었으며, 교육인원은 총 {교육인원}명이었습니다. "
        "'제{교육과정_기수}기 {교육과정}' 교육생의 교육에 대한 '전반적인 만족도'는 {전반만족도}점, "
        "교육효과 체감도 지표인 '역량향상도' 점수는 {역량향상도}점, '현업적용도'는 {현업적용도}점이었습니다. "
        "또한, '교과편성 만족도' {교과편성_만족도}점, '제{교육과정_기수}기 {교육과정}' 전체 강의에 대한 '강의만족도' 평균은 {교육과정별_강의만족도_평균}점이었으며, "
        "'제{교육과정_기수}기 {교육과정}'에 대한 모든 만족도 지표 평균인 '제{교육과정_기수}기 {교육과정}'의 '종합만족도'는 {종합만족도}점으로 "
        "'{교육연도}년' 전체 교육과정 중 '{교육과정_순위}위'를 기록했습니다."
    )
    
    SUBJECT_TEMPLATE = (
        "{교육주차}에 개설된 '제{교육과정_기수}기 {교육과정}'의 '{교과목(강의)}' 교과목(강의)에 대한 "
        "'강의만족도'는 {강의만족도}점으로 '{교육연도}년' 운영된 전체 교과목(강의) 중 '{교과목(강의)_순위}위'를 기록했습니다."
    )
    
    def __init__(self):
        super().__init__(
            loader_id="satisfaction",
            source_dir="data/satisfaction",
            target_dir="vectorstores/vectorstore_unified_satisfaction",
            schema_dir="schemas"
        )
    
    def get_file_patterns(self) -> List[str]:
        """처리할 파일 패턴 반환"""
        return ["course_satisfaction.csv", "subject_satisfaction.csv"]
    
    def process_domain_data(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        만족도 데이터를 통합 처리
        
        Args:
            chunks: textifier로 추출된 CSV 원본 청크들
            
        Returns:
            템플릿이 적용된 통합 처리된 청크들
        """
        course_chunks = []
        subject_chunks = []
        
        # 소스별로 청크 분류
        for chunk in chunks:
            source_file = chunk.metadata.get("file_path", "")
            
            if "course_satisfaction.csv" in source_file:
                course_chunks.append(chunk)
            elif "subject_satisfaction.csv" in source_file:
                subject_chunks.append(chunk)
            else:
                logger.warning(f"Unknown satisfaction source file: {source_file}")
        
        # 각각 처리
        processed_course = self._process_course_chunks(course_chunks)
        processed_subject = self._process_subject_chunks(subject_chunks)
        
        # 통합 반환
        all_processed = processed_course + processed_subject
        logger.info(f"Processed {len(processed_course)} course + {len(processed_subject)} subject = {len(all_processed)} total satisfaction chunks")
        
        return all_processed
    
    def _process_course_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """교육과정 만족도 청크 처리 (기존 로직 보존)"""
        processed_chunks = []
        
        for chunk in chunks:
            try:
                # CSV 행 데이터 파싱
                row_data = self._parse_csv_chunk(chunk.content)
                if not row_data:
                    continue
                
                # 필수 필드 확인 (기존 템플릿 기준)
                required_fields = [
                    '교육주차', '교육과정_기수', '교육과정', '교육과정_유형', 
                    '교육일자', '교육장소', '교육인원', '전반만족도', '역량향상도', 
                    '현업적용도', '교과편성_만족도', '교육과정별_강의만족도_평균', 
                    '종합만족도', '교육연도', '교육과정_순위'
                ]
                
                missing_fields = [field for field in required_fields if field not in row_data]
                if missing_fields:
                    logger.warning(f"Missing course fields: {missing_fields}")
                    continue
                
                # 템플릿 적용
                formatted_content = self.COURSE_TEMPLATE.format(**row_data)
                
                # 메타데이터 생성 (검색 최적화)
                new_metadata = {
                    **chunk.metadata,
                    "satisfaction_type": "course",
                    "education_course": row_data.get('교육과정', ''),
                    "course_session": row_data.get('교육과정_기수', ''),
                    "course_type": row_data.get('교육과정_유형', ''),
                    "education_week": row_data.get('교육주차', ''),
                    "education_year": row_data.get('교육연도', ''),
                    "overall_satisfaction": row_data.get('전반만족도', ''),
                    "comprehensive_satisfaction": row_data.get('종합만족도', ''),
                    "course_ranking": row_data.get('교육과정_순위', ''),
                    "capacity_improvement": row_data.get('역량향상도', ''),
                    "work_application": row_data.get('현업적용도', ''),
                    "curriculum_satisfaction": row_data.get('교과편성_만족도', ''),
                    "lecture_satisfaction_avg": row_data.get('교육과정별_강의만족도_평균', '')
                }
                
                processed_chunk = TextChunk(
                    content=formatted_content,
                    metadata=new_metadata,
                    source_id=chunk.source_id,
                    chunk_index=chunk.chunk_index
                )
                
                processed_chunks.append(processed_chunk)
                
            except Exception as e:
                logger.error(f"Failed to process course chunk from {chunk.source_id}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_chunks)} course satisfaction chunks")
        return processed_chunks
    
    def _process_subject_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """교과목 만족도 청크 처리 (기존 로직 보존)"""
        processed_chunks = []
        
        for chunk in chunks:
            try:
                # CSV 행 데이터 파싱
                row_data = self._parse_csv_chunk(chunk.content)
                if not row_data:
                    continue
                
                # 필수 필드 확인 (기존 템플릿 기준)
                required_fields = [
                    '교육주차', '교육과정_기수', '교육과정', '교과목(강의)', 
                    '강의만족도', '교육연도', '교과목(강의)_순위'
                ]
                
                missing_fields = [field for field in required_fields if field not in row_data]
                if missing_fields:
                    logger.warning(f"Missing subject fields: {missing_fields}")
                    continue
                
                # 템플릿 적용
                formatted_content = self.SUBJECT_TEMPLATE.format(**row_data)
                
                # 메타데이터 생성 (검색 최적화)
                new_metadata = {
                    **chunk.metadata,
                    "satisfaction_type": "subject",
                    "education_course": row_data.get('교육과정', ''),
                    "course_session": row_data.get('교육과정_기수', ''),
                    "subject_name": row_data.get('교과목(강의)', ''),
                    "education_week": row_data.get('교육주차', ''),
                    "education_year": row_data.get('교육연도', ''),
                    "lecture_satisfaction": row_data.get('강의만족도', ''),
                    "subject_ranking": row_data.get('교과목(강의)_순위', '')
                }
                
                processed_chunk = TextChunk(
                    content=formatted_content,
                    metadata=new_metadata,
                    source_id=chunk.source_id,
                    chunk_index=chunk.chunk_index
                )
                
                processed_chunks.append(processed_chunk)
                
            except Exception as e:
                logger.error(f"Failed to process subject chunk from {chunk.source_id}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_chunks)} subject satisfaction chunks")
        return processed_chunks
    
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
    
    def get_satisfaction_statistics(self) -> dict:
        """만족도 통계 정보 반환 (모니터링용)"""
        metadata = self.load_metadata()
        if not metadata:
            return {"status": "not_built"}
        
        return {
            "loader_id": self.loader_id,
            "total_chunks": metadata.total_chunks,
            "total_files": metadata.total_files,
            "last_build": metadata.last_build.isoformat(),
            "supported_types": ["course", "subject"],
            "templates": {
                "course_fields": [
                    "교육주차", "교육과정_기수", "교육과정", "교육과정_유형",
                    "전반만족도", "역량향상도", "현업적용도", "종합만족도", "교육과정_순위"
                ],
                "subject_fields": [
                    "교육주차", "교육과정_기수", "교육과정", "교과목(강의)",
                    "강의만족도", "교과목(강의)_순위"
                ]
            }
        }
    
    def search_by_satisfaction_type(self, satisfaction_type: str) -> dict:
        """
        만족도 타입별 검색 지원 정보
        
        Args:
            satisfaction_type: "course" 또는 "subject"
            
        Returns:
            해당 타입의 메타데이터 필터 정보
        """
        if satisfaction_type == "course":
            return {
                "filter": {"satisfaction_type": "course"},
                "searchable_fields": [
                    "education_course", "course_type", "education_year",
                    "overall_satisfaction", "comprehensive_satisfaction"
                ]
            }
        elif satisfaction_type == "subject":
            return {
                "filter": {"satisfaction_type": "subject"},
                "searchable_fields": [
                    "education_course", "subject_name", "education_year",
                    "lecture_satisfaction"
                ]
            }
        else:
            return {"error": "Invalid satisfaction_type. Use 'course' or 'subject'"}

# 편의 함수들
def build_satisfaction_index(force_rebuild: bool = False) -> bool:
    """통합 만족도 인덱스 빌드"""
    loader = SatisfactionLoader()
    return loader.build_index(force_rebuild=force_rebuild)

def get_satisfaction_status() -> dict:
    """만족도 로더 상태 조회"""
    loader = SatisfactionLoader()
    return loader.get_satisfaction_statistics()

def search_course_satisfaction_info() -> dict:
    """교육과정 만족도 검색 지원 정보"""
    loader = SatisfactionLoader()
    return loader.search_by_satisfaction_type("course")

def search_subject_satisfaction_info() -> dict:
    """교과목 만족도 검색 지원 정보"""
    loader = SatisfactionLoader()
    return loader.search_by_satisfaction_type("subject")

# 스크립트 직접 실행용
if __name__ == "__main__":
    import sys
    
    # 명령행 인자 처리
    force_rebuild = "--force" in sys.argv
    show_stats = "--stats" in sys.argv
    
    if show_stats:
        stats = get_satisfaction_status()
        print(f"📊 Satisfaction Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        sys.exit(0)
    
    print(f"Building unified satisfaction index (force_rebuild={force_rebuild})...")
    
    success = build_satisfaction_index(force_rebuild=force_rebuild)
    
    if success:
        stats = get_satisfaction_status()
        print(f"✅ Build completed successfully!")
        print(f"📊 Course + Subject combined: {stats.get('total_chunks', 0)} chunks")
        print(f"🔍 Search capabilities:")
        print(f"  - Course satisfaction: {search_course_satisfaction_info()}")
        print(f"  - Subject satisfaction: {search_subject_satisfaction_info()}")
    else:
        print("❌ Build failed!")
        sys.exit(1)
