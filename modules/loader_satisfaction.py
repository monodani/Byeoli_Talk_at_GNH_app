#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - 만족도 통합 로더 (BaseLoader 패턴 준수)

notice 로더 패턴을 따라 완전히 수정됨:
- process_domain_data(self) 시그니처로 변경
- 원본 CSV 파일 직접 읽기 로직 추가
- BaseLoader 표준 패턴 완전 준수
- 기존 템플릿 시스템 유지
"""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 프로젝트 모듈 임포트
from modules.base_loader import BaseLoader
from utils.textifier import TextChunk
from utils.config import config

# 로깅 설정
logger = logging.getLogger(__name__)


class SatisfactionLoader(BaseLoader):
    """
    만족도 데이터 통합 로더 - BaseLoader 표준 패턴 준수
    
    처리 대상:
    - data/satisfaction/course_satisfaction.csv (교육과정 만족도)
    - data/satisfaction/subject_satisfaction.csv (교과목 만족도)
    
    특징:
    - notice 로더와 동일한 process_domain_data(self) 시그니처
    - 원본 CSV 파일 직접 읽기
    - 기존 템플릿 시스템 완벽 보존
    - 해시 기반 증분 빌드 지원
    """
    
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
            domain="satisfaction",
            source_dir=config.ROOT_DIR / "data" / "satisfaction",
            vectorstore_dir=config.ROOT_DIR / "vectorstores" / "vectorstore_unified_satisfaction",
            index_name="satisfaction_index"
        )
        
        # 처리할 파일 정의
        self.course_file = self.source_dir / "course_satisfaction.csv"
        self.subject_file = self.source_dir / "subject_satisfaction.csv"
    
    def process_domain_data(self) -> List[TextChunk]:
        """
        BaseLoader 인터페이스 구현: 만족도 데이터 처리
        
        ✅ notice 로더와 동일한 시그니처: process_domain_data(self)
        """
        all_chunks = []
        
        # 1. 교육과정 만족도 처리
        course_chunks = self._process_course_satisfaction()
        all_chunks.extend(course_chunks)
        
        # 2. 교과목 만족도 처리
        subject_chunks = self._process_subject_satisfaction()
        all_chunks.extend(subject_chunks)
        
        logger.info(f"✅ 만족도 통합 처리 완료: 교육과정 {len(course_chunks)}개 + 교과목 {len(subject_chunks)}개 = 총 {len(all_chunks)}개 청크")
        
        return all_chunks
    
    def _process_course_satisfaction(self) -> List[TextChunk]:
        """교육과정 만족도 CSV 직접 읽기 및 처리"""
        chunks = []
        
        if not self.course_file.exists():
            logger.warning(f"교육과정 만족도 파일이 없습니다: {self.course_file}")
            return chunks
        
        try:
            logger.info(f"📊 교육과정 만족도 처리 시작: {self.course_file}")
            
            # CSV 직접 읽기 (자동 인코딩 감지)
            try:
                # 1. 먼저 utf-8로 시도 (표준)
                df = pd.read_csv(self.course_file, encoding='utf-8')
            except UnicodeDecodeError:
                # 2. 실패 시, 한국어 CSV에 자주 사용되는 cp949로 재시도
                df = pd.read_csv(self.course_file, encoding='cp949')
                logger.warning("⚠️ UTF-8 디코딩 실패. CP949 인코딩으로 다시 로드했습니다.")

            logger.info(f"📄 교육과정 만족도 데이터: {len(df)}행 로드됨")
            
            # 각 행을 TextChunk로 변환
            for idx, row in df.iterrows():
                try:
                    # 데이터 검증 및 정제
                    clean_data = self._validate_and_clean_course_data(row.to_dict(), f"course_row_{idx}")
                    if not clean_data:
                        continue
                    
                    # 템플릿 적용
                    try:
                        formatted_content = self.COURSE_TEMPLATE.format(**clean_data)
                    except KeyError as e:
                        logger.error(f"교육과정 템플릿 적용 실패 (행 {idx}): 누락 필드 {e}")
                        continue
                    
                    # 메타데이터 생성
                    metadata = {
                        'source_file': 'course_satisfaction.csv',
                        'source_id': f'satisfaction/course_satisfaction.csv#row_{idx}',
                        'satisfaction_type': 'course',
                        'education_course': clean_data.get('교육과정', ''),
                        'course_session': str(clean_data.get('교육과정_기수', '')),
                        'education_year': str(clean_data.get('교육연도', '')),
                        'overall_satisfaction': self._safe_convert_to_float(clean_data.get('전반만족도', '')),
                        'comprehensive_satisfaction': self._safe_convert_to_float(clean_data.get('종합만족도', '')),
                        'course_ranking': self._safe_convert_to_int(clean_data.get('교육과정_순위', '')),
                        'cache_ttl': 2592000,  # 30일 TTL
                        'processing_date': datetime.now().isoformat(),
                        'chunk_type': 'course_satisfaction'
                    }
                    
                    chunk = TextChunk(
                        text=formatted_content,
                        metadata=metadata
                    )
                    
                    chunks.append(chunk)
                    
                except Exception as e:
                    logger.error(f"교육과정 행 {idx} 처리 실패: {e}")
                    continue
            
            logger.info(f"✅ 교육과정 만족도 처리 완료: {len(chunks)}개 청크 생성")
            
        except Exception as e:
            logger.error(f"❌ 교육과정 만족도 파일 처리 실패: {e}")
        
        return chunks
    
    def _process_subject_satisfaction(self) -> List[TextChunk]:
        """교과목 만족도 CSV 직접 읽기 및 처리"""
        chunks = []
        
        if not self.subject_file.exists():
            logger.warning(f"교과목 만족도 파일이 없습니다: {self.subject_file}")
            return chunks
        
        try:
            logger.info(f"📊 교과목 만족도 처리 시작: {self.subject_file}")
            
            # CSV 직접 읽기 (자동 인코딩 감지)
            try:
                # 1. 먼저 utf-8로 시도 (표준)
                df = pd.read_csv(self.subject_file, encoding='utf-8')
            except UnicodeDecodeError:
                # 2. 실패 시, 한국어 CSV에 자주 사용되는 cp949로 재시도
                df = pd.read_csv(self.subject_file, encoding='cp949')
                logger.warning("⚠️ UTF-8 디코딩 실패. CP949 인코딩으로 다시 로드했습니다.")
            
            logger.info(f"📄 교과목 만족도 데이터: {len(df)}행 로드됨")
            
            # 각 행을 TextChunk로 변환
            for idx, row in df.iterrows():
                try:
                    # 데이터 검증 및 정제
                    clean_data = self._validate_and_clean_subject_data(row.to_dict(), f"subject_row_{idx}")
                    if not clean_data:
                        continue
                    
                    # 템플릿 적용
                    try:
                        formatted_content = self.SUBJECT_TEMPLATE.format(**clean_data)
                    except KeyError as e:
                        logger.error(f"교과목 템플릿 적용 실패 (행 {idx}): 누락 필드 {e}")
                        continue
                    
                    # 메타데이터 생성
                    metadata = {
                        'source_file': 'subject_satisfaction.csv',
                        'source_id': f'satisfaction/subject_satisfaction.csv#row_{idx}',
                        'satisfaction_type': 'subject',
                        'education_course': clean_data.get('교육과정', ''),
                        'course_session': str(clean_data.get('교육과정_기수', '')),
                        'subject_name': clean_data.get('교과목(강의)', ''),
                        'education_year': str(clean_data.get('교육연도', '')),
                        'lecture_satisfaction': self._safe_convert_to_float(clean_data.get('강의만족도', '')),
                        'subject_ranking': self._safe_convert_to_int(clean_data.get('교과목(강의)_순위', '')),
                        'cache_ttl': 2592000,  # 30일 TTL
                        'processing_date': datetime.now().isoformat(),
                        'chunk_type': 'subject_satisfaction'
                    }
                    
                    chunk = TextChunk(
                        text=formatted_content,
                        metadata=metadata
                    )
                    
                    chunks.append(chunk)
                    
                except Exception as e:
                    logger.error(f"교과목 행 {idx} 처리 실패: {e}")
                    continue
            
            logger.info(f"✅ 교과목 만족도 처리 완료: {len(chunks)}개 청크 생성")
            
        except Exception as e:
            logger.error(f"❌ 교과목 만족도 파일 처리 실패: {e}")
        
        return chunks
    
    def _validate_and_clean_course_data(self, row_data: Dict[str, Any], source_id: str) -> Optional[Dict[str, str]]:
        """교육과정 데이터 검증 및 정제"""
        try:
            # 필수 필드 확인
            required_fields = ['교육과정', '교육과정_기수', '전반만족도']
            for field in required_fields:
                if field not in row_data or pd.isna(row_data[field]):
                    logger.warning(f"교육과정 필수 필드 누락 ({source_id}): {field}")
                    return None
            
            # 데이터 정제
            clean_data = {}
            for key, value in row_data.items():
                if pd.isna(value):
                    clean_data[key] = ''
                else:
                    clean_data[key] = str(value).strip()
            
            return clean_data
            
        except Exception as e:
            logger.error(f"교육과정 데이터 검증 실패 ({source_id}): {e}")
            return None
    
    def _validate_and_clean_subject_data(self, row_data: Dict[str, Any], source_id: str) -> Optional[Dict[str, str]]:
        """교과목 데이터 검증 및 정제"""
        try:
            # 필수 필드 확인
            required_fields = ['교육과정', '교과목(강의)', '강의만족도']
            for field in required_fields:
                if field not in row_data or pd.isna(row_data[field]):
                    logger.warning(f"교과목 필수 필드 누락 ({source_id}): {field}")
                    return None
            
            # 데이터 정제
            clean_data = {}
            for key, value in row_data.items():
                if pd.isna(value):
                    clean_data[key] = ''
                else:
                    clean_data[key] = str(value).strip()
            
            return clean_data
            
        except Exception as e:
            logger.error(f"교과목 데이터 검증 실패 ({source_id}): {e}")
            return None
    
    def _safe_convert_to_float(self, value: Any) -> float:
        """안전한 float 변환"""
        try:
            if pd.isna(value) or value == '':
                return 0.0
            return float(str(value).strip())
        except (ValueError, TypeError):
            return 0.0
    
    def _safe_convert_to_int(self, value: Any) -> int:
        """안전한 int 변환"""
        try:
            if pd.isna(value) or value == '':
                return 0
            return int(float(str(value).strip()))
        except (ValueError, TypeError):
            return 0


# ================================================================
# 개발/테스트용 진입점
# ================================================================

def main():
    """개발/테스트용 진입점"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    loader = SatisfactionLoader()
    
    try:
        # BaseLoader의 표준 인터페이스 사용
        success = loader.build_vectorstore()
        
        if success:
            logger.info("✅ 만족도 벡터스토어 구축 완료")
        else:
            logger.error("❌ 만족도 벡터스토어 구축 실패")
            
    except Exception as e:
        logger.error(f"❌ 로더 실행 실패: {e}")
        raise


if __name__ == '__main__':
    main()
