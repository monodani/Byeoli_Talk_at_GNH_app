"""
검증 유틸리티
데이터 검증, 정제, 에러 복구 등의 공통 기능 제공
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

from utils.logging_utils import get_logger

logger = get_logger(__name__)

class DataValidator:
    """데이터 검증 및 정제 클래스"""
    
    # 한국어 날짜 패턴들
    DATE_PATTERNS = [
        r'\d{4}\.\d{1,2}\.\d{1,2}',  # 2025.3.15
        r'\d{4}-\d{1,2}-\d{1,2}',   # 2025-03-15
        r'\d{4}/\d{1,2}/\d{1,2}',   # 2025/03/15
        r'\d{4}\.\d{1,2}\.\d{1,2}\.~\d{4}\.\d{1,2}\.\d{1,2}\.',  # 2025.3.15.~2025.3.17.
        r'\d{4}\.\d{1,2}\.\d{1,2}~\d{1,2}\.\d{1,2}',  # 2025.3.15~3.17
    ]
    
    # 전화번호 패턴들
    PHONE_PATTERNS = [
        r'^\d{2,3}-\d{3,4}-\d{4}$',  # 055-254-2025
        r'^\d{10,11}$',               # 05525402025
        r'^\(\d{2,3}\)\s?\d{3,4}-\d{4}$',  # (055) 254-2025
    ]
    
    @staticmethod
    def normalize_numeric_string(value: str) -> str:
        """숫자 문자열 정규화"""
        if not value or not isinstance(value, str):
            return "0"
        
        # 공백, 쉼표 제거
        cleaned = re.sub(r'[,\s]', '', value.strip())
        
        # 순수 숫자인지 확인
        if re.match(r'^-?\d*\.?\d+$', cleaned):
            return cleaned
        
        # 숫자 부분만 추출 시도
        numbers = re.findall(r'-?\d*\.?\d+', cleaned)
        if numbers:
            return numbers[0]
        
        return "0"
    
    @staticmethod
    def safe_float_convert(value: Any, default: float = 0.0) -> float:
        """안전한 float 변환"""
        if value is None:
            return default
        
        try:
            if isinstance(value, (int, float)):
                return float(value)
            
            # 문자열 처리
            cleaned = DataValidator.normalize_numeric_string(str(value))
            result = float(cleaned)
            
            # 만족도 점수 범위 검증 (0-5점)
            if 0 <= result <= 5:
                return result
            elif result > 5:
                logger.warning(f"Satisfaction score {result} exceeds maximum (5.0), capping to 5.0")
                return 5.0
            elif result < 0:
                logger.warning(f"Satisfaction score {result} is negative, setting to 0.0")
                return 0.0
            
            return result
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Float conversion failed for '{value}': {e}")
            return default
    
    @staticmethod
    def safe_int_convert(value: Any, default: int = 0) -> int:
        """안전한 int 변환"""
        if value is None:
            return default
        
        try:
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            
            # 문자열 처리
            cleaned = DataValidator.normalize_numeric_string(str(value))
            return int(float(cleaned))  # float을 거쳐서 변환 (소수점 처리)
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Int conversion failed for '{value}': {e}")
            return default
    
    @staticmethod
    def validate_phone_number(phone: str) -> bool:
        """전화번호 형식 검증"""
        if not phone or not isinstance(phone, str):
            return False
        
        cleaned = phone.strip()
        for pattern in DataValidator.PHONE_PATTERNS:
            if re.match(pattern, cleaned):
                return True
        
        return False
    
    @staticmethod
    def normalize_phone_number(phone: str) -> str:
        """전화번호 정규화"""
        if not phone:
            return ""
        
        # 숫자만 추출
        digits = re.sub(r'\D', '', phone)
        
        if len(digits) == 11 and digits.startswith('0'):
            # 055-254-2025 형태로 변환
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        elif len(digits) == 10:
            # 지역번호 2자리인 경우
            return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"
        
        return phone  # 변환 실패 시 원본 반환
    
    @staticmethod
    def validate_date_format(date_str: str) -> bool:
        """날짜 형식 검증"""
        if not date_str or not isinstance(date_str, str):
            return False
        
        for pattern in DataValidator.DATE_PATTERNS:
            if re.search(pattern, date_str.strip()):
                return True
        
        return False
    
    @staticmethod
    def extract_year_from_string(text: str) -> Optional[int]:
        """문자열에서 연도 추출"""
        if not text:
            return None
        
        # 2024, 2025 등의 4자리 연도 찾기
        years = re.findall(r'20\d{2}', str(text))
        if years:
            year = int(years[0])
            if 2000 <= year <= 2100:
                return year
        
        return None
    
    @staticmethod
    def clean_text_field(text: str, max_length: int = 500) -> str:
        """텍스트 필드 정제"""
        if not text:
            return ""
        
        # 기본 정제
        cleaned = str(text).strip()
        
        # 연속된 공백 정리
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 길이 제한
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length-3] + "..."
            logger.debug(f"Text truncated to {max_length} characters")
        
        return cleaned
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
        """필수 필드 검증"""
        missing_fields = []
        
        for field in required_fields:
            value = data.get(field)
            if value is None or str(value).strip() == "":
                missing_fields.append(field)
        
        return len(missing_fields) == 0, missing_fields
    
    @staticmethod
    def apply_field_defaults(data: Dict[str, Any], field_defaults: Dict[str, Any]) -> Dict[str, Any]:
        """필드 기본값 적용"""
        result = data.copy()
        
        for field, default_value in field_defaults.items():
            if field not in result or not str(result[field]).strip():
                result[field] = default_value
                logger.debug(f"Applied default value for field '{field}': {default_value}")
        
        return result

class SatisfactionDataValidator(DataValidator):
    """만족도 데이터 전용 검증기"""
    
    # 교육과정 필드 기본값
    COURSE_FIELD_DEFAULTS = {
        '교육과정': '정보없음',
        '교육과정_유형': '미분류',
        '교육주차': '정보없음',
        '교육일자': '미상',
        '교육장소': '미상',
        '교육인원': '0',
        '교육과정_기수': '0',
        '교육연도': '0',
        '교육과정_순위': '0',
        '전반만족도': '0.0',
        '역량향상도': '0.0',
        '현업적용도': '0.0',
        '교과편성_만족도': '0.0',
        '교육과정별_강의만족도_평균': '0.0',
        '종합만족도': '0.0'
    }
    
    # 교과목 필드 기본값
    SUBJECT_FIELD_DEFAULTS = {
        '교육과정': '정보없음',
        '교과목(강의)': '정보없음',
        '교육주차': '정보없음',
        '교육과정_기수': '0',
        '교육연도': '0',
        '교과목(강의)_순위': '0',
        '강의만족도': '0.0'
    }
    
    @classmethod
    def validate_and_clean_course_data(cls, row_data: Dict[str, str], source_id: str = "") -> Dict[str, str]:
        """교육과정 데이터 검증 및 정제"""
        # 필수 필드 확인
        required_fields = list(cls.COURSE_FIELD_DEFAULTS.keys())
        is_valid, missing_fields = cls.validate_required_fields(row_data, required_fields)
        
        if missing_fields:
            logger.info(f"Course data missing fields in {source_id}: {missing_fields}")
        
        # 기본값 적용
        clean_data = cls.apply_field_defaults(row_data, cls.COURSE_FIELD_DEFAULTS)
        
        # 특별 처리
        clean_data = cls._process_course_special_fields(clean_data, source_id)
        
        return clean_data
    
    @classmethod
    def validate_and_clean_subject_data(cls, row_data: Dict[str, str], source_id: str = "") -> Dict[str, str]:
        """교과목 데이터 검증 및 정제"""
        # 필수 필드 확인
        required_fields = list(cls.SUBJECT_FIELD_DEFAULTS.keys())
        is_valid, missing_fields = cls.validate_required_fields(row_data, required_fields)
        
        if missing_fields:
            logger.info(f"Subject data missing fields in {source_id}: {missing_fields}")
        
        # 기본값 적용
        clean_data = cls.apply_field_defaults(row_data, cls.SUBJECT_FIELD_DEFAULTS)
        
        # 특별 처리
        clean_data = cls._process_subject_special_fields(clean_data, source_id)
        
        return clean_data
    
    @classmethod
    def _process_course_special_fields(cls, data: Dict[str, str], source_id: str) -> Dict[str, str]:
        """교육과정 데이터 특별 처리"""
        # 만족도 점수 정규화
        satisfaction_fields = ['전반만족도', '역량향상도', '현업적용도', '교과편성_만족도', 
                             '교육과정별_강의만족도_평균', '종합만족도']
        
        for field in satisfaction_fields:
            original_value = data.get(field, '0.0')
            normalized_value = cls.safe_float_convert(original_value)
            data[field] = str(normalized_value)
        
        # 정수 필드 정규화
        int_fields = ['교육인원', '교육과정_기수', '교육연도', '교육과정_순위']
        for field in int_fields:
            original_value = data.get(field, '0')
            normalized_value = cls.safe_int_convert(original_value)
            data[field] = str(normalized_value)
        
        # 연도 추출 시도
        if data.get('교육연도') == '0':
            for field in ['교육일자', '교육주차']:
                extracted_year = cls.extract_year_from_string(data.get(field, ''))
                if extracted_year:
                    data['교육연도'] = str(extracted_year)
                    logger.debug(f"Extracted year {extracted_year} from {field} in {source_id}")
                    break
        
        # 텍스트 필드 정제
        text_fields = ['교육과정', '교육과정_유형', '교육주차', '교육일자', '교육장소']
        for field in text_fields:
            if field in data:
                data[field] = cls.clean_text_field(data[field])
        
        return data
    
    @classmethod
    def _process_subject_special_fields(cls, data: Dict[str, str], source_id: str) -> Dict[str, str]:
        """교과목 데이터 특별 처리"""
        # 만족도 점수 정규화
        satisfaction_score = cls.safe_float_convert(data.get('강의만족도', '0.0'))
        data['강의만족도'] = str(satisfaction_score)
        
        # 정수 필드 정규화
        int_fields = ['교육과정_기수', '교육연도', '교과목(강의)_순위']
        for field in int_fields:
            original_value = data.get(field, '0')
            normalized_value = cls.safe_int_convert(original_value)
            data[field] = str(normalized_value)
        
        # 연도 추출 시도
        if data.get('교육연도') == '0':
            extracted_year = cls.extract_year_from_string(data.get('교육주차', ''))
            if extracted_year:
                data['교육연도'] = str(extracted_year)
                logger.debug(f"Extracted year {extracted_year} from 교육주차 in {source_id}")
        
        # 텍스트 필드 정제
        text_fields = ['교육과정', '교과목(강의)', '교육주차']
        for field in text_fields:
            if field in data:
                data[field] = cls.clean_text_field(data[field])
        
        return data

# 전역 인스턴스
data_validator = DataValidator()
satisfaction_validator = SatisfactionDataValidator()
