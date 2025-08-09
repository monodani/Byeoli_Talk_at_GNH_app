def validate_schema(self, file_path: Path, schema_path: Path) -> bool:
        """CSV 파일의 스키마 검증 강화"""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            
            # CSV 파일 다중 인코딩 시도로 읽기
            content = self._read_csv_with_fallback_encoding(file_path)
            if not content:
                logger.error(f"Cannot read CSV file with any encoding: {file_path}")
                return False
            
            # CSV 파싱 및 헤더 확인
            import pandas as pd
            from io import StringIO
            import csv
            
            reader = csv.DictReader(StringIO(content))
            headers = [col.strip() for col in reader.fieldnames] if reader.fieldnames else []
            
            if not headers:
                logger.error(f"No headers found in CSV file: {file_path}")
                return False
            
            logger.info(f"CSV headers detected: {headers}")
            
            # 필수 컬럼 검증
            validation_errors = []
            
            if 'required' in schema:
                missing_cols = set(schema['required']) - set(headers)
                if missing_cols:
                    validation_errors.append(f"Missing required columns: {missing_cols}")
            
            # 컬럼 타입 기본 검증
            if 'properties' in schema:
                unknown_cols = set(headers) - set(schema['properties'].keys())
                if unknown_cols:
                    logger.warning(f"Unknown columns in {file_path}: {unknown_cols}")
            
            # 데이터 샘플 검증 (처음 5행)
            row_validation_errors = []
            try:
                reader = csv.DictReader(StringIO(content))
                for i, row in enumerate(reader):
                    if i >= 5:  # 처음 5행만 검증
                        break
                    
                    row_errors = self._validate_row_types(row, schema, i + 1)
                    if row_errors:
                        row_validation_errors.extend(row_errors)
                        
            except Exception as e:
                validation_errors.append(f"Row validation failed: {e}")
            
            # 결과 처리
            if validation_errors:
                logger.error(f"Schema validation failed for {file_path}:")
                for error in validation_errors:
                    logger.error(f"  - {error}")
                return False
            
            if row_validation_errors:
                logger.warning(f"Data validation warnings for {file_path}:")
                for error in row_validation_errors[:10]:  # 최대 10개만 표시
                    logger.warning(f"  - {error}")
                # 경고는 있지만 스키마 검증은 통과
            
            logger.info(f"Schema validation passed for {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Schema validation failed for {file_path}: {e}")
            return False
    
    def _read_csv_with_fallback_encoding(self, file_path: Path) -> Optional[str]:
        """다중 인코딩 시도로 CSV 읽기"""
        encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    logger.debug(f"Successfully read {file_path} with encoding: {encoding}")
                    return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Unexpected error reading {file_path} with {encoding}: {e}")
                continue
        
        return None
    
    def _validate_row_types(self, row: Dict[str, str], schema: Dict, row_num: int) -> List[str]:
        """행 데이터 타입 검증"""
        errors = []
        
        if 'properties' not in schema:
            return errors
        
        for field, value in row.items():
            field = field.strip()
            value_str = str(value).strip()
            
            if field in schema['properties'] and value_str:
                field_schema = schema['properties'][field]
                
                # oneOf 스키마 처리
                if 'oneOf' in field_schema:
                    valid = False
                    for option in field_schema['oneOf']:
                        if self._validate_single_type(value_str, option):
                            valid = True
                            break
                    
                    if not valid:
                        errors.append(f"Row {row_num}, field '{field}': value '{value_str}' doesn't match any allowed type")
                
                # 단일 타입 처리
                elif 'type' in field_schema:
                    if not self._validate_single_type(value_str, field_schema):
                        expected_type = field_schema['type']
                        errors.append(f"Row {row_num}, field '{field}': expected {expected_type}, got '{value_str}'")
        
        return errors
    
    def _validate_single_type(self, value: str, type_schema: Dict) -> bool:
        """단일 타입 검증"""
        field_type = type_schema.get('type')
        
        if field_type == 'string':
            return True  # 모든 값은 문자열로 읽힐 수 있음
        
        elif field_type == 'integer':
            try:
                int(value)
                return True
            except ValueError:
                return False
        
        elif field_type == 'number':
            try:
                float(value)
                return True
            except ValueError:
                return False
        
        # 패턴 검증
        if 'pattern' in type_schema:
            import re
            pattern = type_schema['pattern']
            try:
                return bool(re.match(pattern, value))
            except re.error:
                logger.warning(f"Invalid regex pattern: {pattern}")
                return True
        
        return True
