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
                    row_chunks = self._create_chunks(row_text, row_source_id, row_metadata)
                    chunks.extend(row_chunks)
                    
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
