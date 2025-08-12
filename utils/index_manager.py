def _load_domain(self, domain: str):
    """
    단일 도메인의 벡터스토어를 로드 (pickle 우회 전략)
    """
    meta = self.metadata[domain]
    
    logger.info(f"🔄 도메인 {domain} 로드 시작...")
    logger.debug(f"  - FAISS 경로: {meta.faiss_path}")
    logger.debug(f"  - PKL 경로: {meta.pkl_path}")
    logger.debug(f"  - BM25 경로: {meta.bm25_path}")
    
    try:
        if not meta.exists():
            logger.warning(f"⚠️ 도메인 {domain}에 필요한 인덱스 파일이 없습니다. 로드 건너뜁니다.")
            meta.vectorstore = None
            meta.bm25 = None
            return
        
        start_time = time.time()
        
        # 임베딩 모델 사용 (글로벌 또는 도메인별)
        embeddings_to_use = meta.embeddings or self.embeddings
        if not embeddings_to_use:
            logger.warning(f"⚠️ {domain} 임베딩 모델이 없어 FAISS 로드를 건너뜁니다.")
            meta.vectorstore = None
        else:
            # FAISS 인덱스 로드
            meta.vectorstore = FAISS.load_local(
                str(meta.vectorstore_path),
                embeddings_to_use,
                index_name=f"{domain}_index",
                allow_dangerous_deserialization=True
            )
            logger.info(f"✅ 도메인 {domain} FAISS 인덱스 로드 완료")
        
        # ✅ 핵심 변경: pickle 파일을 완전히 무시하고 FAISS에서만 로드
        meta.documents = []
        documents_loaded = False
        
        # 유일한 전략: FAISS docstore에서만 로드 (pickle 완전 우회)
        if meta.vectorstore:
            logger.info(f"🔄 {domain} FAISS docstore에서 문서 직접 로드 (pickle 우회)")
            try:
                # FAISS 내부 docstore 직접 접근
                raw_documents = list(meta.vectorstore.docstore._dict.values())
                logger.info(f"📄 {domain} FAISS docstore에서 {len(raw_documents)}개 문서 발견")
                
                for i, doc in enumerate(raw_documents):
                    try:
                        # LangChain Document → TextChunk 직접 변환 (pickle 없이)
                        chunk = TextChunk(
                            text=doc.page_content,
                            metadata=doc.metadata if hasattr(doc, 'metadata') else {},
                            source_id=doc.metadata.get('source_id', f'{domain}_{i}') if hasattr(doc, 'metadata') else f'{domain}_{i}',
                            chunk_index=i
                        )
                        meta.documents.append(chunk)
                        
                        # 진행 상황 로깅 (큰 데이터셋의 경우)
                        if i > 0 and i % 50 == 0:
                            logger.debug(f"📝 {domain} 문서 변환 진행: {i}/{len(raw_documents)}")
                            
                    except Exception as chunk_error:
                        logger.warning(f"⚠️ {domain} 청크 {i} 변환 실패: {chunk_error}")
                        # 변환 실패 시 기본 청크라도 생성
                        try:
                            fallback_chunk = TextChunk(
                                text=str(doc.page_content) if hasattr(doc, 'page_content') else f"문서 {i} 내용",
                                metadata={'fallback': True, 'domain': domain},
                                source_id=f'{domain}_fallback_{i}',
                                chunk_index=i
                            )
                            meta.documents.append(fallback_chunk)
                        except:
                            logger.warning(f"⚠️ {domain} 청크 {i} 폴백도 실패, 건너뜀")
                            continue
                
                if meta.documents:
                    documents_loaded = True
                    logger.info(f"✅ {domain} FAISS에서 {len(meta.documents)}개 문서 로드 완료")
                else:
                    logger.warning(f"⚠️ {domain} FAISS에서 변환된 문서가 없음")
                    
            except Exception as faiss_error:
                logger.error(f"❌ {domain} FAISS docstore 접근 실패: {faiss_error}")
                logger.debug(f"FAISS 오류 상세:\n{traceback.format_exc()}")
        else:
            logger.warning(f"⚠️ {domain} 벡터스토어가 로드되지 않아 문서 로드 불가")
        
        # 문서가 로드되지 않았으면 기본 더미 생성
        if not documents_loaded:
            logger.warning(f"⚠️ {domain} 모든 문서 로드 실패, 도메인별 더미 문서 생성")
            
            # 도메인별 의미있는 더미 데이터 생성
            domain_dummy_data = {
                "satisfaction": "만족도 조사 결과에 대한 정보를 제공합니다.",
                "general": "경상남도인재개발원의 일반 정보와 학칙을 제공합니다.",
                "publish": "교육계획서와 평가서 등 공식 발행물 정보를 제공합니다.",
                "cyber": "사이버 교육 일정과 과정 정보를 제공합니다.",
                "menu": "구내식당 메뉴와 식단 정보를 제공합니다.",
                "notice": "공지사항과 안내사항을 제공합니다."
            }
            
            dummy_text = domain_dummy_data.get(domain, f"{domain} 도메인의 정보를 제공합니다.")
            dummy_chunk = TextChunk(
                text=dummy_text,
                metadata={'domain': domain, 'type': 'dummy', 'created_at': datetime.now().isoformat()},
                source_id=f'{domain}_dummy',
                chunk_index=0
            )
            meta.documents = [dummy_chunk]
            logger.info(f"🔄 {domain} 더미 문서 생성 완료: '{dummy_text[:50]}...'")
        
        # BM25 인덱스 로드 시도 (실패해도 계속 진행)
        bm25_loaded = False
        if meta.bm25_path.exists():
            try:
                logger.info(f"🔄 {domain} BM25 인덱스 로드 시도")
                with open(meta.bm25_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                    if isinstance(bm25_data, tuple):
                        meta.bm25, _ = bm25_data
                    else:
                        meta.bm25 = bm25_data
                bm25_loaded = True
                logger.info(f"✅ 도메인 {domain} BM25 인덱스 로드 완료")
            except Exception as bm25_error:
                logger.warning(f"⚠️ 도메인 {domain} BM25 로드 실패: {bm25_error}")
                meta.bm25 = None
        else:
            logger.debug(f"⚠️ 도메인 {domain} BM25 인덱스 파일이 없습니다.")
            meta.bm25 = None
        
        # 로드 상태 업데이트
        meta.last_loaded = datetime.now()
        meta.load_count += 1
        meta.last_hash = meta.get_file_hash()
        elapsed = time.time() - start_time
        
        # 로드 상태 요약
        status_parts = []
        if meta.vectorstore:
            status_parts.append("FAISS")
        if bm25_loaded:
            status_parts.append("BM25")
        if meta.documents:
            status_parts.append(f"문서 {len(meta.documents)}개")
        
        logger.info(f"✅ 도메인 {domain} 로드 성공! ({', '.join(status_parts)}, {elapsed:.2f}초)")
        
    except Exception as e:
        meta.error_count += 1
        logger.error(f"❌ 도메인 {domain} 로드 실패: {e}")
        logger.debug(f"상세 오류:\n{traceback.format_exc()}")
        
        # 최종 안전장치: 에러 시에도 최소 기능 제공
        try:
            meta.vectorstore = None
            meta.bm25 = None
            
            if not meta.documents:
                error_chunk = TextChunk(
                    text=f"{domain} 도메인에 대한 정보를 로드하는 중 문제가 발생했습니다. 관리자에게 문의하세요.",
                    metadata={
                        'domain': domain, 
                        'type': 'error_fallback',
                        'error': str(e),
                        'created_at': datetime.now().isoformat()
                    },
                    source_id=f'{domain}_error',
                    chunk_index=0
                )
                meta.documents = [error_chunk]
                logger.info(f"🆘 {domain} 에러 폴백 문서 생성 완료")
                
        except Exception as final_error:
            logger.error(f"💥 {domain} 최종 폴백도 실패: {final_error}")
            # 이 경우 meta.documents는 빈 리스트로 남음
