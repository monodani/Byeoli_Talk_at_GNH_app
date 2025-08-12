# ================================================================
# 5. ContextManager 메인 클래스 (OpenAI 호환성 수정)
# ================================================================

class ContextManager:
    """대화 컨텍스트 관리 싱글톤"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # ✅ OpenAI 클라이언트 안전한 초기화 (에러 핸들링 강화)
        try:
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            logger.info("✅ OpenAI 클라이언트 초기화 성공")
        except TypeError as e:
            if 'proxies' in str(e):
                logger.warning("⚠️ proxies 매개변수 오류 발생, 대체 초기화 시도")
                # proxies 오류 시 대체 초기화 방식
                try:
                    self.openai_client = openai.OpenAI(
                        api_key=config.OPENAI_API_KEY,
                        timeout=30.0
                    )
                    logger.info("✅ 대체 방식으로 OpenAI 클라이언트 초기화 성공")
                except Exception as e2:
                    logger.error(f"❌ 대체 초기화도 실패: {e2}")
                    self.openai_client = None
            else:
                logger.error(f"❌ OpenAI 초기화 실패: {e}")
                self.openai_client = None
        except Exception as e:
            logger.error(f"❌ 예상치 못한 OpenAI 초기화 오류: {e}")
            self.openai_client = None
        
        # 메모리 기반 세션 저장소 (st.session_state와 연동)
        self.conversations: Dict[str, ConversationContext] = {}
        
        # 설정값
        self.recent_messages_window = config.CONVERSATION_RECENT_MESSAGES_WINDOW  # 6턴
        self.summary_update_interval = config.CONVERSATION_SUMMARY_UPDATE_INTERVAL  # 4턴
        self.summary_token_threshold = config.CONVERSATION_SUMMARY_TOKEN_THRESHOLD  # 1000토큰
        
        # 컴포넌트 초기화 (OpenAI 클라이언트 상태에 따른 안전한 초기화)
        self.entity_extractor = EntityExtractor()
        
        # OpenAI 클라이언트가 있을 때만 AI 기반 컴포넌트 초기화
        if self.openai_client:
            try:
                self.context_summarizer = ContextSummarizer(self.openai_client)
                self.followup_detector = FollowUpDetector(self.openai_client)
                self.query_expander = QueryExpander(self.openai_client)
                logger.info("✅ AI 기반 컴포넌트 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ AI 컴포넌트 초기화 실패: {e}, 기본 모드로 동작")
                self.context_summarizer = None
                self.followup_detector = None
                self.query_expander = None
        else:
            logger.warning("⚠️ OpenAI 클라이언트 없음, AI 기능 없이 기본 모드로 동작")
            self.context_summarizer = None
            self.followup_detector = None
            self.query_expander = None
        
        self._initialized = True
        
        # 초기화 상태 로깅
        status = "완전" if self.openai_client else "제한적"
        logger.info(f"🎯 ContextManager {status} 초기화 완료")
    
    def _estimate_tokens(self, text: str) -> int:
        """텍스트 토큰 수 추정 (1토큰 ≈ 3~4글자)"""
        return len(text) // 3
    
    def _create_context_hash(self, summary: str, entities: List[str]) -> str:
        """컨텍스트 해시 생성 (캐시 키용)"""
        context_data = {
            "summary": summary,
            "entities": sorted(entities)  # 순서 무관하게 정렬
        }
        context_str = json.dumps(context_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(context_str.encode('utf-8')).hexdigest()[:16]
    
    def get_or_create_context(self, conversation_id: str) -> ConversationContext:
        """대화 컨텍스트 가져오기 또는 생성"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationContext(
                conversation_id=conversation_id,
                summary="",
                recent_messages=[],
                entities=[],
                updated_at=datetime.now(timezone.utc)
            )
            logger.debug(f"새 대화 컨텍스트 생성: {conversation_id}")
        
        return self.conversations[conversation_id]
    
    def add_message(self, conversation_id: str, role: str, text: str) -> ConversationContext:
        """메시지 추가 및 컨텍스트 업데이트 (AI 기능 안전 처리)"""
        context = self.get_or_create_context(conversation_id)
        
        # 새 메시지 추가
        new_message = ChatTurn(
            role=role,
            text=text,
            ts=datetime.now(timezone.utc)
        )
        context.recent_messages.append(new_message)
        
        # 윈도우 크기 유지 (최근 N턴만 보관)
        if len(context.recent_messages) > self.recent_messages_window:
            context.recent_messages = context.recent_messages[-self.recent_messages_window:]
        
        # 엔티티 업데이트 (사용자 메시지만)
        if role == "user":
            new_entities = self.entity_extractor.extract_entities(text)
            context.entities.extend(new_entities)
            context.entities = list(set(context.entities))[:30]  # 중복 제거 및 최대 30개
        
        # 요약 업데이트 (AI 기능 사용 가능할 때만)
        should_update_summary = (
            len(context.recent_messages) % self.summary_update_interval == 0 or  # 매 N턴
            self._estimate_tokens(" ".join([msg.text for msg in context.recent_messages])) > self.summary_token_threshold  # 토큰 임계값 초과
        )
        
        if should_update_summary and len(context.recent_messages) >= 2 and self.context_summarizer:
            try:
                context.summary = self.context_summarizer.generate_summary(context.recent_messages)
                logger.debug(f"대화 요약 업데이트: {conversation_id}")
            except Exception as e:
                logger.warning(f"요약 생성 실패: {e}")
        
        context.updated_at = datetime.now(timezone.utc)
        
        return context
    
    def create_query_request(self, 
                           conversation_id: str, 
                           query_text: str, 
                           trace_id: Optional[str] = None) -> QueryRequest:
        """QueryRequest 객체 생성 (AI 기능 안전 처리)"""
        
        # 사용자 메시지 추가
        context = self.add_message(conversation_id, "user", query_text)
        
        # 후속질문 감지 (AI 기능 사용 가능할 때만)
        is_followup = False
        if self.followup_detector:
            try:
                is_followup = self.followup_detector.detect_followup(query_text, context.recent_messages)
            except Exception as e:
                logger.warning(f"후속질문 감지 실패: {e}")
        
        # 쿼리 확장 (후속질문인 경우 + AI 기능 사용 가능할 때만)
        expanded_query = query_text
        if is_followup and self.query_expander:
            try:
                expanded_query = self.query_expander.expand_query(query_text, context)
            except Exception as e:
                logger.warning(f"쿼리 확장 실패: {e}")
        
        # trace_id 생성 (제공되지 않은 경우)
        if not trace_id:
            trace_id = f"{conversation_id}-{int(time.time())}"
        
        request = QueryRequest(
            text=expanded_query,  # 확장된 쿼리 사용
            context=context,
            follow_up=is_followup,
            trace_id=trace_id,
            routing_hints={}
        )
        
        logger.info(f"QueryRequest 생성: {conversation_id}, follow_up={is_followup}, expanded={expanded_query != query_text}")
        return request
    
    def add_response(self, conversation_id: str, response_text: str) -> ConversationContext:
        """시스템 응답 추가"""
        return self.add_message(conversation_id, "assistant", response_text)
    
    def get_context_hash(self, conversation_id: str) -> str:
        """컨텍스트 해시 반환 (캐시 키용)"""
        if conversation_id not in self.conversations:
            return "empty"
        
        context = self.conversations[conversation_id]
        return self._create_context_hash(context.summary, context.entities)
    
    def clear_context(self, conversation_id: str):
        """특정 대화 컨텍스트 삭제"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"컨텍스트 삭제: {conversation_id}")
    
    def cleanup_old_contexts(self, max_age_hours: int = 24):
        """오래된 컨텍스트 정리"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        
        old_conversations = [
            conv_id for conv_id, context in self.conversations.items()
            if context.updated_at and context.updated_at.timestamp() < cutoff_time
        ]
        
        for conv_id in old_conversations:
            del self.conversations[conv_id]
        
        if old_conversations:
            logger.info(f"오래된 컨텍스트 {len(old_conversations)}개 정리 완료")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """대화 통계 반환"""
        total_conversations = len(self.conversations)
        total_messages = sum(len(ctx.recent_messages) for ctx in self.conversations.values())
        
        active_conversations = sum(
            1 for ctx in self.conversations.values()
            if ctx.updated_at and (datetime.now(timezone.utc) - ctx.updated_at).seconds < 3600
        )
        
        return {
            "total_conversations": total_conversations,
            "active_conversations": active_conversations,
            "total_messages": total_messages,
            "avg_messages_per_conversation": total_messages / max(total_conversations, 1),
            "unique_entities": len(set(
                entity for ctx in self.conversations.values() 
                for entity in ctx.entities
            )),
            "ai_features_enabled": self.openai_client is not None
        }
    
    def export_conversation(self, conversation_id: str) -> Optional[Dict]:
        """대화 내용 내보내기 (디버그용)"""
        if conversation_id not in self.conversations:
            return None
        
        context = self.conversations[conversation_id]
        return {
            "conversation_id": conversation_id,
            "summary": context.summary,
            "entities": context.entities,
            "messages": [asdict(msg) for msg in context.recent_messages],
            "updated_at": context.updated_at.isoformat() if context.updated_at else None,
            "ai_features_enabled": self.openai_client is not None
        }
