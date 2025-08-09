"""
Context Manager Module: 대화형 RAG를 위한 컨텍스트 관리

주요 기능:
1. 대화 세션 상태 관리 (메모리 기반)
2. 컨텍스트 요약 및 엔티티 추출
3. 후속질문 감지 및 처리
4. 캐시 키 생성 및 해시 관리
"""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

import openai
from .config import get_config
from .contracts import ConversationContext, ChatTurn, QueryRequest
from .logging_utils import get_logger, log_timer

logger = get_logger(__name__)
config = get_config()


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
            
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        # 메모리 기반 세션 저장소 (st.session_state와 연동)
        self.conversations: Dict[str, ConversationContext] = {}
        
        # 설정값
        self.recent_messages_window = config.CONVERSATION_RECENT_MESSAGES_WINDOW  # 6턴
        self.summary_update_interval = config.CONVERSATION_SUMMARY_UPDATE_INTERVAL  # 4턴
        self.summary_token_threshold = config.CONVERSATION_SUMMARY_TOKEN_THRESHOLD  # 1000토큰
        
        # 엔티티 추출용 패턴
        self.entity_patterns = {
            'date': r'\d{4}[-년]\d{1,2}[-월]\d{1,2}[일]?|\d{1,2}[-/]\d{1,2}|\d{4}년?',
            'satisfaction': r'만족도|평가|점수|성적',
            'course': r'교육과정|과정|코스|프로그램',
            'subject': r'교과목|과목|강의|수업',
            'cyber': r'사이버|온라인|민간위탁|나라배움터',
            'menu': r'식단|메뉴|점심|식사',
            'department': r'부서|팀|과|실|담당자',
            'phone': r'\d{2,3}-\d{3,4}-\d{4}|\d{10,11}',
        }
        
        # 후속질문 감지 패턴
        self.followup_indicators = {
            'reference': ['그것', '그거', '이것', '이거', '위의', '앞의', '해당', '그', '이'],
            'continuation': ['또', '그리고', '추가로', '더', '다음', '계속', '그런데', '그럼'],
            'clarification': ['자세히', '구체적으로', '더 알려', '설명해', '어떻게', '왜', '언제'],
            'comparison': ['차이', '비교', '다른', '같은', '비슷한'],
        }
        
        self._initialized = True
        logger.info("ContextManager initialized")
    
    def _estimate_tokens(self, text: str) -> int:
        """텍스트 토큰 수 추정 (1토큰 ≈ 3~4글자)"""
        return len(text) // 3
    
    def _extract_entities(self, text: str) -> List[str]:
        """텍스트에서 엔티티 추출"""
        entities = set()
        text_lower = text.lower()
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 1:  # 단일 글자 제외
                    entities.add(match.strip())
        
        # 키워드 기반 엔티티 추가
        keywords_found = []
        from .config import get_keyword_rules
        keyword_rules = get_keyword_rules()
        
        for handler_id, keywords in keyword_rules.items():
            for keyword in keywords:
                if keyword in text_lower and len(keyword) > 2:
                    keywords_found.append(keyword)
        
        entities.update(keywords_found[:10])  # 상위 10개만
        
        return list(entities)[:20]  # 최대 20개 엔티티
    
    def _detect_followup(self, current_query: str, recent_messages: List[ChatTurn]) -> bool:
        """후속질문 감지"""
        if not recent_messages:
            return False
        
        current_lower = current_query.lower()
        
        # 1. 지시대명사 패턴 확인
        for indicator in self.followup_indicators['reference']:
            if indicator in current_lower:
                return True
        
        # 2. 연결사 패턴 확인
        for indicator in self.followup_indicators['continuation']:
            if current_lower.startswith(indicator):
                return True
        
        # 3. 명확화 요청 패턴
        for indicator in self.followup_indicators['clarification']:
            if indicator in current_lower:
                return True
        
        # 4. 이전 메시지와 엔티티 겹침 확인
        if len(recent_messages) >= 2:
            last_assistant = recent_messages[-1]
            last_user = recent_messages[-2] if len(recent_messages) >= 2 else None
            
            if last_assistant.role == 'assistant' and last_user and last_user.role == 'user':
                current_entities = set(self._extract_entities(current_query))
                prev_entities = set(self._extract_entities(last_user.text + " " + last_assistant.text))
                
                # 50% 이상 엔티티 겹치면 후속질문으로 판단
                if current_entities and len(current_entities.intersection(prev_entities)) / len(current_entities) > 0.5:
                    return True
        
        return False
    
    def _generate_summary(self, messages: List[ChatTurn]) -> str:
        """대화 요약 생성"""
        if not messages:
            return ""
        
        # 메시지를 텍스트로 변환 (최근 10턴만)
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        conversation_text = ""
        
        for msg in recent_messages:
            role_prefix = "사용자: " if msg.role == "user" else "시스템: "
            conversation_text += f"{role_prefix}{msg.text}\n"
        
        # 토큰 제한 (최대 1500토큰 입력)
        if len(conversation_text) > 4500:  # 약 1500토큰
            conversation_text = conversation_text[-4500:]
        
        prompt = f"""다음 대화 내용을 간결하게 요약해주세요. 경상남도인재개발원 관련 주요 정보와 사용자 의도를 중심으로 3-4문장으로 정리하세요.

대화 내용:
{conversation_text}

요약 (200자 이내):"""

        try:
            with log_timer("context_summary_generation"):
                response = self.openai_client.chat.completions.create(
                    model=config.OPENAI_MODEL_ROUTER,  # 경량 모델 사용
                    messages=[
                        {"role": "system", "content": "You are a conversation summarizer. Provide concise summaries in Korean."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.1,
                    timeout=5.0
                )
                
                summary = response.choices[0].message.content.strip()
                logger.debug(f"Generated summary: {summary[:100]}...")
                return summary
                
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            # 폴백: 최근 사용자 메시지 기반 간단 요약
            user_messages = [msg.text for msg in recent_messages[-3:] if msg.role == "user"]
            if user_messages:
                return f"사용자가 {', '.join(user_messages[:2])}에 대해 질문함"
            return "대화 진행 중"
    
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
            logger.debug(f"Created new conversation context: {conversation_id}")
        
        return self.conversations[conversation_id]
    
    def add_message(self, conversation_id: str, role: str, text: str) -> ConversationContext:
        """메시지 추가 및 컨텍스트 업데이트"""
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
        
        # 엔티티 업데이트
        new_entities = self._extract_entities(text)
        context.entities.extend(new_entities)
        context.entities = list(set(context.entities))[:30]  # 중복 제거 및 최대 30개
        
        # 요약 업데이트 조건 확인
        should_update_summary = (
            len(context.recent_messages) % self.summary_update_interval == 0 or  # 매 N턴
            self._estimate_tokens(" ".join([msg.text for msg in context.recent_messages])) > self.summary_token_threshold  # 토큰 임계값 초과
        )
        
        if should_update_summary and len(context.recent_messages) >= 2:
            context.summary = self._generate_summary(context.recent_messages)
            logger.debug(f"Updated summary for {conversation_id}")
        
        context.updated_at = datetime.now(timezone.utc)
        
        return context
    
    def create_query_request(self, 
                           conversation_id: str, 
                           query_text: str, 
                           trace_id: Optional[str] = None) -> QueryRequest:
        """QueryRequest 객체 생성 (컨텍스트 포함)"""
        
        # 사용자 메시지 추가
        context = self.add_message(conversation_id, "user", query_text)
        
        # 후속질문 감지
        is_followup = self._detect_followup(query_text, context.recent_messages)
        
        # trace_id 생성 (제공되지 않은 경우)
        if not trace_id:
            trace_id = f"{conversation_id}-{int(time.time())}"
        
        request = QueryRequest(
            text=query_text,
            context=context,
            follow_up=is_followup,
            trace_id=trace_id,
            routing_hints={}
        )
        
        logger.info(f"Created QueryRequest for {conversation_id}, follow_up: {is_followup}")
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
            logger.info(f"Cleared context for {conversation_id}")
    
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
            logger.info(f"Cleaned up {len(old_conversations)} old contexts")
    
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
            "avg_messages_per_conversation": total_messages / max(total_conversations, 1)
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
            "updated_at": context.updated_at.isoformat() if context.updated_at else None
        }


# 싱글톤 인스턴스
_context_manager = None


def get_context_manager() -> ContextManager:
    """ContextManager 싱글톤 인스턴스 반환"""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


# 편의 함수들
def create_query_request(conversation_id: str, query_text: str, trace_id: Optional[str] = None) -> QueryRequest:
    """QueryRequest 생성 편의 함수"""
    manager = get_context_manager()
    return manager.create_query_request(conversation_id, query_text, trace_id)


def add_response(conversation_id: str, response_text: str) -> ConversationContext:
    """응답 추가 편의 함수"""
    manager = get_context_manager()
    return manager.add_response(conversation_id, response_text)


def get_context_hash(conversation_id: str) -> str:
    """컨텍스트 해시 편의 함수"""
    manager = get_context_manager()
    return manager.get_context_hash(conversation_id)


def cleanup_old_contexts():
    """오래된 컨텍스트 정리 편의 함수"""
    manager = get_context_manager()
    manager.cleanup_old_contexts()


# 테스트 함수
def test_context_manager():
    """ContextManager 테스트 함수"""
    manager = get_context_manager()
    
    # 테스트 대화 시뮬레이션
    conv_id = "test-conversation-123"
    
    # 첫 번째 질문
    req1 = manager.create_query_request(conv_id, "2024년 교육과정 만족도 결과를 보여주세요")
    print(f"Query 1 - Follow-up: {req1.follow_up}, Entities: {req1.context.entities[:5]}")
    
    # 시스템 응답 추가
    manager.add_response(conv_id, "2024년 교육과정 만족도는 전체 평균 4.2점입니다. 세부 과정별로는...")
    
    # 후속 질문
    req2 = manager.create_query_request(conv_id, "그 중에서 가장 높은 점수를 받은 과정은?")
    print(f"Query 2 - Follow-up: {req2.follow_up}, Context hash: {manager.get_context_hash(conv_id)}")
    
    # 통계 출력
    stats = manager.get_conversation_stats()
    print(f"Stats: {stats}")
    
    return manager


if __name__ == "__main__":
    test_context_manager()
