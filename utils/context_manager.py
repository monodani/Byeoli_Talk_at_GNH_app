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


# ================================================================
# 1. EntityExtractor 클래스 (실제 데이터 기반)
# ================================================================

class EntityExtractor:
    """실제 프로젝트 데이터 기반 엔티티 추출기"""
    
    def __init__(self):
        # 실제 확인된 담당부서 연락처
        self.department_contacts = {
            '총무담당': '055-254-2013',
            '평가분석담당': '055-254-2023', 
            '교육기획담당': '055-254-2053',
            '교육운영1담당': '055-254-2063',
            '교육운영2담당': '055-254-2073',
            '사이버담당': '055-254-2083'
        }
        
        # 실제 교육과정명들
        self.education_courses = [
            # 리더십 교육
            '중견리더 과정', '과장급 필수 역량향상 과정', '5급 관리자 리더십 과정',
            '시·군 팀장 리더십 과정', '시·군 팀장 역량향상 과정', '코칭 리더십 과정',
            
            # 기본교육  
            '신규 임용(후보)자 과정', '신규공무원 역량향상 심화 과정',
            '7·8급 승진자 역량향상 과정', '6급 승진자 역량향상 과정', 
            '전입공무원 역량향상 과정', '소통과 공감 과정',
            
            # 직무교육
            '행사실무 과정', '기획능력 향상 과정', '명품 스피치 과정',
            '면접관 양성 과정', '공공언어 바르게 쓰기 과정',
            
            # 사이버교육
            '청탁금지법의 이해', '개인보호 장비관리', '국제회의 협상과정',
            '소방공무원법', '알기 쉬운 소방설비2', 'CISD리더의 역할 및 방법'
        ]
        
        # 나라배움터 분류
        self.nara_categories = ['직무', '소양', '시책', '디지털', 'Gov-MOOC']
        
        # 교육 분류체계
        self.domain_categories = {
            '기본역량': ['공직가치', '미래 변화 대응', '글로벌 마인드'],
            '리더십역량': ['의사결정', '동기 부여', '팀워크 형성', '업무관계망 형성'],
            '직무역량': ['기획력', '설득/협상력', '의사 표현력', '현장지향성']
        }
        
        # 날짜 패턴들
        self.date_patterns = [
            r'2024[-년\.]\d{1,2}[-월\.]\d{1,2}[일]?',
            r'2025[-년\.]\d{1,2}[-월\.]\d{1,2}[일]?',
            r'\d{1,2}[-\.]\d{1,2}[-\.]\s*~\s*\d{1,2}[-\.]\d{1,2}[-\.]',
            r'\d{1,2}월\s*\d{1,2}주차',
            r'[1-9]기',
            r'\d{1,2}주'
        ]
        
        # 연락처 패턴
        self.phone_pattern = r'055-254-\d{4}'
        
        logger.info("🔍 EntityExtractor 초기화 완료")
    
    def extract_entities(self, text: str) -> List[str]:
        """텍스트에서 엔티티 추출 (우선순위 기반)"""
        entities = set()
        text_lower = text.lower()
        
        # 1. 담당부서 추출 (최우선)
        for dept in self.department_contacts.keys():
            dept_variants = [dept, dept.replace('담당', ''), dept + '부서']
            for variant in dept_variants:
                if variant.lower() in text_lower:
                    entities.add(dept)
                    break
        
        # 2. 교육과정명 추출 (정확 매칭 + 유사도)
        for course in self.education_courses:
            # 정확 매칭
            if course.lower() in text_lower:
                entities.add(course)
            # 부분 매칭 (키워드 기반)
            elif any(keyword in text_lower for keyword in course.lower().split() if len(keyword) > 2):
                entities.add(course)
        
        # 3. 날짜/시기 추출
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.update(match for match in matches if len(match.strip()) > 1)
        
        # 4. 연락처 추출
        phone_matches = re.findall(self.phone_pattern, text)
        entities.update(phone_matches)
        
        # 5. 교육분류 추출
        for category in self.nara_categories:
            if category in text_lower:
                entities.add(category)
        
        # 도메인 분류
        for domain, keywords in self.domain_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                entities.add(domain)
        
        # 6. 일반 키워드 (만족도, 평가 등)
        general_keywords = ['만족도', '평가', '점수', '성적', '사이버교육', '온라인', 
                          '민간위탁', '나라배움터', '식단', '메뉴', '구내식당']
        for keyword in general_keywords:
            if keyword in text_lower:
                entities.add(keyword)
        
        result = list(entities)[:20]  # 최대 20개 제한
        logger.debug(f"추출된 엔티티: {result}")
        return result


# ================================================================
# 2. ContextSummarizer 클래스
# ================================================================

class ContextSummarizer:
    """대화 내용 요약 생성기 (gpt-4o-mini 기반)"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.model = config.OPENAI_MODEL_ROUTER  # gpt-4o-mini
        
        logger.info("📝 ContextSummarizer 초기화 완료")
    
    def generate_summary(self, recent_messages: List[ChatTurn]) -> str:
        """대화 내용을 200자 이내로 요약"""
        if not recent_messages:
            return ""
        
        # 대화 내용 구성
        conversation_text = []
        for msg in recent_messages[-6:]:  # 최근 6턴만
            role_kr = "사용자" if msg.role == "user" else "시스템"
            conversation_text.append(f"{role_kr}: {msg.content}")
        
        conversation_str = "\n".join(conversation_text)
        
        # 경남인재개발원 특화 요약 프롬프트
        prompt = f"""다음은 경상남도인재개발원 관련 대화입니다. 핵심 내용을 200자 이내로 요약하세요.

요약 기준:
- 교육과정명, 담당부서, 날짜 등 구체적 정보 우선
- 사용자의 주요 질문 의도 파악
- 경남인재개발원 업무 관련 키워드 강조

대화 내용:
{conversation_str}

요약 (200자 이내):"""

        try:
            with log_timer("context_summary_generation"):
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "경남인재개발원 전문 요약 AI입니다. 간결하고 정확한 요약을 제공합니다."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.1,
                    timeout=5.0
                )
                
                summary = response.choices[0].message.content.strip()
                logger.debug(f"생성된 요약: {summary[:100]}...")
                return summary
                
        except Exception as e:
            logger.warning(f"요약 생성 실패: {e}")
            # 폴백: 최근 사용자 메시지 기반 간단 요약
            user_messages = [msg.content for msg in recent_messages[-3:] if msg.role == "user"]
            if user_messages:
                return f"사용자가 {', '.join(user_messages[:2][:50])}에 대해 질문함"
            return "대화 진행 중"


# ================================================================
# 3. FollowUpDetector 클래스
# ================================================================

class FollowUpDetector:
    """후속질문 감지기 (패턴 + LLM 하이브리드)"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.model = config.OPENAI_MODEL_ROUTER  # gpt-4o-mini
        
        # 명확한 후속질문 패턴들
        self.followup_patterns = {
            'reference': ['그것', '그거', '이것', '이거', '위의', '앞의', '해당', '그', '이', '저것'],
            'continuation': ['또', '그리고', '추가로', '더', '다음', '계속', '그런데', '그럼', '그러면'],
            'clarification': ['자세히', '구체적으로', '더 알려', '설명해', '어떻게', '왜', '언제', '어디서'],
            'comparison': ['차이', '비교', '다른', '같은', '비슷한', '반대로'],
            'quantification': ['몇', '얼마', '언제까지', '며칠', '몇 시간']
        }
        
        logger.info("🔄 FollowUpDetector 초기화 완료")
    
    def detect_followup(self, current_query: str, recent_messages: List[ChatTurn]) -> bool:
        """후속질문 여부 판단"""
        if not recent_messages or len(recent_messages) < 2:
            return False
        
        current_lower = current_query.lower().strip()
        
        # 1차: 명확한 패턴 매칭 (빠른 판단)
        pattern_score = self._calculate_pattern_score(current_lower)
        
        if pattern_score >= 0.7:  # 명확한 후속질문 패턴
            logger.debug(f"패턴 기반 후속질문 감지: {pattern_score:.3f}")
            return True
        elif pattern_score <= 0.2:  # 명확히 독립적 질문
            logger.debug(f"패턴 기반 독립 질문 판단: {pattern_score:.3f}")
            return False
        
        # 2차: LLM 기반 판단 (모호한 경우만)
        return self._llm_based_detection(current_query, recent_messages)
    
    def _calculate_pattern_score(self, query: str) -> float:
        """패턴 기반 후속질문 점수 계산"""
        score = 0.0
        word_count = len(query.split())
        
        # 지시대명사 (강한 신호)
        for indicator in self.followup_patterns['reference']:
            if indicator in query:
                score += 0.4
        
        # 연결사로 시작 (강한 신호)
        for indicator in self.followup_patterns['continuation']:
            if query.startswith(indicator):
                score += 0.5
        
        # 명확화 요청
        for indicator in self.followup_patterns['clarification']:
            if indicator in query:
                score += 0.3
        
        # 비교 요청
        for indicator in self.followup_patterns['comparison']:
            if indicator in query:
                score += 0.2
        
        # 수량화 요청
        for indicator in self.followup_patterns['quantification']:
            if indicator in query:
                score += 0.2
        
        # 질문이 너무 짧으면 후속질문일 가능성 증가
        if word_count <= 3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _llm_based_detection(self, current_query: str, recent_messages: List[ChatTurn]) -> bool:
        """LLM 기반 후속질문 감지"""
        try:
            # 이전 대화 컨텍스트 구성
            context_msgs = []
            for msg in recent_messages[-4:]:  # 최근 4턴
                role_kr = "사용자" if msg.role == "user" else "챗봇"
                context_msgs.append(f"{role_kr}: {msg.content}")
            
            context_str = "\n".join(context_msgs)
            
            prompt = f"""다음 대화에서 마지막 사용자 질문이 이전 대화의 후속질문인지 판단하세요.

이전 대화:
{context_str}

현재 질문: {current_query}

후속질문 판단 기준:
- 이전 답변의 특정 부분에 대한 추가 질문
- 지시대명사 사용 ("그것", "이것" 등)
- 이전 맥락 없이는 이해하기 어려운 질문
- 비교나 추가 세부사항 요청

답변: YES 또는 NO만 답하세요."""

            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "후속질문 판단 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.0,
                timeout=3.0
            )
            
            result = response.choices[0].message.content.strip().upper()
            is_followup = result == "YES"
            
            logger.debug(f"LLM 기반 후속질문 판단: {is_followup}")
            return is_followup
            
        except Exception as e:
            logger.warning(f"LLM 기반 후속질문 감지 실패: {e}")
            # 폴백: 패턴 점수 기준으로 판단
            return self._calculate_pattern_score(current_query.lower()) >= 0.5


# ================================================================
# 4. QueryExpander 클래스
# ================================================================

class QueryExpander:
    """쿼리 확장기 (지시어/대명사 해소)"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.model = config.OPENAI_MODEL_ROUTER  # gpt-4o-mini
        
        logger.info("🔍 QueryExpander 초기화 완료")
    
    def expand_query(self, query: str, context: ConversationContext) -> str:
        """컨텍스트 기반 쿼리 확장"""
        if not context or not context.recent_messages:
            return query
        
        # 지시어/대명사가 있는지 확인
        pronouns = ['그것', '그거', '이것', '이거', '그', '이', '저것', '위의', '앞의', '해당']
        has_pronoun = any(pronoun in query for pronoun in pronouns)
        
        if not has_pronoun:
            return query
        
        try:
            # 이전 대화에서 핵심 엔티티 추출
            previous_entities = context.entities[-10:] if context.entities else []
            
            # 최근 어시스턴트 응답에서 주요 내용 추출
            recent_assistant_msgs = [
                msg.content for msg in context.recent_messages[-3:] 
                if msg.role == "assistant"
            ]
            
            context_info = f"""
주요 엔티티: {', '.join(previous_entities)}
최근 답변: {' '.join(recent_assistant_msgs)[:300]}
요약: {context.summary}
"""

            prompt = f"""다음 대화 맥락에서 사용자의 질문에 있는 지시어나 대명사를 구체적인 내용으로 바꿔주세요.

대화 맥락:
{context_info}

사용자 질문: {query}

지시어 해소 지침:
- "그것", "이것" → 구체적인 교육과정명이나 정책명
- "그", "이" → 앞서 언급된 구체적 대상
- "위의", "앞의" → 이전에 언급된 특정 항목

확장된 질문:"""

            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "대화 맥락을 이해하여 지시어를 구체적 내용으로 바꾸는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.1,
                timeout=4.0
            )
            
            expanded_query = response.choices[0].message.content.strip()
            
            # 확장된 쿼리가 너무 길거나 이상하면 원본 사용
            if len(expanded_query) > len(query) * 3 or len(expanded_query) < 5:
                logger.warning("쿼리 확장 결과 이상, 원본 사용")
                return query
            
            logger.debug(f"쿼리 확장: '{query}' → '{expanded_query}'")
            return expanded_query
            
        except Exception as e:
            logger.warning(f"쿼리 확장 실패: {e}")
            return query


# ================================================================
# 5. ContextManager 메인 클래스
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
            
        # Streamlit secrets 또는 환경변수에서 API 키 가져오기
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY")
        except:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in Streamlit secrets or environment")
        
        # proxies 매개변수 제거!
        self.openai_client = openai.OpenAI(
            api_key=api_key
            # proxies 제거됨
        )
        
        # 메모리 기반 세션 저장소 (st.session_state와 연동)
        self.conversations: Dict[str, ConversationContext] = {}
        
        # 설정값
        self.recent_messages_window = config.CONVERSATION_RECENT_MESSAGES_WINDOW  # 6턴
        self.summary_update_interval = config.CONVERSATION_SUMMARY_UPDATE_INTERVAL  # 4턴
        self.summary_token_threshold = config.CONVERSATION_SUMMARY_TOKEN_THRESHOLD  # 1000토큰
        
        # 컴포넌트 초기화
        self.entity_extractor = EntityExtractor()
        self.context_summarizer = ContextSummarizer(self.openai_client)
        self.followup_detector = FollowUpDetector(self.openai_client)
        self.query_expander = QueryExpander(self.openai_client)
        
        self._initialized = True
        logger.info("🎯 ContextManager 초기화 완료")
    
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
            # ✅ 수정: ConversationContext 스키마에 맞춰 수정
            self.conversations[conversation_id] = ConversationContext(
                session_id=conversation_id,  
                turns=[],                    
                entities={},                 
                summary="",
                updated_at=datetime.now()
            )
            logger.debug(f"새 대화 컨텍스트 생성: {conversation_id}")
        
        return self.conversations[conversation_id]
    
    def add_message(self, conversation_id: str, role: str, text: str) -> ConversationContext:
        """메시지 추가 및 컨텍스트 업데이트"""
        context = self.get_or_create_context(conversation_id)
        
        # ✅ 수정: ChatTurn 구조에 맞춰 수정
        new_message = ChatTurn(
            role=MessageRole(role) if isinstance(role, str) else role,
            content=text,  
            timestamp=datetime.now() 
        )
        
        # ✅ 수정: recent_messages 속성을 통해 turns에 추가
        context.recent_messages.append(new_message)
        
        # 윈도우 크기 유지 (최근 N턴만 보관)
        if len(context.recent_messages) > self.recent_messages_window:
            context.recent_messages = context.recent_messages[-self.recent_messages_window:]
        
        # 엔티티 업데이트 (사용자 메시지만)
        if role == "user":
            new_entities = self.entity_extractor.extract_entities(text)
            # ✅ 수정: entities는 Dict[str, List[str]] 구조
            if "extracted" not in context.entities:
                context.entities["extracted"] = []
            context.entities["extracted"].extend(new_entities)
            context.entities["extracted"] = list(set(context.entities["extracted"]))[:30]
        
        # 요약 업데이트 조건 확인
        should_update_summary = (
            len(context.recent_messages) % self.summary_update_interval == 0 or
            self._estimate_tokens(" ".join([msg.content for msg in context.recent_messages])) > self.summary_token_threshold
        )
        
        if should_update_summary and len(context.recent_messages) >= 2:
            context.summary = self.context_summarizer.generate_summary(context.recent_messages)
            logger.debug(f"대화 요약 업데이트: {conversation_id}")
        
        context.updated_at = datetime.now()
        
        return context
    
    def create_query_request(self, 
                           conversation_id: str, 
                           query_text: str, 
                           trace_id: Optional[str] = None) -> QueryRequest:
        """QueryRequest 객체 생성 (컨텍스트 포함)"""
        
        # 사용자 메시지 추가
        context = self.add_message(conversation_id, "user", query_text)
        
        # 후속질문 감지
        is_followup = self.followup_detector.detect_followup(query_text, context.recent_messages)
        
        # 쿼리 확장 (후속질문인 경우)
        expanded_query = query_text
        if is_followup:
            expanded_query = self.query_expander.expand_query(query_text, context)
        
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
            ))
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
        

    def update_context(self, conversation_id: str, role, content: str) -> ConversationContext:
        """컨텍스트 업데이트 메소드 (app.py 호환성)"""
        try:
            role_value = role.value if hasattr(role, 'value') else str(role)
            return self.add_message(conversation_id, role_value, content)
        except Exception as e:
            logger.warning(f"update_context 실패: {e}")
            # 기본 컨텍스트 반환
            return self.get_or_create_context(conversation_id)




# ================================================================
# 6. 싱글톤 인스턴스 및 편의 함수들
# ================================================================

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


# ================================================================
# 7. 테스트 함수
# ================================================================

def test_context_manager():
    """ContextManager 테스트 함수"""
    manager = get_context_manager()
    
    print("🧪 ContextManager 테스트 시작")
    
    # 테스트 대화 시뮬레이션
    conv_id = "test-conversation-123"
    
    try:
        # 첫 번째 질문
        print("\n=== 테스트 1: 첫 번째 질문 ===")
        req1 = manager.create_query_request(conv_id, "2024년 중견리더 과정 만족도 결과를 보여주세요")
        print(f"Query 1 - Follow-up: {req1.follow_up}")
        print(f"Entities: {req1.context.entities[:5]}")
        print(f"Original: 2024년 중견리더 과정 만족도 결과를 보여주세요")
        print(f"Expanded: {req1.content}")
        
        # 시스템 응답 추가
        manager.add_response(conv_id, "2024년 중견리더 과정 만족도는 전체 평균 4.2점입니다. 기본역량 14.33%, 리더십역량 14.70%, 직무역량 24.64% 향상되었습니다.")
        
        # 후속 질문
        print("\n=== 테스트 2: 후속 질문 ===")
        req2 = manager.create_query_request(conv_id, "그 중에서 가장 높은 향상도를 보인 역량은?")
        print(f"Query 2 - Follow-up: {req2.follow_up}")
        print(f"Original: 그 중에서 가장 높은 향상도를 보인 역량은?")
        print(f"Expanded: {req2.content}")
        print(f"Context hash: {manager.get_context_hash(conv_id)}")
        
        # 시스템 응답 추가
        manager.add_response(conv_id, "직무역량이 24.64%로 가장 높은 향상도를 보였습니다.")
        
        # 새로운 독립적 질문
        print("\n=== 테스트 3: 독립적 질문 ===")
        req3 = manager.create_query_request(conv_id, "사이버교육 담당자 연락처 알려주세요")
        print(f"Query 3 - Follow-up: {req3.follow_up}")
        print(f"Entities: {req3.context.entities[:5]}")
        
        # 엔티티 추출 테스트
        print("\n=== 테스트 4: 엔티티 추출 ===")
        test_texts = [
            "교육기획담당에게 문의하세요",
            "055-254-2053으로 연락바랍니다",
            "2025년 3월 리더십 교육 일정",
            "나라배움터 직무교육 과정"
        ]
        
        for text in test_texts:
            entities = manager.entity_extractor.extract_entities(text)
            print(f"'{text}' → {entities}")
        
        # 통계 출력
        print("\n=== 테스트 5: 통계 ===")
        stats = manager.get_conversation_stats()
        print(f"통계: {stats}")
        
        # 대화 내용 내보내기
        print("\n=== 테스트 6: 대화 내용 내보내기 ===")
        exported = manager.export_conversation(conv_id)
        if exported:
            print(f"대화 ID: {exported['conversation_id']}")
            print(f"요약: {exported['summary']}")
            print(f"엔티티 수: {len(exported['entities'])}")
            print(f"메시지 수: {len(exported['messages'])}")
        
        print("\n✅ 모든 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


# ================================================================
# 8. 성능 최적화 유틸리티
# ================================================================

class PerformanceMonitor:
    """ContextManager 성능 모니터링"""
    
    def __init__(self):
        self.metrics = {
            'entity_extraction_times': [],
            'summary_generation_times': [],
            'followup_detection_times': [],
            'query_expansion_times': [],
            'total_request_times': []
        }
    
    def record_time(self, operation: str, duration_ms: float):
        """성능 메트릭 기록"""
        if operation in self.metrics:
            self.metrics[operation].append(duration_ms)
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """성능 통계 반환"""
        stats = {}
        for operation, times in self.metrics.items():
            if times:
                stats[operation] = {
                    'avg_ms': sum(times) / len(times),
                    'max_ms': max(times),
                    'min_ms': min(times),
                    'count': len(times)
                }
        return stats


# 성능 모니터 싱글톤
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """성능 모니터 인스턴스 반환"""
    return _performance_monitor


# ================================================================
# 9. 메인 실행부
# ================================================================

if __name__ == "__main__":
    """ContextManager 개발 테스트"""
    import sys
    import os
    
    # 테스트 환경 설정
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("테스트를 위해 .env 파일에 API 키를 설정해주세요.")
        sys.exit(1)
    
    # 테스트 실행
    success = test_context_manager()
    
    if success:
        print("\n🎉 ContextManager 완성 및 테스트 성공!")
        
        # 성능 통계 출력
        perf_stats = get_performance_monitor().get_performance_stats()
        if perf_stats:
            print("\n📊 성능 통계:")
            for operation, stats in perf_stats.items():
                print(f"  {operation}: {stats['avg_ms']:.1f}ms (평균)")
    else:
        print("\n💥 테스트 실패")
        sys.exit(1)
