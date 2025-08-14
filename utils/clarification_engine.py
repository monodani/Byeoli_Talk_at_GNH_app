# utils/clarification_engine.py (새 파일)
"""
재질문 엔진: 모호한 질문에 대한 명확화 요청 시스템
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from utils.config import get_openai_api_key

logger = logging.getLogger(__name__)

@dataclass
class ClarificationSuggestion:
    """재질문 제안"""
    question: str
    keywords: List[str]
    domain: str
    confidence: float

class ClarificationEngine:
    """재질문 엔진"""
    
    def __init__(self):
        self.domain_keywords = {
            "satisfaction": {
                "keywords": ["만족도", "평가", "점수", "결과", "교육과정", "교과목", "강의"],
                "clarifications": [
                    "어떤 교육과정의 만족도를 알고 싶으신가요?",
                    "특정 교과목 만족도를 찾고 계신가요?",
                    "전체 교육과정 만족도 순위를 원하시나요?",
                    "만족도 결과를 어떤 기준으로 보고 싶으신가요? (과정별/교과목별/연도별)"
                ]
            },
            "general": {
                "keywords": ["연락처", "담당자", "부서", "전화", "규정", "학칙"],
                "clarifications": [
                    "어떤 부서의 연락처를 찾고 계신가요?",
                    "특정 업무 담당자를 찾으시나요?",
                    "어떤 규정이나 학칙에 대해 궁금하신가요?",
                    "구체적으로 어떤 정보가 필요하신지 말씀해 주세요."
                ]
            },
            "cyber": {
                "keywords": ["사이버", "온라인", "민간위탁", "나라배움터", "교육"],
                "clarifications": [
                    "어떤 종류의 사이버 교육을 찾고 계신가요?",
                    "특정 기간의 교육 일정을 원하시나요?",
                    "민간위탁 교육과 나라배움터 중 어느 것을 찾으시나요?",
                    "직무/소양/시책 교육 중 어떤 분야인가요?"
                ]
            },
            "menu": {
                "keywords": ["메뉴", "식단", "식당", "급식", "점심", "저녁"],
                "clarifications": [
                    "언제의 식단을 확인하고 싶으신가요? (오늘/이번주/특정날짜)",
                    "어떤 시간대의 메뉴를 원하시나요? (조식/중식/석식)",
                    "구내식당 운영시간을 알고 싶으신가요?",
                    "특정 요일의 메뉴를 찾고 계신가요?"
                ]
            },
            "notice": {
                "keywords": ["공지", "안내", "알림", "새로운", "최근"],
                "clarifications": [
                    "어떤 종류의 공지사항을 찾고 계신가요?",
                    "특정 기간의 공지사항을 원하시나요?",
                    "긴급 공지사항을 찾으시나요?",
                    "특정 주제의 안내사항이 필요하신가요?"
                ]
            },
            "publish": {
                "keywords": ["계획서", "평가서", "보고서", "교육계획", "성과"],
                "clarifications": [
                    "어떤 연도의 교육계획서를 원하시나요?",
                    "교육훈련계획서와 종합평가서 중 어느 것을 찾으시나요?",
                    "특정 분야의 교육계획을 원하시나요?",
                    "계획서와 평가서 중 어떤 문서가 필요하신가요?"
                ]
            }
        }
    
    def analyze_query(self, query: str, low_confidence_domains: List[str]) -> Optional[ClarificationSuggestion]:
        """
        쿼리 분석 및 재질문 제안 생성
        
        Args:
            query: 사용자 쿼리
            low_confidence_domains: 낮은 신뢰도를 가진 도메인들
            
        Returns:
            재질문 제안 또는 None
        """
        query_lower = query.lower()
        
        # 1. 키워드 기반 도메인 매칭
        domain_scores = {}
        for domain, info in self.domain_keywords.items():
            score = sum(1 for keyword in info["keywords"] if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score / len(info["keywords"])
        
        # 2. 가장 관련성 높은 도메인 선택
        if not domain_scores:
            return self._generate_general_clarification(query)
        
        best_domain = max(domain_scores.keys(), key=lambda d: domain_scores[d])
        best_score = domain_scores[best_domain]
        
        # 3. 도메인별 맞춤 재질문 생성
        if best_score > 0.2:  # 최소 관련성 임계값
            clarifications = self.domain_keywords[best_domain]["clarifications"]
            
            # 쿼리 길이와 복잡도에 따른 재질문 선택
            if len(query.split()) <= 3:  # 짧은 쿼리
                question = clarifications[0]  # 가장 기본적인 재질문
            else:
                question = clarifications[-1]  # 더 구체적인 재질문
            
            return ClarificationSuggestion(
                question=question,
                keywords=self.domain_keywords[best_domain]["keywords"],
                domain=best_domain,
                confidence=best_score
            )
        
        return self._generate_general_clarification(query)
    
    def _generate_general_clarification(self, query: str) -> ClarificationSuggestion:
        """일반적인 재질문 생성"""
        general_questions = [
            "더 구체적으로 어떤 정보를 원하시나요?",
            "어떤 분야에 대해 궁금하신지 말씀해 주세요.",
            "질문을 좀 더 자세히 설명해 주시겠어요?",
            "다음 중 어떤 것과 관련된 질문인가요?\n• 교육만족도 📊\n• 연락처/규정 📋\n• 사이버교육 💻\n• 구내식당 🍽️\n• 공지사항 📢\n• 교육계획서 📚"
        ]
        
        # 쿼리 길이에 따른 재질문 선택
        if len(query.split()) <= 2:
            question = general_questions[-1]  # 선택지 제공
        else:
            question = general_questions[0]  # 구체화 요청
        
        return ClarificationSuggestion(
            question=question,
            keywords=[],
            domain="general",
            confidence=0.5
        )
    
    def generate_smart_clarification(self, query: str, context: Any = None) -> str:
        """
        LLM 기반 스마트 재질문 생성 (고급 기능)
        
        Args:
            query: 사용자 쿼리
            context: 대화 컨텍스트
            
        Returns:
            생성된 재질문
        """
        try:
            from langchain_openai import ChatOpenAI
            
            api_key = get_openai_api_key()
            if not api_key:
                return self._generate_general_clarification(query).question
            
            llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-4o-mini",
                temperature=0.3
            )
            
            # 컨텍스트 정보 구성
            context_info = ""
            if context and hasattr(context, 'recent_messages') and context.recent_messages:
                recent_msgs = context.recent_messages[-3:]  # 최근 3개 메시지
                context_info = "\n".join([f"- {msg.role}: {msg.content}" for msg in recent_msgs])
            
            prompt = f"""사용자가 경상남도인재개발원 AI 어시스턴트에게 다음과 같은 질문을 했습니다:
"{query}"

이 질문이 모호하거나 추상적이어서 구체적인 답변을 제공하기 어렵습니다.
사용자의 의도를 명확히 파악하기 위한 친근하고 도움이 되는 재질문을 생성해주세요.

가능한 주제 분야:
- 교육만족도 조사 결과
- 연락처 및 담당자 정보  
- 학칙 및 규정
- 사이버 교육 일정
- 구내식당 메뉴
- 공지사항
- 교육계획서 및 평가서

{f"최근 대화 내용:{context_info}" if context_info else ""}

재질문은 한국어로 작성하고, 친근한 말투를 사용하며, 구체적인 선택지나 예시를 포함해주세요."""

            response = llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.warning(f"LLM 기반 재질문 생성 실패: {e}")
            return self._generate_general_clarification(query).question


# handlers/base_handler.py에 추가할 메서드
def _should_request_clarification(self, query: str, confidence: float, results: List) -> bool:
    """재질문이 필요한지 판단"""
    
    # 1. 컨피던스가 매우 낮은 경우
    if confidence < self.confidence_threshold * 0.7:
        return True
    
    # 2. 검색 결과가 없거나 매우 적은 경우
    if len(results) == 0:
        return True
    
    # 3. 쿼리가 너무 짧거나 모호한 경우
    query_tokens = query.strip().split()
    if len(query_tokens) <= 2 and any(word in query.lower() for word in ["뭐", "어떻게", "언제", "어디", "누구"]):
        return True
    
    # 4. 일반적인 단어만 포함된 경우
    generic_words = {"교육", "정보", "알려줘", "궁금", "문의", "질문", "도움"}
    if all(token in generic_words for token in query_tokens if len(token) > 1):
        return True
    
    return False

def _generate_clarification_response(self, query: str, context: Any = None) -> HandlerResponse:
    """재질문 응답 생성"""
    
    from utils.clarification_engine import ClarificationEngine
    
    clarification_engine = ClarificationEngine()
    
    # 스마트 재질문 생성
    clarification_question = clarification_engine.generate_smart_clarification(query, context)
    
    # 도메인별 예시 질문 추가
    domain_examples = {
        "satisfaction": [
            "• 2024년 교육과정 만족도는?",
            "• 교과목 만족도 순위 보여줘",
            "• 소통과 공감 과정 만족도는?"
        ],
        "general": [
            "• 총무담당 연락처 알려줘",
            "• 학칙 출석 규정 설명해줘",
            "• 교육운영팀 담당자는?"
        ],
        "cyber": [
            "• 나라배움터 교육 일정은?",
            "• 민간위탁 교육 목록 보여줘",
            "• 이번 달 사이버교육 과정은?"
        ],
        "menu": [
            "• 오늘 점심 메뉴 뭐야?",
            "• 이번 주 식단표 보여줘",
            "• 구내식당 운영시간은?"
        ],
        "notice": [
            "• 최신 공지사항 있어?",
            "• 중요한 안내사항 알려줘",
            "• 이번 주 새로운 소식은?"
        ],
        "publish": [
            "• 2025년 교육계획 요약해줘",
            "• 2024년 교육성과는?",
            "• 교육훈련계획서 보여줘"
        ]
    }
    
    examples = domain_examples.get(self.domain, domain_examples["general"])
    
    response_text = f"""🤔 {clarification_question}

**💡 이런 식으로 질문해 주시면 더 정확한 답변을 드릴 수 있어요:**
{chr(10).join(examples)}

궁금한 점을 좀 더 구체적으로 말씀해 주시면 최선을 다해 도와드리겠습니다! 😊"""
    
    return HandlerResponse(
        text=response_text,
        confidence=0.95,  # 재질문은 높은 신뢰도
        citations=[],
        handler_name=f"{self.domain}_clarification",
        metadata={
            "type": "clarification_request",
            "original_query": query,
            "domain": self.domain,
            "requires_user_input": True
        }
    )
