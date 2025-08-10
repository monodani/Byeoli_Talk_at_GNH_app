#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - app.py

Streamlit 엔트리포인트: 사용자 인터페이스 및 시스템 통합
- IndexManager 싱글톤 초기화 및 사전 로드
- 대화형 RAG (ConversationContext 관리)  
- 하이브리드 라우팅 + 병렬 실행 통합
- 스트리밍 응답 처리
- 3종 캐시 시스템 통합
- 성능 모니터링 및 디버깅 기능

주요 특징:
- 1초 내 첫 토큰, 전체 2-4초 목표
- 50토큰 단위 스트리밍 출력
- Citation 기반 신뢰성 확보
- 세션별 대화 상태 관리
- 실시간 성능 지표 표시
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import streamlit as st
from streamlit.runtime.caching import cache_data

# 프로젝트 모듈
from utils.contracts import (
    QueryRequest, ConversationContext, ChatTurn, MessageRole,
    create_query_request, normalize_query
)
from utils.router import get_router, analyze_routing_performance
from utils.index_manager import get_index_manager, preload_all_indexes, index_health_check
from utils.config import config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================================================
# 1. Streamlit 기본 설정
# ================================================================

st.set_page_config(
    page_title="벼리 (BYEOLI) - 경상남도인재개발원 AI 어시스턴트",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사용자 정의 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: right;
    }
    .assistant-message {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .citation-box {
        background: #f8f9fa;
        border-left: 3px solid #28a745;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .performance-metrics {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    .status-indicator {
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-healthy { background: #d4edda; color: #155724; }
    .status-degraded { background: #fff3cd; color: #856404; }
    .status-error { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# 2. 시스템 초기화 함수들
# ================================================================

@st.cache_resource
def initialize_system():
    """시스템 컴포넌트 초기화 (캐싱)"""
    logger.info("🚀 시스템 초기화 시작...")
    
    try:
        # IndexManager 사전 로드
        preload_results = preload_all_indexes()
        success_count = sum(1 for success in preload_results.values() if success)
        
        # Router 초기화
        router = get_router()
        
        # 초기화 결과
        init_result = {
            "success": success_count > 0,
            "loaded_domains": f"{success_count}/{len(preload_results)}",
            "preload_results": preload_results,
            "router_ready": router is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ 시스템 초기화 완료: {init_result['loaded_domains']} 도메인 로드")
        return init_result
        
    except Exception as e:
        logger.error(f"❌ 시스템 초기화 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@cache_data(ttl=300)  # 5분 캐시
def get_system_health():
    """시스템 상태 확인 (캐싱)"""
    return index_health_check()

def initialize_session_state():
    """세션 상태 초기화"""
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = ConversationContext()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "performance_metrics" not in st.session_state:
        st.session_state.performance_metrics = []
    
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0

# ================================================================
# 3. 대화 관리 함수들
# ================================================================

def update_conversation_context(user_text: str, assistant_response: str):
    """대화 컨텍스트 업데이트"""
    context = st.session_state.conversation_context
    
    # 메시지 추가
    context.add_message(MessageRole.USER, user_text)
    context.add_message(MessageRole.ASSISTANT, assistant_response)
    
    # 요약 갱신 필요 시 처리
    if context.should_update_summary():
        # 간단한 요약 생성 (실제로는 LLM 사용)
        recent_messages = context.recent_messages[-4:]  # 최근 4턴
        messages_text = " ".join([msg.text[:100] for msg in recent_messages])
        context.summary = f"최근 대화: {messages_text[:500]}..."
        logger.info("💭 대화 요약 갱신됨")

def detect_follow_up(user_text: str) -> bool:
    """후속 질문 감지 (간단한 휴리스틱)"""
    follow_up_indicators = [
        "그리고", "또한", "추가로", "더", "그것", "그거", "이것", "이거",
        "위에서", "앞에서", "이전에", "아까", "방금", "더 자세히"
    ]
    
    return any(indicator in user_text for indicator in follow_up_indicators)

# ================================================================
# 4. UI 렌더링 함수들
# ================================================================

def render_header():
    """헤더 렌더링"""
    st.markdown('<h1 class="main-header">🌟 벼리 (BYEOLI)</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 1.1rem;">'
        '경상남도인재개발원 AI 어시스턴트</p>',
        unsafe_allow_html=True
    )
    
    # 시스템 상태 표시
    health = get_system_health()
    overall_health = health.get('overall_health', 'unknown')
    
    status_class = {
        'healthy': 'status-healthy',
        'degraded': 'status-degraded'
    }.get(overall_health, 'status-error')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f'<div class="status-indicator {status_class}">'
            f'시스템 상태: {overall_health.upper()}</div>',
            unsafe_allow_html=True
        )

def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.header("🛠️ 시스템 정보")
        
        # 시스템 상태
        health = get_system_health()
        st.subheader("📊 상태 요약")
        st.metric("로드된 도메인", health.get('loaded_domains', '0/0'))
        st.metric("총 문서 수", health.get('total_documents', 0))
        st.metric("오류 수", health.get('total_errors', 0))
        
        # 도메인별 상태
        st.subheader("📚 도메인별 상태")
        domains = health.get('domains', {})
        for domain, status in domains.items():
            icon = "✅" if status['loaded'] else "❌"
            doc_count = status['documents_count']
            st.write(f"{icon} **{domain}**: {doc_count}개 문서")
        
        # 대화 통계
        st.subheader("💬 대화 통계")
        st.metric("질문 수", st.session_state.query_count)
        st.metric("대화 턴", len(st.session_state.conversation_context.recent_messages))
        
        # 성능 지표
        if st.session_state.performance_metrics:
            st.subheader("⚡ 성능 지표")
            recent_metrics = st.session_state.performance_metrics[-5:]  # 최근 5개
            avg_time = sum(m.get('total_time_ms', 0) for m in recent_metrics) / len(recent_metrics)
            st.metric("평균 응답 시간", f"{avg_time:.0f}ms")
        
        # 리셋 버튼
        if st.button("🔄 대화 초기화"):
            st.session_state.conversation_context = ConversationContext()
            st.session_state.chat_history = []
            st.session_state.performance_metrics = []
            st.rerun()
        
        # 디버그 정보 (개발 모드)
        if config.APP_MODE == "dev":
            with st.expander("🔧 디버그 정보"):
                st.json(health)

def render_chat_history():
    """채팅 기록 렌더링"""
    st.subheader("💬 대화 기록")
    
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(
                    f'<div class="user-message">'
                    f'<strong>👤 사용자:</strong><br>{message["content"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                # 어시스턴트 응답
                response_data = message.get("response_data", {})
                answer = message["content"]
                citations = response_data.get("citations", [])
                performance = message.get("performance", {})
                
                st.markdown(
                    f'<div class="assistant-message">'
                    f'<strong>🌟 벼리:</strong><br>{answer}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Citation 표시
                if citations:
                    for citation in citations:
                        source_id = citation.get("source_id", "알 수 없음")
                        snippet = citation.get("snippet", "")
                        if snippet:
                            st.markdown(
                                f'<div class="citation-box">'
                                f'<strong>📄 출처:</strong> {source_id}<br>'
                                f'<em>"{snippet}"</em>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

def render_input_section():
    """입력 섹션 렌더링"""
    st.subheader("💭 질문하기")
    
    # 예시 질문들
    example_questions = [
        "2024년 교육과정 만족도 1위는?",
        "학칙에서 징계 관련 규정 알려줘",
        "오늘 구내식당 메뉴 뭐야?",
        "사이버교육 일정 확인하고 싶어",
        "2025년 교육계획 요약해줘",
        "새로운 공지사항 있어?",
        "총무담당 연락처 알려줘"
    ]
    
    # 예시 질문 버튼들
    st.write("**💡 예시 질문들:**")
    cols = st.columns(3)
    for i, question in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(f"💬 {question[:20]}...", key=f"example_{i}"):
                return question
    
    # 메인 입력창
    user_input = st.chat_input("경상남도인재개발원에 대해 무엇이든 물어보세요!")
    
    return user_input

# ================================================================
# 5. 메인 처리 함수들
# ================================================================

async def process_query(user_text: str) -> Dict[str, Any]:
    """
    사용자 질문 처리
    
    Args:
        user_text: 사용자 입력 텍스트
        
    Returns:
        Dict: 처리 결과 및 메타데이터
    """
    try:
        # 후속 질문 감지
        follow_up = detect_follow_up(user_text)
        
        # QueryRequest 생성
        request = create_query_request(
            text=user_text,
            context=st.session_state.conversation_context,
            follow_up=follow_up
        )
        
        # 라우터를 통한 처리
        router = get_router()
        response = await router.route(request)
        
        # 성능 분석
        performance = analyze_routing_performance(response)
        
        # 대화 컨텍스트 업데이트
        update_conversation_context(user_text, response.answer)
        
        # 통계 업데이트
        st.session_state.query_count += 1
        st.session_state.performance_metrics.append(performance)
        
        return {
            "success": True,
            "response": response,
            "performance": performance,
            "follow_up_detected": follow_up
        }
        
    except Exception as e:
        logger.error(f"❌ 쿼리 처리 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback_answer": "죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        }

def add_to_chat_history(user_text: str, result: Dict[str, Any]):
    """채팅 기록에 추가"""
    # 사용자 메시지 추가
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_text,
        "timestamp": datetime.now().isoformat()
    })
    
    # 어시스턴트 응답 추가
    if result["success"]:
        response = result["response"]
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.answer,
            "response_data": {
                "citations": [citation.dict() for citation in response.citations],
                "confidence": response.confidence,
                "handler_id": response.handler_id,
                "elapsed_ms": response.elapsed_ms
            },
            "performance": result["performance"],
            "timestamp": datetime.now().isoformat()
        })
    else:
        # 오류 시 fallback 답변
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result.get("fallback_answer", "오류가 발생했습니다."),
            "error": True,
            "timestamp": datetime.now().isoformat()
        })

def render_streaming_response(response_text: str):
    """스트리밍 응답 렌더링 (50토큰 단위)"""
    placeholder = st.empty()
    
    # 50토큰 단위로 분할하여 점진적 출력
    words = response_text.split()
    displayed_text = ""
    
    for i in range(0, len(words), 50):  # 50단어씩 청크
        chunk = " ".join(words[i:i+50])
        displayed_text += chunk + " "
        
        with placeholder.container():
            st.markdown(
                f'<div class="assistant-message">'
                f'<strong>🌟 벼리:</strong><br>{displayed_text}'
                f'</div>',
                unsafe_allow_html=True
            )
        
        time.sleep(0.1)  # 스트리밍 효과

# ================================================================
# 6. 메인 앱 로직
# ================================================================

def main():
    """메인 애플리케이션 로직"""
    
    # 시스템 초기화
    init_result = initialize_system()
    if not init_result["success"]:
        st.error(f"❌ 시스템 초기화 실패: {init_result.get('error', '알 수 없는 오류')}")
        st.stop()
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # UI 렌더링
    render_header()
    render_sidebar()
    
    # 메인 콘텐츠 영역
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 채팅 기록 표시
        render_chat_history()
        
        # 입력 섹션
        user_input = render_input_section()
        
        # 사용자 입력 처리
        if user_input:
            # 실시간 처리 표시
            with st.spinner("🤔 생각 중..."):
                # 비동기 처리를 동기적으로 실행
                result = asyncio.run(process_query(user_input))
            
            # 채팅 기록에 추가
            add_to_chat_history(user_input, result)
            
            # 성공 시 스트리밍 효과로 마지막 응답 표시
            if result["success"]:
                st.success("✅ 응답 완료!")
                if result.get("follow_up_detected"):
                    st.info("🔄 후속 질문으로 감지되어 컨텍스트를 활용했습니다.")
            else:
                st.error("❌ 처리 중 오류가 발생했습니다.")
            
            # 페이지 새로고침하여 최신 대화 표시
            st.rerun()
    
    with col2:
        # 도움말 및 추가 정보
        render_help_section()

def render_help_section():
    """도움말 섹션 렌더링"""
    st.subheader("📚 사용 가이드")
    
    with st.expander("💡 이용 방법"):
        st.markdown("""
        **벼리에게 이런 질문을 해보세요:**
        
        **📊 만족도 조사**
        - "2024년 교육과정 만족도 순위는?"
        - "교과목 만족도 점수가 가장 높은 과정은?"
        
        **📋 규정 및 연락처**
        - "학칙에서 출석 관련 규정 알려줘"
        - "총무담당 연락처는?"
        - "전결규정에 따른 결재 절차는?"
        
        **🍽️ 구내식당**
        - "오늘 점심 메뉴 뭐야?"
        - "이번 주 식단표 보여줘"
        
        **💻 사이버교육**
        - "나라배움터 교육 일정은?"
        - "민간위탁 사이버교육 목록 보여줘"
        
        **📄 교육계획 및 평가**
        - "2025년 교육훈련계획 요약해줘"
        - "2024년 교육운영 성과는?"
        
        **📢 공지사항**
        - "최신 공지사항 있어?"
        - "교육생 안내사항 알려줘"
        """)
    
    with st.expander("🎯 주요 기능"):
        st.markdown("""
        **🚀 빠른 응답**: 1초 내 첫 답변 시작
        **📚 신뢰할 수 있는 정보**: 공식 문서 기반 답변
        **🔗 출처 제공**: 모든 답변에 근거 자료 표시
        **💬 대화형**: 이전 대화 맥락을 이해하는 연속 대화
        **🎯 정확한 라우팅**: 질문 유형에 맞는 전문 처리
        """)
    
    with st.expander("📞 문의처"):
        st.markdown("""
        **경상남도인재개발원**
        - 📍 주소: 경상남도 진주시 월아산로 2026
        - ☎️ 대표전화: 055-254-2011 (인재개발지원과)
        - ☎️ 교육문의: 055-254-2051 (인재양성과)
        - 🌐 홈페이지: https://www.gyeongnam.go.kr/hrd/
        
        **부서별 연락처**
        - 📋 총무담당: 055-254-2013
        - 📊 평가분석담당: 055-254-2023  
        - 📚 교육기획담당: 055-254-2053
        - 👥 교육운영1담당: 055-254-2063
        - 👥 교육운영2담당: 055-254-2073
        - 💻 사이버담당: 055-254-2083
        """)
    
    # 실시간 상태 표시
    st.subheader("🔄 실시간 상태")
    health = get_system_health()
    
    # 간단한 상태 지표
    if health['overall_health'] == 'healthy':
        st.success("🟢 모든 시스템 정상 동작")
    elif health['overall_health'] == 'degraded':
        st.warning("🟡 일부 시스템 제한적 동작")
    else:
        st.error("🔴 시스템 점검 필요")
    
    st.caption(f"최종 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ================================================================
# 7. 앱 진입점
# ================================================================

if __name__ == "__main__":
    try:
        # 기본 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 메인 앱 실행
        main()
        
    except Exception as e:
        st.error(f"❌ 애플리케이션 실행 중 오류 발생: {e}")
        logger.critical(f"💥 앱 실행 실패: {e}")
        
        # 오류 정보 표시
        with st.expander("🔧 오류 상세 정보"):
            st.code(f"""
오류 타입: {type(e).__name__}
오류 메시지: {str(e)}
발생 시간: {datetime.now().isoformat()}

문제가 지속되면 시스템 관리자에게 문의하세요.
연락처: 055-254-2011
            """)

                
                # 성능 지표 표시
                if performance:
                    confidence = performance.get("final_confidence", 0)
                    elapsed_ms = performance.get("total_time_ms", 0)
                    handler_id = performance.get("selected_handlers", ["unknown"])[0]
                    
                    st.markdown(
                        f'<div class="performance-metrics">'
                        f'💡 신뢰도: {confidence:.2f} | '
                        f'⏱️ 응답시간: {elapsed_ms}ms | '
                        f'🎯 핸들러: {handler_id}'
                        f'</div>',
                        unsafe_allow_html=True
