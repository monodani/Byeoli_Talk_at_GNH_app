#!/usr/bin/env python3
"""
벼리톡@경상남도인재개발원 (경상남도인재개발원 RAG 챗봇) - app.py

Streamlit 엔트리포인트: 사용자 인터페이스 및 시스템 통합
- IndexManager 싱글톤 초기화 및 사전 로드
- 대화형 RAG (ConversationContext 관리)  
- 하이브리드 라우팅 + 병렬 실행 통합
- 단어 단위 스트리밍 응답 처리
- 벼리 캐릭터 이미지 동적 선택
- Graceful Degradation (APP_MODE 기반)
- 3종 캐시 시스템 통합
- 성능 모니터링 및 디버깅 기능

주요 특징:
- 1초 내 첫 토큰, 전체 2-4초 목표
- 단어 단위 자연스러운 스트리밍 출력 (50ms)
- 벼리 캐릭터 33개 이미지 상황별 선택
- Citation 기반 신뢰성 확보
- 세션별 대화 상태 관리 (UUID)
- 실시간 성능 지표 표시
"""

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import streamlit as st
from streamlit.runtime.caching import cache_data

# 프로젝트 모듈
try:
    from utils.contracts import (
        QueryRequest, ConversationContext, ChatTurn, MessageRole,
        HandlerResponse, Citation
    )
    from utils.router import get_router
    from utils.index_manager import get_index_manager, preload_all_indexes, index_health_check
    from utils.config import config
    from utils.context_manager import ContextManager
except ImportError as e:
    st.error(f"❌ 모듈 import 실패: {e}")
    st.stop()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 앱 모드 설정 (환경변수 기반)
APP_MODE = os.environ.get('APP_MODE', 'development')  # production, development
IS_PRODUCTION = APP_MODE == 'production'

# ================================================================
# 1. Streamlit 기본 설정 및 CSS
# ================================================================

st.set_page_config(
    page_title="벼리톡@경상남도인재개발원 - AI 어시스턴트",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 벼리 캐릭터 이미지 매핑 (33개)
BYEOLI_IMAGES = {
    # 감정 및 심리 상태
    "excited": "assets/Byeoli/excited_Byeoli.png",
    "happy": "assets/Byeoli/happy_Byeoli.png",
    "flattered": "assets/Byeoli/flattered_Byeoli.png",
    "sorry": "assets/Byeoli/sorry_Byeoli.png",
    "shameful": "assets/Byeoli/shameful_Byeoli.png", 
    "sullen": "assets/Byeoli/sullen_Byeoli.png",
    "screaming": "assets/Byeoli/screaming_Byeoli.png",
    "worrying": "assets/Byeoli/worrying_Byeoli.png",
    "cold": "assets/Byeoli/feel_cold_Byeoli.png",
    "hot": "assets/Byeoli/feel_hot_Byeoli.png",
    "hungry": "assets/Byeoli/feel_hungry.png",
    
    # 행동 및 활동
    "advicing": "assets/Byeoli/advicing_Byeoli.png",
    "typing": "assets/Byeoli/typing_Byeoli.png",
    "writing": "assets/Byeoli/writing_Byeoli.png",
    "cellphoning": "assets/Byeoli/cellphoning_Byeoli.png",
    "presentating": "assets/Byeoli/presentating_Byeoli.png",
    "hardworking": "assets/Byeoli/hardworking_Byeoli.png",
    "yes_sir": "assets/Byeoli/yes_sir_Byeoli.png",
    "you_call_me": "assets/Byeoli/you_call_me_Byeoli.png",
    "good_night": "assets/Byeoli/good_night_Byeoli.png",
    "go_to_work": "assets/Byeoli/go_to_work_Byeoli.png",
    "getting_off": "assets/Byeoli/getting_off_Byeoli.png",
    
    # 상황 및 자연 현상
    "mistake": "assets/Byeoli/Byeoli_mistake.png",
    "rainy": "assets/Byeoli/rainy_Byeoli.png",
    "snowy": "assets/Byeoli/snowy_Byeoli.png",
    "thunder": "assets/Byeoli/thunder_Byeoli.png",
    "dry_day": "assets/Byeoli/dry_day_Byeoli.png",
    "gale": "assets/Byeoli/gale_and_Byeoli.png",
    "typhoon": "assets/Byeoli/typooon_Byeoli.png",
    "masked": "assets/Byeoli/Masked_Byeoli_for_dust.png",
    
    # 기본값
    "default": "assets/Byeoli/advicing_Byeoli.png"
}

# 사용자 정의 CSS
st.markdown("""
<style>
    /* 메인 헤더 스타일 */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* 채팅 컨테이너 */
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 2px solid #e1e5e9;
        border-radius: 15px;
        margin-bottom: 1rem;
        background: #fafbfc;
    }
    
    /* 사용자 메시지 말풍선 */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.8rem 0 0.8rem auto;
        max-width: 75%;
        text-align: right;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        position: relative;
    }
    
    .user-message::after {
        content: '';
        position: absolute;
        bottom: 0;
        right: -8px;
        width: 0;
        height: 0;
        border: 8px solid transparent;
        border-top-color: #764ba2;
        border-bottom: 0;
        margin-left: -8px;
        margin-bottom: -8px;
    }
    
    /* 벼리 메시지 말풍선 */
    .assistant-message {
        background: white;
        color: #2c3e50;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.8rem auto 0.8rem 0;
        max-width: 75%;
        border: 2px solid #e1e5e9;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        position: relative;
    }
    
    .assistant-message::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: -8px;
        width: 0;
        height: 0;
        border: 8px solid transparent;
        border-top-color: white;
        border-bottom: 0;
        margin-right: -8px;
        margin-bottom: -8px;
    }
    
    /* 벼리 캐릭터 이미지 */
    .byeoli-avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        margin-right: 10px;
        vertical-align: top;
        border: 2px solid #e1e5e9;
    }
    
    /* Citation 박스 */
    .citation-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-left: 4px solid #28a745;
        padding: 0.8rem;
        margin: 0.8rem 0;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #2c3e50;
    }
    
    /* 성능 지표 */
    .performance-metrics {
        font-size: 0.75rem;
        color: #6c757d;
        margin-top: 0.5rem;
        text-align: right;
        font-style: italic;
    }
    
    /* 상태 표시기 */
    .status-indicator {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .status-healthy { 
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
        color: #155724; 
    }
    
    .status-degraded { 
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
        color: #856404; 
    }
    
    .status-error { 
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); 
        color: #721c24; 
    }
    
    /* 입력 섹션 스타일 */
    .input-section {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #e1e5e9;
        margin-top: 1rem;
    }
    
    /* 버튼 스타일 개선 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* 사이드바 스타일 */
    .sidebar-info {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* 애니메이션 효과 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* 타이핑 애니메이션 */
    @keyframes typing {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }
    
    .typing-indicator {
        animation: typing 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# 2. 벼리 캐릭터 이미지 선택 로직
# ================================================================

def get_byeoli_image(response: HandlerResponse = None, answer: str = "") -> str:
    """
    응답 내용과 상황에 따라 적절한 벼리 캐릭터 이미지 선택
    
    Args:
        response: 핸들러 응답 객체
        answer: 응답 텍스트
        
    Returns:
        str: 이미지 파일 경로
    """
    try:
        # 응답 객체가 있으면 해당 정보 우선 활용
        if response:
            answer = response.answer
            handler_id = getattr(response, 'handler_id', '')
            confidence = getattr(response, 'confidence', 1.0)
            
            # fallback 핸들러이거나 낮은 컨피던스
            if handler_id == 'fallback' or confidence < 0.3:
                if any(word in answer for word in ['죄송', '미안', '오류', '실패', '문제']):
                    return BYEOLI_IMAGES["sorry"]
                else:
                    return BYEOLI_IMAGES["mistake"]
        
        # 텍스트 내용 기반 이미지 선택
        answer_lower = answer.lower()
        
        # 에러/실수 관련
        if any(word in answer for word in ['죄송', '미안', '오류', '실패', '문제', '찾을 수 없', '어려움']):
            return BYEOLI_IMAGES["sorry"]
        
        # 긍정적 감정
        if any(word in answer for word in ['좋', '훌륭', '완료', '성공', '감사', '기쁨', '축하']):
            return BYEOLI_IMAGES["happy"]
        
        # 안내/조언
        if any(word in answer for word in ['안내', '문의', '연락', '담당', '도움', '방법', '절차']):
            return BYEOLI_IMAGES["advicing"]
        
        # 업무 관련
        if any(word in answer for word in ['업무', '작업', '처리', '진행', '개발', '분석']):
            return BYEOLI_IMAGES["typing"]
        
        # 발표/설명
        if any(word in answer for word in ['발표', '설명', '소개', '계획', '보고', '평가']):
            return BYEOLI_IMAGES["presentating"]
        
        # 글쓰기/문서
        if any(word in answer for word in ['작성', '문서', '보고서', '계획서', '평가서']):
            return BYEOLI_IMAGES["writing"]
        
        # 날씨 관련
        if any(word in answer for word in ['비', '우천', '강우']):
            return BYEOLI_IMAGES["rainy"]
        elif any(word in answer for word in ['눈', '설', '겨울']):
            return BYEOLI_IMAGES["snowy"]
        elif any(word in answer for word in ['맑', '화창', '좋은날씨']):
            return BYEOLI_IMAGES["dry_day"]
        
        # 식사 관련
        if any(word in answer for word in ['식단', '메뉴', '식사', '밥', '급식']):
            return BYEOLI_IMAGES["excited"]
        
        # 인사/마무리
        if any(word in answer for word in ['안녕', '좋은하루', '수고', '마무리', '끝']):
            return BYEOLI_IMAGES["good_night"]
        
        # 기본값 (조언하는 벼리)
        return BYEOLI_IMAGES["advicing"]
        
    except Exception as e:
        logger.error(f"이미지 선택 중 오류: {e}")
        return BYEOLI_IMAGES["default"]

# ================================================================
# 3. 시스템 초기화 (Graceful Degradation)
# ================================================================

@st.cache_data(ttl=3600)  # 1시간 캐시
def initialize_system() -> Dict[str, Any]:
    """
    시스템 초기화 (APP_MODE에 따른 Graceful Degradation)
    
    Returns:
        Dict[str, Any]: 초기화 결과
    """
    try:
        logger.info(f"🚀 벼리톡 시스템 초기화 시작 (모드: {APP_MODE})")
        
        # 1. 환경변수 검증
        if not config.OPENAI_API_KEY:
            error_msg = "OPENAI_API_KEY가 설정되지 않았습니다."
            if IS_PRODUCTION:
                return {"success": False, "error": error_msg, "mode": "fallback"}
            else:
                raise ValueError(f"개발 환경: {error_msg}")
        
        # 2. IndexManager 초기화 시도
        try:
            index_manager = get_index_manager()
            preload_result = preload_all_indexes()
            
            if not preload_result["success"]:
                logger.warning(f"IndexManager 초기화 실패: {preload_result['error']}")
                if IS_PRODUCTION:
                    return {
                        "success": True, 
                        "mode": "limited",
                        "index_loaded": False,
                        "warning": "일부 기능이 제한됩니다."
                    }
                else:
                    raise Exception(f"개발 환경: IndexManager 실패 - {preload_result['error']}")
            
            # 3. Router 초기화
            router = get_router()
            
            # 4. 건강 상태 체크
            health_status = index_health_check()
            
            return {
                "success": True,
                "mode": "full",
                "index_loaded": True,
                "router_loaded": True,
                "health_status": health_status,
                "loaded_indexes": preload_result.get("loaded_indexes", []),
                "performance": preload_result.get("performance", {})
            }
            
        except Exception as e:
            logger.error(f"시스템 초기화 중 오류: {e}")
            if IS_PRODUCTION:
                # 운영 환경: 제한적 서비스 제공
                return {
                    "success": True,
                    "mode": "fallback",
                    "index_loaded": False,
                    "error": str(e),
                    "warning": "현재 기본 서비스만 이용 가능합니다."
                }
            else:
                # 개발 환경: 상세 에러 표시
                return {
                    "success": False, 
                    "error": f"개발 환경 디버깅: {str(e)}\n{traceback.format_exc()}"
                }
                
    except Exception as e:
        error_msg = f"치명적 초기화 오류: {str(e)}"
        logger.critical(error_msg)
        return {"success": False, "error": error_msg}

# ================================================================
# 4. 세션 상태 관리 (UUID 기반)
# ================================================================

def initialize_session_state():
    """세션 상태 초기화 (UUID 기반 conversation_id)"""
    
    # 기본 세션 상태 초기화
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    
    # ContextManager 초기화 시 예외 처리
    if 'context_manager' not in st.session_state:
        try:
            st.session_state.context_manager = ContextManager()
        except Exception as e:
            logger.warning(f"ContextManager 초기화 실패: {e}")
            # Graceful Degradation: 기본 대화 컨텍스트만 사용
            st.session_state.context_manager = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = ConversationContext(
            conversation_id=st.session_state.conversation_id,
            session_id=str(uuid.uuid4())
        )
    
    if 'system_status' not in st.session_state:
        st.session_state.system_status = initialize_system()
    
    if 'performance_stats' not in st.session_state:
        st.session_state.performance_stats = {
            "total_queries": 0,
            "avg_response_time": 0,
            "success_rate": 100,
            "last_query_time": None
        }

def reset_session():
    """세션 초기화 (새 대화 시작)"""
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.conversation_context = ConversationContext(
        conversation_id=st.session_state.conversation_id,
        session_id=str(uuid.uuid4())
    )
    
    # ContextManager 컨텍스트도 초기화
    if st.session_state.context_manager:
        try:
            st.session_state.context_manager.get_or_create_context(
                st.session_state.conversation_id
            )
        except Exception as e:
            logger.warning(f"ContextManager 세션 초기화 실패: {e}")
    
    st.session_state.performance_stats = {
        "total_queries": 0,
        "avg_response_time": 0,
        "success_rate": 100,
        "last_query_time": None
    }
    logger.info(f"🔄 새로운 세션 시작: {st.session_state.conversation_id}")

# ================================================================
# 5. UI 렌더링 함수들
# ================================================================

def render_header():
    """메인 헤더 렌더링 (벼리 캐릭터 포함)"""
    
    col1, col2 = st.columns([1, 8])
    
    with col1:
        # 메인 벼리 캐릭터 이미지
        if Path("assets/Byeoli/advicing_Byeoli.png").exists():
            st.image("assets/Byeoli/advicing_Byeoli.png", width=100)
        else:
            st.markdown("🌟", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="main-header fade-in">
            <h1>🌟 벼리톡@경상남도인재개발원</h1>
            <p>경상남도인재개발원 AI 어시스턴트 - 경상남도인재개발원에 대한 모든 궁금증, 벼리에게 물어보세요!</p>
        </div>
        """, unsafe_allow_html=True)

def render_sidebar():
    """사이드바 렌더링 (도움말 + 상태 정보)"""

    # --- 시스템 상태 확인 (클로드 제안 통합) ---
    st.sidebar.markdown("### ⚙️ 시스템 상태")
    if "OPENAI_API_KEY" in st.secrets:
        st.sidebar.success("✅ Streamlit Secrets에서 API 키 로드됨")
        # 보안을 위해 API 키 일부만 표시
        api_key_part = st.secrets.get("OPENAI_API_KEY", "")[:10]
        st.sidebar.info(f"🔑 API Key: `{api_key_part}...`")
    else:
        st.sidebar.warning("⚠️ Streamlit Secrets 없음 (로컬 환경)")
    st.sidebar.markdown("---")
    
    st.image(str(config.ROOT_DIR / 'assets/images/logo.png'))
    st.markdown("### 챗봇 상태")
    
    # 시스템 초기화 및 상태 정보 세션에 저장
    # 초기화 로직은 이 곳에서 한 번만 실행되도록 유지
    with st.spinner("시스템 초기화 중..."):
        # index_health_check()의 결과를 st.session_state에 직접 저장
        st.session_state.system_status = index_health_check()
    
    # st.sidebar 컨텍스트를 한 번만 사용하여 모든 사이드바 요소를 렌더링
    with st.sidebar:
        st.markdown("### 🎯 챗봇 상태") # '시스템 상태'와 중복되지 않도록 제목을 변경
        
        # 시스템 상태 표시
        status = st.session_state.system_status
        if status["success"]:
            if status.get("mode") == "full":
                st.markdown('<div class="status-indicator status-healthy">🟢 정상 운영</div>', unsafe_allow_html=True)
            elif status.get("mode") == "limited":
                st.markdown('<div class="status-indicator status-degraded">🟡 제한적 서비스</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-indicator status-error">🔴 기본 서비스</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-error">🔴 시스템 오류</div>', unsafe_allow_html=True)
        
        # 성능 통계
        stats = st.session_state.performance_stats
        st.markdown("### 📊 성능 지표")
        st.metric("총 질문 수", stats["total_queries"])
        st.metric("평균 응답시간", f"{stats['avg_response_time']:.2f}초")
        st.metric("성공률", f"{stats['success_rate']:.1f}%")
        
        # 대화 정보
        st.markdown("### 💬 대화 정보")
        st.write(f"**세션 ID**: `{st.session_state.conversation_id[:8]}...`")
        st.write(f"**대화 횟수**: {len(st.session_state.chat_history) // 2}회")
        
        # ContextManager 상태 표시
        if st.session_state.context_manager:
            st.markdown('<div class="status-indicator status-healthy">🤖 고급 컨텍스트 관리</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-degraded">🔧 기본 컨텍스트 관리</div>', unsafe_allow_html=True)
        
        # 세션 초기화 버튼
        if st.button("🔄 새 대화 시작", use_container_width=True):
            reset_session()
            st.rerun()
        
        # 도움말
        st.markdown("---")
        st.markdown("### 📚 사용 가이드")
        
        with st.expander("💡 질문 예시"):
            st.markdown("""
            **📊 만족도 조사**
            - 2024년 교육과정 만족도는?
            - 교과목 만족도 순위 보여줘
            
            **📋 규정 및 연락처**
            - 학칙 출석 규정 알려줘
            - 총무담당 연락처는?
            
            **🍽️ 구내식당**
            - 오늘 점심 메뉴 뭐야?
            - 이번 주 식단표 보여줘
            
            **💻 사이버교육**
            - 나라배움터 교육 일정은?
            - 민간위탁 교육 목록은?
            
            **📄 교육계획 및 평가**
            - 2025년 교육계획 요약해줘
            - 2024년 교육성과는?
            
            **📢 공지사항**
            - 최신 공지사항 있어?
            - 벼리 캐릭터 소개해줘
            """)
        
        with st.expander("⚙️ 시스템 정보"):
            if status.get("loaded_indexes"):
                st.write("**로드된 인덱스:**")
                for idx in status["loaded_indexes"]:
                    st.write(f"- {idx}")
            
            if status.get("performance"):
                perf = status["performance"]
                st.write(f"**초기화 시간**: {perf.get('total_time', 0):.2f}초")
                st.write(f"**메모리 사용량**: {perf.get('memory_usage', 'N/A')}")

def render_chat_history():
    """채팅 기록 표시 (말풍선 스타일)"""
    
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        # 초기 환영 메시지
        welcome_image = get_byeoli_image(answer="안녕하세요! 반가워요!")
        
        st.markdown(f"""
        <div class="assistant-message fade-in">
            <img src="{welcome_image}" class="byeoli-avatar" onerror="this.style.display='none'">
            <strong>🌟 벼리:</strong><br>
            안녕하세요! 경상남도인재개발원 AI 어시스턴트 벼리입니다! 🌟<br><br>
            교육과정 및 강의실 정보, 학칙, 구내식당 식단표, 공지사항 등 궁금한 것이 있으시면 언제든 물어보세요!<br>
            어떤 도움이 필요하신가요?
        </div>
        """, unsafe_allow_html=True)
    else:
        # 기존 대화 기록 표시
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-message fade-in">
                    <strong>👤 사용자:</strong><br>{msg["content"]}
                </div>
                """, unsafe_allow_html=True)
                
            else:  # assistant
                # 응답에 따른 벼리 이미지 선택
                byeoli_image = get_byeoli_image(answer=msg["content"])
                
                # Citation 처리
                citations_html = ""
                if msg.get("citations"):
                    citations_html = "<div class='citation-box'><strong>📚 참고자료:</strong><br>"
                    for idx, citation in enumerate(msg["citations"][:3], 1):
                        source = citation.get("source_id", "알 수 없음")
                        snippet = citation.get("snippet", "")
                        if snippet:
                            citations_html += f"{idx}. {source}: {snippet[:100]}...<br>"
                        else:
                            citations_html += f"{idx}. {source}<br>"
                    citations_html += "</div>"
                
                # 성능 지표
                performance_html = ""
                if msg.get("elapsed_ms"):
                    confidence = msg.get("confidence", 0)
                    handler = msg.get("handler_id", "unknown")
                    performance_html = f"""
                    <div class="performance-metrics">
                        ⏱️ {msg['elapsed_ms']}ms | 🎯 {confidence:.2f} | 🔧 {handler}
                    </div>
                    """
                
                st.markdown(f"""
                <div class="assistant-message fade-in">
                    <img src="{byeoli_image}" class="byeoli-avatar" onerror="this.style.display='none'">
                    <strong>🌟 벼리:</strong><br>
                    {msg["content"]}
                    {citations_html}
                    {performance_html}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_input_section() -> Optional[str]:
    """입력 섹션 렌더링"""
    
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    # 입력 폼
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "🤔 무엇이 궁금하신가요?",
                placeholder="예: 오늘 점심 메뉴 뭐야?, 2024년 교육 만족도는?, 학칙 출석 규정 알려줘",
                label_visibility="collapsed"
            )
        
        with col2:
            submitted = st.form_submit_button("📤 전송", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if submitted and user_input.strip():
        return user_input.strip()
    
    return None

def render_streaming_response(response_text: str, byeoli_image: str = None):
    """단어 단위 스트리밍 응답 렌더링 (수정된 버전)"""
    
    if not byeoli_image:
        byeoli_image = get_byeoli_image(answer=response_text)
    
    placeholder = st.empty()
    
    # 단어 단위로 분할하여 점진적 출력
    words = response_text.split()
    displayed_text = ""
    
    for i, word in enumerate(words):
        displayed_text += word + " "
        
        with placeholder.container():
            st.markdown(f"""
            <div class="assistant-message fade-in">
                <img src="{byeoli_image}" class="byeoli-avatar" onerror="this.style.display='none'">
                <strong>🌟 벼리:</strong><br>
                {displayed_text}<span class="typing-indicator">▊</span>
            </div>
            """, unsafe_allow_html=True)
        
        time.sleep(0.05)  # 단어당 50ms 지연
    
    # 최종 응답 (타이핑 커서 제거)
    with placeholder.container():
        st.markdown(f"""
        <div class="assistant-message fade-in">
            <img src="{byeoli_image}" class="byeoli-avatar" onerror="this.style.display='none'">
            <strong>🌟 벼리:</strong><br>
            {displayed_text}
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# 6. 비즈니스 로직 (쿼리 처리)
# ================================================================

async def process_query(user_input: str) -> Dict[str, Any]:
    """
    사용자 쿼리 처리 (비동기)
    
    Args:
        user_input: 사용자 입력
        
    Returns:
        Dict[str, Any]: 처리 결과
    """
    start_time = time.time()
    
    try:
        logger.info(f"🔍 쿼리 처리 시작: '{user_input}'")
        
        # 1. 쿼리 요청 객체 생성
        query_request = QueryRequest(
            text=user_input,
            context=st.session_state.conversation_context,
            trace_id=str(uuid.uuid4())[:8]
        )
        
        # 2. ContextManager 사용 가능 여부 확인 및 컨텍스트 업데이트
        if st.session_state.context_manager:
            try:
                # 대화 컨텍스트 업데이트 (ContextManager 활용)
                updated_context = st.session_state.context_manager.update_context(
                    st.session_state.conversation_id,
                    MessageRole.USER,
                    user_input
                )
                query_request.context = updated_context
                st.session_state.conversation_context = updated_context
            except Exception as e:
                logger.warning(f"ContextManager 컨텍스트 업데이트 실패: {e}")
                # 기본 컨텍스트 수동 업데이트
                st.session_state.conversation_context.add_message(MessageRole.USER, user_input)
        else:
            # ContextManager 없이 기본 컨텍스트 업데이트
            st.session_state.conversation_context.add_message(MessageRole.USER, user_input)
        
        # 3. 시스템 상태에 따른 처리 분기
        system_status = st.session_state.system_status
        
        if not system_status["success"] or system_status.get("mode") == "fallback":
            # Fallback 모드: 기본 응답만 제공
            return await _handle_fallback_mode(user_input, start_time)
        
        # 3. 정상 모드: Router를 통한 처리
        try:
            router = get_router()
            response = await asyncio.to_thread(router.route, query_request)
            
            # 4. 응답 처리 및 컨텍스트 업데이트
            elapsed_time = time.time() - start_time
            
            # 대화 컨텍스트 업데이트 (ContextManager 또는 기본 방식)
            if st.session_state.context_manager:
                try:
                    updated_context = st.session_state.context_manager.update_context(
                        st.session_state.conversation_id,
                        MessageRole.ASSISTANT,
                        response.answer
                    )
                    st.session_state.conversation_context = updated_context
                except Exception as e:
                    logger.warning(f"응답 컨텍스트 업데이트 실패: {e}")
                    st.session_state.conversation_context.add_message(MessageRole.ASSISTANT, response.answer)
            else:
                st.session_state.conversation_context.add_message(MessageRole.ASSISTANT, response.answer)
            
            # 성능 통계 업데이트
            _update_performance_stats(elapsed_time, True)
            
            return {
                "success": True,
                "response": response,
                "elapsed_time": elapsed_time,
                "follow_up_detected": query_request.follow_up
            }
            
        except asyncio.TimeoutError:
            logger.warning("⏰ 쿼리 처리 타임아웃")
            return await _handle_timeout_error(user_input, start_time)
            
        except Exception as e:
            logger.error(f"❌ 쿼리 처리 중 오류: {e}")
            return await _handle_processing_error(user_input, str(e), start_time)
    
    except Exception as e:
        logger.error(f"💥 치명적 쿼리 처리 오류: {e}")
        elapsed_time = time.time() - start_time
        _update_performance_stats(elapsed_time, False)
        
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time
        }

async def _handle_fallback_mode(user_input: str, start_time: float) -> Dict[str, Any]:
    """Fallback 모드 처리"""
    
    elapsed_time = time.time() - start_time
    _update_performance_stats(elapsed_time, True)
    
    # 키워드 기반 기본 응답
    fallback_responses = {
        "인사": ["안녕", "안녕하세요", "반갑", "처음", "hello", "hi"],
        "감사": ["감사", "고마워", "고맙", "thanks", "thank you"],
        "식단": ["식단", "메뉴", "밥", "식사", "점심", "저녁"],
        "연락처": ["연락처", "전화", "문의", "담당자"],
        "교육": ["교육", "과정", "훈련", "수업", "강의"],
        "만족도": ["만족도", "평가", "설문", "조사"]
    }
    
    response_text = "안녕하세요! 현재 시스템 점검 중으로 기본 서비스만 제공됩니다."
    
    for category, keywords in fallback_responses.items():
        if any(keyword in user_input for keyword in keywords):
            if category == "인사":
                response_text = "안녕하세요! 경상남도인재개발원 AI 어시스턴트 벼리입니다! 🌟"
            elif category == "감사":
                response_text = "천만에요! 언제든 도움이 필요하시면 말씀해 주세요! 😊"
            elif category == "식단":
                response_text = "구내식당 관련 문의는 인재개발지원과 총무담당(055-254-2096)으로 연락해 주세요."
            elif category == "연락처":
                response_text = "경상남도인재개발원 대표번호: 055-254-2051입니다."
            elif category == "교육":
                response_text = "교육과정 운영 관련 문의는 인재양성과 교육기획담당(055-254-2051)으로 연락해 주세요."
            elif category == "평가 및 만족도":
                response_text = "평가 및 만족도 조사 관련은 인재개발지원과 평가분석담당(055-254-2021)으로 문의해 주세요."
            elif category == "보건소·숙소동 운영, 차량지원 및 시설 관리":
                response_text = "보건소 및 숙소동, 차량지원 및 시설 관리 등 관련은 인재개발지원과 총무담당(055-254-2011)으로 문의해 주세요."                
            break
    
    # 가상 응답 객체 생성
    mock_response = type('MockResponse', (), {
        'answer': response_text,
        'citations': [],
        'confidence': 0.5,
        'handler_id': 'fallback',
        'elapsed_ms': int(elapsed_time * 1000)
    })()
    
    return {
        "success": True,
        "response": mock_response,
        "elapsed_time": elapsed_time,
        "mode": "fallback"
    }

async def _handle_timeout_error(user_input: str, start_time: float) -> Dict[str, Any]:
    """타임아웃 에러 처리"""
    
    elapsed_time = time.time() - start_time
    _update_performance_stats(elapsed_time, False)
    
    error_response = type('ErrorResponse', (), {
        'answer': "죄송합니다. 응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.",
        'citations': [],
        'confidence': 0.0,
        'handler_id': 'timeout',
        'elapsed_ms': int(elapsed_time * 1000)
    })()
    
    return {
        "success": False,
        "response": error_response,
        "elapsed_time": elapsed_time,
        "error": "timeout"
    }

async def _handle_processing_error(user_input: str, error_msg: str, start_time: float) -> Dict[str, Any]:
    """처리 오류 핸들링"""
    
    elapsed_time = time.time() - start_time
    _update_performance_stats(elapsed_time, False)
    
    if IS_PRODUCTION:
        # 운영 환경: 친절한 에러 메시지
        friendly_msg = "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주시거나, 인재개발지원과(055-254-2011) 또는 인재양성과(055-254-2051)로 문의해 주세요."
    else:
        # 개발 환경: 상세 에러 정보
        friendly_msg = f"개발 환경 오류: {error_msg}"
    
    error_response = type('ErrorResponse', (), {
        'answer': friendly_msg,
        'citations': [],
        'confidence': 0.0,
        'handler_id': 'error',
        'elapsed_ms': int(elapsed_time * 1000)
    })()
    
    return {
        "success": False,
        "response": error_response,
        "elapsed_time": elapsed_time,
        "error": error_msg
    }

def _update_performance_stats(elapsed_time: float, success: bool):
    """성능 통계 업데이트"""
    
    stats = st.session_state.performance_stats
    
    stats["total_queries"] += 1
    stats["last_query_time"] = datetime.now()
    
    # 평균 응답시간 계산 (이동 평균)
    if stats["avg_response_time"] == 0:
        stats["avg_response_time"] = elapsed_time
    else:
        stats["avg_response_time"] = (stats["avg_response_time"] * 0.8 + elapsed_time * 0.2)
    
    # 성공률 계산 (최근 100개 기준)
    if success:
        stats["success_rate"] = min(stats["success_rate"] + 0.5, 100)
    else:
        stats["success_rate"] = max(stats["success_rate"] - 2, 0)

def add_to_chat_history(user_input: str, result: Dict[str, Any]):
    """채팅 기록에 추가"""
    
    # 사용자 메시지 추가
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now()
    })
    
    # 어시스턴트 응답 추가
    if result["success"] and "response" in result:
        response = result["response"]
        
        # Citation 변환
        citations = []
        if hasattr(response, 'citations') and response.citations:
            citations = [
                {
                    "source_id": getattr(citation, 'source_id', ''),
                    "snippet": getattr(citation, 'snippet', '')
                }
                for citation in response.citations
            ]
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.answer,
            "citations": citations,
            "confidence": getattr(response, 'confidence', 0.0),
            "handler_id": getattr(response, 'handler_id', 'unknown'),
            "elapsed_ms": getattr(response, 'elapsed_ms', 0),
            "timestamp": datetime.now()
        })
    else:
        # 오류 응답
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": result.get("error", "알 수 없는 오류가 발생했습니다."),
            "citations": [],
            "confidence": 0.0,
            "handler_id": "error",
            "elapsed_ms": int(result.get("elapsed_time", 0) * 1000),
            "timestamp": datetime.now()
        })

# ================================================================
# 7. 메인 앱 실행 로직
# ================================================================

def main():
    """메인 애플리케이션 로직"""
    
    try:
        # 세션 상태 초기화
        initialize_session_state()
        
        # UI 렌더링
        render_header()
        render_sidebar()
        
        # 메인 콘텐츠 영역
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 채팅 기록 표시
            render_chat_history()
            
            # 입력 섹션
            user_input = render_input_section()
            
            # 사용자 입력 처리
            if user_input:
                # 실시간 처리 표시
                with st.spinner("🤔 벼리가 생각하고 있어요..."):
                    # 비동기 처리를 동기적으로 실행
                    result = asyncio.run(process_query(user_input))
                
                # 채팅 기록에 추가
                add_to_chat_history(user_input, result)
                
                # 성공 시 스트리밍 효과로 마지막 응답 표시
                if result["success"] and "response" in result:
                    response = result["response"]
                    
                    # 상황에 맞는 벼리 이미지 선택
                    byeoli_image = get_byeoli_image(response, response.answer)
                    
                    # 스트리밍 응답 표시
                    st.success("✅ 응답 완료!")
                    
                    # 추가 정보 표시
                    if result.get("follow_up_detected"):
                        st.info("🔄 후속 질문으로 감지되어 대화 맥락을 활용했습니다.")
                    
                    if result.get("mode") == "fallback":
                        st.warning("⚠️ 현재 기본 서비스 모드로 운영 중입니다.")
                        
                else:
                    st.error("❌ 처리 중 오류가 발생했습니다.")
                    if not IS_PRODUCTION and result.get("error"):
                        st.code(result["error"])
                
                # 페이지 새로고침하여 최신 대화 표시
                st.rerun()
        
        with col2:
            # 시스템 상태가 불안정한 경우 알림
            status = st.session_state.system_status
            if not status["success"]:
                st.error("🚨 시스템 오류")
                st.write("관리자에게 문의해 주세요.")
                if not IS_PRODUCTION:
                    st.code(status.get("error", ""))
            
            elif status.get("mode") == "limited":
                st.warning("⚠️ 제한된 서비스")
                st.write(status.get("warning", ""))
            
            elif status.get("mode") == "fallback":
                st.info("ℹ️ 기본 서비스 모드")
                st.write("핵심 기능만 이용 가능합니다.")
    
    except Exception as e:
        st.error("💥 애플리케이션 치명적 오류")
        logger.critical(f"메인 앱 실행 오류: {e}")
        
        if not IS_PRODUCTION:
            st.code(f"개발 환경 오류:\n{traceback.format_exc()}")
        else:
            st.write("시스템 관리자에게 문의해 주세요.")
            st.write("**시스템 관리자 연락처**: 055-254-2023")

# ================================================================
# 8. 애플리케이션 진입점
# ================================================================

if __name__ == "__main__":
    try:
        # 로깅 설정
        if IS_PRODUCTION:
            logging.getLogger().setLevel(logging.WARNING)
        else:
            logging.getLogger().setLevel(logging.INFO)
        
        # 메인 앱 실행
        main()
        
    except KeyboardInterrupt:
        logger.info("👋 사용자에 의해 앱 종료")
    except Exception as e:
        logger.critical(f"💥 앱 시작 실패: {e}")
        st.error("애플리케이션을 시작할 수 없습니다.")
        
        if not IS_PRODUCTION:
            st.code(traceback.format_exc())
