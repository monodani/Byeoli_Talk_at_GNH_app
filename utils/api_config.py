# utils/api_config.py (새 파일)
import os
import streamlit as st

def get_openai_api_key():
    """
    API 키 가져오기 (우선순위: Streamlit Secrets > 환경변수 > .env)
    """
    # 1. Streamlit Secrets 확인
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            return api_key
    except:
        pass
    
    # 2. 환경변수 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # 3. .env 파일 확인 (로컬 개발용)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key
    except:
        pass
    
    return None

def init_openai_client():
    """OpenAI 클라이언트 초기화"""
    import openai
    
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    return openai.OpenAI(api_key=api_key)
