import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# 환경 변수/Secrets에서 API KEY 가져오기
openai_api_key = st.secrets["OPENAI_API_KEY"]

# 1. 스트리밍 핸들러 정의
class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.text_area = st.empty()
        self.full_text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.full_text += token
        self.text_area.markdown(self.full_text)

# 2. 대화내역 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("📞 공공 민원 챗봇")

# 3. 사용자 입력
user_input = st.chat_input("민원을 입력하세요:")

# 4. 입력시 처리
if user_input:
    st.session_state.chat_history.append(("민원인", user_input))
    handler = StreamHandler()
    llm = ChatOpenAI(
        streaming=True,
        callbacks=[handler],
        openai_api_key=openai_api_key,
        model_name="gpt-4o",  # 원하는 모델 이름 사용 (예: gpt-4o, gpt-4-turbo 등)
        temperature=0.0,
    )
    response = llm.predict(user_input)
    st.session_state.chat_history.append(("챗봇", response))

# 5. 누적 대화 출력
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
