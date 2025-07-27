import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

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

# 3. 사용자 입력
user_input = st.chat_input("민원을 입력하세요")

# 4. 입력시 처리
if user_input:
    st.session_state.chat_history.append(("민원인", user_input))
    handler = StreamHandler()
    llm = ChatOpenAI(streaming=True, callbacks=[handler])
    response = llm.predict(user_input)
    st.session_state.chat_history.append(("챗봇", response))

# 5. 누적 대화 출력
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
