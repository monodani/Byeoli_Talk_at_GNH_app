import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage

# 1. API KEY 가져오기
openai_api_key = st.secrets["OPENAI_API_KEY"]

# 2. PDF에서 데이터 추출 및 벡터스토어 생성
@st.cache_resource(show_spinner=True)
def load_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(pages, embeddings)
    return vectorstore

vectorstore = load_vectorstore("통계가이드.pdf")

# 3. 프롬프트 템플릿 정의
prompt = PromptTemplate.from_template("""
SYSTEM: 당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 '주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다' 라고 답하세요.
기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요. 출처(page, source)를 답변에 포함하세요. 답변은 한글로 답변해 주세요.

HUMAN:
#Question: {question}

#Context: {context}

#Answer:
""")

# 4. 스트리밍 핸들러 클래스 정의
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.generated_text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.generated_text += token
        self.placeholder.markdown(self.generated_text)

# 5. 대화 내역 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 6. 상단 로고 및 제목 구성
col1, col2 = st.columns([1, 6])
with col1:
    st.image("byeory.png", width=60)  # 로고 이미지 파일
with col2:
    st.markdown("### 벼리봇@경남인재개발원, 무엇이든 물으세요!")

# 7. 사용자 입력
user_input = st.chat_input("경남인재개발원에 대해 벼리에게 물어보세요! :▷")

# 8. 문서 포맷 정리 함수
def format_docs(docs):
    return "\n".join([
        f"<document><content>{doc.page_content}</content><source>{doc.metadata.get('source','통계가이드.pdf')}</source><page>{doc.metadata.get('page',-1)+1 if 'page' in doc.metadata else '?'}></page></document>"
        for doc in docs
    ])

# 9. 입력 시 처리
if user_input:
    # (1) 사용자 질문 저장
    st.session_state.chat_history.append(("민원인", user_input))

    # (2) 관련 문서 검색
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    search_results = retriever.invoke(user_input)
    context = format_docs(search_results)

    # (3) 프롬프트 생성
    formatted_prompt = prompt.format(question=user_input, context=context)

    # (4) 답변 출력: 말풍선 안에서 스트리밍 표시
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        handler = StreamHandler(message_placeholder)
        llm = ChatOpenAI(
            streaming=True,
            callbacks=[handler],
            openai_api_key=openai_api_key,
            model_name="gpt-4o-mini",  # ✅ 모델 변경 완료
            temperature=0.3,
        )

        llm.invoke([HumanMessage(content=formatted_prompt)])  # ✅ 스트리밍 작동

    # (5) 답변 저장
    st.session_state.chat_history.append(("챗봇", handler.generated_text))

# 10. 대화 내역 말풍선 출력
for role, msg in st.session_state.chat_history:
    with st.chat_message("user" if role == "민원인" else "assistant"):
        st.markdown(msg)
