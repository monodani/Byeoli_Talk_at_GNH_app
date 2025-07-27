import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# 1. API KEY 가져오기
openai_api_key = st.secrets["OPENAI_API_KEY"]

# 2. PDF에서 데이터 추출 및 벡터스토어 생성 (앱 시작 시 1회만)
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

# 4. 대화 내역(딕셔너리 리스트)와 각 메시지의 출력용 st.empty() 리스트
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_placeholders" not in st.session_state:
    st.session_state.chat_placeholders = []

st.title("📊 통계가이드 기반 민원 챗봇")

def format_docs(docs):
    return "\n".join([
        f"<document><content>{doc.page_content}</content><source>{doc.metadata.get('source','통계가이드.pdf')}</source><page>{doc.metadata.get('page',-1)+1 if 'page' in doc.metadata else '?'}></page></document>"
        for doc in docs
    ])

# 5. 답변 스트리밍 핸들러 (챗봇 답변에만 연결)
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.full_text = ""
    def on_llm_new_token(self, token: str, **kwargs):
        self.full_text += token
        self.placeholder.markdown(self.full_text)

# 6. 사용자 입력 처리
user_input = st.chat_input("통계가이드에서 궁금한 점을 질문하세요:")

if user_input:
    # 1) 사용자 메시지(민원인) 추가 및 출력 위치 생성
    st.session_state.chat_history.append({"role": "민원인", "message": user_input})
    st.session_state.chat_placeholders.append(st.empty())
    with st.session_state.chat_placeholders[-1]:
        with st.chat_message("민원인"):
            st.markdown(user_input)
    
    # 2) context 추출 및 프롬프트 생성
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    search_results = retriever.invoke(user_input)
    context = format_docs(search_results)
    formatted_prompt = prompt.format(question=user_input, context=context)
    
    # 3) 챗봇 답변 자리(빈 문자열) 추가 및 출력 위치 생성 (맨 아래에서만 스트리밍)
    st.session_state.chat_history.append({"role": "챗봇", "message": ""})
    st.session_state.chat_placeholders.append(st.empty())
    with st.session_state.chat_placeholders[-1]:
        with st.chat_message("챗봇"):
            placeholder = st.empty()
            handler = StreamHandler(placeholder)
            llm = ChatOpenAI(
                streaming=True,
                callbacks=[handler],
                openai_api_key=openai_api_key,
                model_name="gpt-4o-mini",
                temperature=0.4,
            )
            response = llm.predict(formatted_prompt)
            st.session_state.chat_history[-1]["message"] = response
            placeholder.markdown(response)

# 7. 새로고침/최초 로딩시 기존 대화 내역 순차 출력
for idx, chat in enumerate(st.session_state.chat_history):
    # 만약 메시지 수보다 placeholder가 부족하다면 생성
    if idx >= len(st.session_state.chat_placeholders):
        st.session_state.chat_placeholders.append(st.empty())
    with st.session_state.chat_placeholders[idx]:
        with st.chat_message(chat["role"]):
            st.markdown(chat["message"])

# 8. 최신 메시지로 자동 스크롤
st.markdown(
    """
    <script>
    var elem = document.getElementById('bottom');
    if (elem) elem.scrollIntoView({behavior: "smooth", block: "end"});
    </script>
    <div id="bottom"></div>
    """,
    unsafe_allow_html=True
)
