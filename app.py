import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

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

# 4. 스트리밍 핸들러
class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.text_area = st.empty()
        self.full_text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.full_text += token
        self.text_area.markdown(self.full_text)

# 5. 대화 내역 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("📊 통계가이드 기반 민원 챗봇")

# 6. 사용자 입력
user_input = st.chat_input("통계가이드.pdf에서 궁금한 점을 질문하세요:")

def format_docs(docs):
    return "\n".join([
        f"<document><content>{doc.page_content}</content><source>{doc.metadata.get('source','통계가이드.pdf')}</source><page>{doc.metadata.get('page',-1)+1 if 'page' in doc.metadata else '?'}></page></document>"
        for doc in docs
    ])

# 7. 입력시 처리
if user_input:
    st.session_state.chat_history.append(("민원인", user_input))

    # (1) PDF에서 context 추출
    retriever = vectorstore.as_retriever(search_kwargs={"k":5})
    search_results = retriever.invoke(user_input)
    context = format_docs(search_results)
    
    # (2) 프롬프트 적용
    formatted_prompt = prompt.format(question=user_input, context=context)
    
    # (3) 답변 생성 (스트리밍)
    handler = StreamHandler()
    llm = ChatOpenAI(
        streaming=True,
        callbacks=[handler],
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",
        temperature=0.4,
    )
    response = llm.predict(formatted_prompt)
    st.session_state.chat_history.append(("챗봇", response))

# 8. 대화 내역 출력 (최신 대화가 아래에 보이도록, 스크롤 자동 하단)
# 1) 전체 대화 내역을 for문으로 위에서 아래로 출력
for idx, (role, msg) in enumerate(st.session_state.chat_history):
    # st.chat_message는 Streamlit 1.25 이상에서 지원
    with st.chat_message(role):
        st.markdown(msg)
        # 답변이 최신(마지막)인 경우, 여기에 자동 스크롤 anchor 삽입
        if idx == len(st.session_state.chat_history) - 1:
            st.markdown('<div id="bottom"></div>', unsafe_allow_html=True)

# 2) 답변 생성/갱신 시 자동으로 아래로 스크롤
st.markdown(
    """
    <script>
    var elem = document.getElementById('bottom');
    if (elem) elem.scrollIntoView({behavior: "smooth", block: "end"});
    </script>
    """,
    unsafe_allow_html=True
)
