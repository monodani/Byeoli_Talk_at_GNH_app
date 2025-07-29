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

# 2. 벡터스토어 로드 함수
@st.cache_resource(show_spinner=True)
def load_vectorstore(pdf_path, name):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    for p in pages:
        p.metadata["doc_name"] = name  # 문서 출처 구분
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(pages, embeddings)
    return vectorstore

# 3. 두 PDF 문서 로드
vector_2024 = load_vectorstore("2024년도 교육훈련종합평가서.pdf", "2024년도 교육훈련 종합평가서")
vector_2025 = load_vectorstore("2025년 교육훈련계획서.pdf", "2025년 교육훈련계획서")

# 4. 검색 통합 함수
def combined_search(question):
    retrievers = [
        vector_2024.as_retriever(search_kwargs={"k": 5}),
        vector_2025.as_retriever(search_kwargs={"k": 5})
    ]
    all_results = []
    for retriever in retrievers:
        all_results.extend(retriever.invoke(question))
    return all_results

# 5. 문서 형식 정리 함수
def format_docs(docs):
    def clean_table(text):
        # 연속된 공백 → 탭으로 변환 (GPT가 표처럼 추정할 수 있도록)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            if " " in line:
                # 공백이 많은 경우 탭으로 치환 시도
                parts = [part for part in line.split(" ") if part.strip()]
                cleaned_line = "\t".join(parts)
                cleaned_lines.append(cleaned_line)
            else:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    return "\n\n".join([
        f"{clean_table(doc.page_content)}\n\n[출처: {doc.metadata.get('doc_name')}, p.{doc.metadata.get('page', -1) + 1}]"
        for doc in docs
    ])



# 6. 프롬프트 템플릿 정의
prompt = PromptTemplate.from_template("""
SYSTEM: 당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
        문맥에 표 형식 데이터가 포함된 경우, 열(Column) 이름과 숫자 값을 잘 연결하여 의미를 해석해 주세요.
        검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 '주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다' 라고 답하세요.
        기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요. 출처(page, source)를 답변에 포함하세요. 답변은 한글로 답변해 주세요.

HUMAN:
#Question: {question}

#Context: {context}

#Answer:
""")

# 7. 스트리밍 핸들러
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.generated_text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.generated_text += token
        self.placeholder.markdown(self.generated_text)

# 8. 대화 기록 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 9. 상단 UI: 로고 + 타이틀
col1, col2 = st.columns([1, 6])
with col1:
    st.image("byeory.png", width=60)
with col2:
    st.markdown("### 벼리톡@경남인재개발원")

# 10. 사용자 입력
user_input = st.chat_input("경남인재개발원 교육과정에 대해 벼리에게 물어보세요! : ▷")

# 11. 사용자 입력 처리
if user_input:
    
    # 사용자 질문 기록
    st.session_state.chat_history.append(("민원인", user_input))
    # 빈 챗봇 응답 자리 추가
    st.session_state.chat_history.append(("벼리", ""))  # 미리 자리 만들기

# 12. 대화 출력 (챗봇 스트리밍도 여기서 함께 처리)
for i, (role, msg) in enumerate(st.session_state.chat_history):
    if role == "민원인":
        with st.chat_message("user"):
            st.markdown(msg)

    elif role == "벼리" and msg == "":
        with st.chat_message("assistant", avatar="byeory.png"):
            message_placeholder = st.empty()
            handler = StreamHandler(message_placeholder)

            # 사용자 입력은 직전 항목 기준
            last_user_input = None
            for j in range(i - 1, -1, -1):
                if st.session_state.chat_history[j][0] == "민원인":
                    last_user_input = st.session_state.chat_history[j][1]
                    break

            if last_user_input:
                # 검색 + 답변 생성
                search_results = combined_search(last_user_input)
                context = format_docs(search_results)
                formatted_prompt = prompt.format(question=last_user_input, context=context)

                llm = ChatOpenAI(
                    model_name="gpt-4o",
                    streaming=True,
                    callbacks=[handler],
                    openai_api_key=openai_api_key,
                    temperature=0.3,
                )
                llm.invoke([HumanMessage(content=formatted_prompt)])

                # 스트리밍된 답변 저장
                st.session_state.chat_history[i] = ("벼리", handler.generated_text)

    else:
        with st.chat_message("assistant", avatar="byeory2.png"):
            st.markdown(msg)
