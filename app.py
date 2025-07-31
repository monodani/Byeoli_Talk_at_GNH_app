import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import CSVLoader
from operator import attrgetter

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

def load_vectorstore_csv(csv_path, name):
    loader = CSVLoader(file_path=csv_path, encoding="utf-8", csv_args={'delimiter': ','})
    docs = loader.load()
    for d in docs:
        d.metadata["doc_name"] = name
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


# 3. 두 PDF 문서 로드
vector_2024_교육평가 = load_vectorstore("2024년도 교육훈련종합평가서.pdf", "2024년도 교육훈련 종합평가서")
vector_2025_교육계획 = load_vectorstore("2025년 교육훈련계획서.pdf", "2025년 교육훈련계획서")
vector_2025_교과목 = load_vectorstore_csv("2025 교과목 만족도.csv", "2025 교과목 만족도")
vector_2025_교육과정 = load_vectorstore_csv("2025 교육과정 종합만족도.csv", "2025 교육과정 종합만족도")

# 4. 검색 통합 함수
def combined_search(question):
    retrievers = [
        vector_2024_교육평가.as_retriever(search_kwargs={"k": 5}),
        vector_2025_교육계획.as_retriever(search_kwargs={"k": 5}),
        vector_2025_교과목.as_retriever(search_kwargs={"k": 5}),
        vector_2025_교육과정.as_retriever(search_kwargs={"k": 5}),
    ]
    all_results = []
    for retriever in retrievers:
        all_results.extend(retriever.invoke(question))
    # 유사도 점수가 있다면 기준으로 정렬
    sorted_results = sorted(all_results, key=lambda x: x.metadata.get("score", 0), reverse=True)
    # 상위 10개만 사용
    return sorted_results[:10]

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
    
# 🔁 최근 대화 기록을 문자열로 변환하는 함수
def convert_chat_history_to_str(history, max_turns=3):
    """
    최근 대화 이력 중 마지막 max_turns 쌍(민원인-벼리)을 추출해 문자열로 변환합니다.
    """
    turns = []
    count = 0
    for i in range(len(history) - 2, -1, -2):  # 벼리 발화 기준으로 묶기
        if history[i][0] == "민원인" and history[i+1][0] == "벼리":
            turns.append(f"민원인: {history[i][1]}\n벼리: {history[i+1][1]}")
            count += 1
        if count >= max_turns:
            break
    return "\n\n".join(reversed(turns))

# 6. 프롬프트 템플릿 정의
prompt = PromptTemplate.from_template("""
SYSTEM: 당신의 이름은 "벼리"입니다. 당신은 경남인재개발원 관련 문서를 바탕으로 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다.
        당신의 임무는 주어진 문맥(context)과 대화 이력(chat history)을 참고하여 사용자의 질문(question)에 정확하고 맥락에 맞는 답하는 것입니다.
        검색된 다음 문맥(context)을 사용하여 질문(question)에 답하세요. 만약, 주어진 문맥(context)에서 답을 찾을 수 없거나 답을 모른다면, 사용자가 당신에게 주어진 정보에 접근할 수 있도록 힌트를 주거나 사용자 질문을 되물어 사용자 의도를 정확히 파악하세요.
        그럼에도, 해당 질문이 당신에게 주어진 정보 기반으로는 답을 할 수 없거나 모른다면, "벼리가 답하기 어려운 내용이에요...ㅠ_ㅠ" 또는 "벼리가 잘 모르는 내용이에요...ㅜ"라고 답하면서 사용자 질문과 유사성이 높은 당신이 지닌 정보가 무엇인지 알려주고 이를 원하는지 물으세요.
        문맥에 표 형식의 데이터가 포함된 경우, 문맥(context)을 표의 열(column)과 행(row)의 이름에 잘 연결하여,
        해당되는 그 열(column)과 행(row)이 교차하는 셀의 데이터 값을 불러와 그 의미를 해석해 주세요.
        기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요. 출처(page, source)를 답변에 포함하세요. 답변은 한글로 답변해 주세요.

HUMAN:
#Question: {question}

#Context:
다음은 교육과정별 및 교과목별 관련 정보들이 표 형식으로 정리된 문서입니다.
각 과정의 열(column)과 행(row)의 이름과 데이터 값을 연결해 분석한 후,
질문에 가장 부합하는 정보를 찾아주세요.

{context}

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
col1, col2 = st.columns([2, 8])
with col1:
    st.image("byeory.png", width=100)
with col2:
    st.markdown('<h1 style="margin-top: 10px;">벼리톡@경남인재개발원</h1>', unsafe_allow_html=True) 

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
                formatted_prompt = prompt.format(
                      question=last_user_input,
                      context=format_docs(search_results),
                      chat_history=convert_chat_history_to_str(st.session_state.chat_history)
)

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
