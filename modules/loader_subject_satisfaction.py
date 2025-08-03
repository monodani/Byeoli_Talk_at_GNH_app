import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

def load_subject_satisfaction_csv() -> pd.DataFrame:
    df = pd.read_csv('subject_satisfaction.csv', encoding='cp949')
    return df

def convert_subject_df_to_texts(df: pd.DataFrame) -> list[str]:
    texts = []
    for _, row in df.iterrows():
        txt = (
            f"{row['교육주차']}에 개설된 {row['교육과정']}에서 {row['교과목']} 교과목의 "
            f"강사강의 만족도는 {row['강사강의 만족도']}점으로 전체 {row['순위']}를 기록했습니다."
        )
        texts.append(txt)
    return texts

def save_subject_texts_to_vectorstore(texts: list[str]):
    documents = [Document(page_content=t) for t in texts]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("vectorstore_satisfaction")  # ← 디렉토리명 수정

def build_subject_vectorstore():
    df = load_subject_satisfaction_csv()
    texts = convert_subject_df_to_texts(df)
    save_subject_texts_to_vectorstore(texts)
    return df

# # 함수 실행 (← 반드시 함수 밖에서!)
# df_subject = build_subject_vectorstore()