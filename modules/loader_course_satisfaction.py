import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

def load_course_satisfaction_csv() -> pd.DataFrame:
    df = pd.read_csv('course_satisfaction.csv', encoding='cp949')
    return df

def convert_course_df_to_texts(df: pd.DataFrame) -> list[str]:
    texts = []
    for _, row in df.iterrows():
        txt = (
            f"{row['교육주차']}에 개설된 {row['교육과정']}은(는) {row['교육장소']}에서 진행되었으며, "
            f"교육인원은 {row['교육인원']}명이었습니다. "
            f"교육에 대한 전반적인 만족도 점수는 {row['전반만족도']}점, "
            f"교육생이 체감한 교육의 효과성 정도를 가늠할 수 있는 역량향상도와 현업적용도는 각각 {row['역량향상도']}점과 {row['현업적용도']}점이었으며, "
            f"또한, 교과편성에 대한 만족도 {row['교과편성']}점, 강사강의 만족도 {row['강사강의 만족도']}점으로, "
            f"모든 지표에 대한 평균인 종합만족도는 {row['종합만족도']}점(전체 {row['순위']})을 기록했습니다."
        )
        texts.append(txt)
    return texts

def save_course_texts_to_vectorstore(texts: list[str]):
    documents = [Document(page_content=t) for t in texts]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("vectorstore_course_satisfaction")  # ← 디렉토리명 수정

def build_course_vectorstore():
    df = load_course_satisfaction_csv()
    texts = convert_course_df_to_texts(df)
    save_course_texts_to_vectorstore(texts)
    return df

# # 함수 실행 (← 반드시 함수 밖에서!)
# df_course = build_course_vectorstore()