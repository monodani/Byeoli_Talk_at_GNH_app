# test_faiss.py - 문제 진단용 스크립트
import os
import toml
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.DEBUG)

# ---- 0) API 키 로드 ----
secrets_path = os.path.expanduser("~/.streamlit/secrets.toml")
if os.path.exists(secrets_path):
    secrets = toml.load(secrets_path)
    api_key = secrets.get("OPENAI_API_KEY")
else:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY를 찾을 수 없습니다. secrets.toml 또는 환경변수를 확인하세요.")

# ---- 1) 임베딩 테스트 ----
print("=" * 50)
print("1. 임베딩 모델 테스트")
print("=" * 50)

EMBED_MODEL = "text-embedding-3-small"  # 현재 런타임 모델
embeddings = OpenAIEmbeddings(api_key=api_key, model=EMBED_MODEL)

test_text = "교육만족도"
test_embedding = embeddings.embed_query(test_text)
embed_dim = len(test_embedding)
print(f"✅ 임베딩 차원: {embed_dim}")
print(f"✅ 임베딩 타입: {type(test_embedding)}")
print(f"✅ 샘플 값: {test_embedding[:5]}")

# ---- 2) FAISS 인덱스 로드 ----
print("\n" + "=" * 50)
print("2. FAISS 인덱스 로드 테스트")
print("=" * 50)

try:
    vectorstore_path = Path("vectorstores/vectorstore_unified_satisfaction")
    faiss_file = vectorstore_path / "satisfaction_index.faiss"
    pkl_file = vectorstore_path / "satisfaction_index.pkl"

    print(f"FAISS 파일 존재: {faiss_file.exists()} - {faiss_file}")
    print(f"PKL   파일 존재: {pkl_file.exists()} - {pkl_file}")

    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        index_name="satisfaction_index",
        allow_dangerous_deserialization=True,
    )

    print(f"✅ FAISS 로드 성공!")
    # 더 안전한 문서수 확인
    ntotal = getattr(vectorstore.index, "ntotal", None)
    print(f"✅ 문서 수(FAISS ntotal): {ntotal}")

    # ---- 3) 검색 테스트 ----
    print("\n" + "=" * 50)
    print("3. 검색 테스트")
    print("=" * 50)

    # 인덱스 차원
    index_dim = getattr(vectorstore.index, "d", None)
    print(f"인덱스 차원: {index_dim}")
    print(f"인덱스 타입: {type(vectorstore.index)}")

    query = "교육만족도"
    print(f"\n쿼리: '{query}'")

    try:
        query_embedding = embeddings.embed_query(query)
        print(f"쿼리 임베딩 차원: {len(query_embedding)}")

        results = vectorstore.similarity_search_with_score(query, k=3)
        print(f"✅ 검색 성공! 결과 수: {len(results)}")

        for i, (doc, score) in enumerate(results, 1):
            print(f"\n결과 {i}:")
            print(f"  점수: {score:.4f}")
            preview = (doc.page_content or "")[:200].replace("\n", " ")
            print(f"  내용: {preview}...")

    except Exception as search_error:
        print(f"❌ 검색 실패!")
        print(f"에러 타입: {type(search_error).__name__}")
        print(f"에러 메시지: {str(search_error)}")

        import traceback
        traceback.print_exc()

        if index_dim is not None:
            print(f"\n⚠️ 차원 확인:")
            print(f"  인덱스 차원: {index_dim}")
            print(f"  쿼리 차원: {len(query_embedding) if 'query_embedding' in locals() else 'N/A'}")
            if 'query_embedding' in locals() and index_dim != len(query_embedding):
                print(f"  ❌ 차원 불일치! {index_dim} != {len(query_embedding)}")
                print(f"\n💡 해결 방법:")
                print(f"  1) 인덱스를 현재 임베딩 모델({EMBED_MODEL}, 차원 {embed_dim})로 재생성")
                print(f"  2) 또는 런타임 임베딩 모델을 인덱스 차원({index_dim})에 맞게 변경")

except Exception as e:
    print(f"❌ FAISS 로드 실패!")
    print(f"에러: {e}")
    import traceback
    traceback.print_exc()

# ---- 4) 대체 임베딩 모델 차원 점검 ----
print("\n" + "=" * 50)
print("4. 대체 임베딩 모델 테스트")
print("=" * 50)

models_to_test = [
    "text-embedding-ada-002",   # 1536
    "text-embedding-3-small",   # 1536
    "text-embedding-3-large",   # 3072
]

for model_name in models_to_test:
    try:
        test_embeddings = OpenAIEmbeddings(api_key=api_key, model=model_name)
        test_vec = test_embeddings.embed_query("test")
        print(f"✅ {model_name}: {len(test_vec)} 차원")
    except Exception as e:
        print(f"❌ {model_name}: {e}")
