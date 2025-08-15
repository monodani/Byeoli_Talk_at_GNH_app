# test_faiss.py - 문제 진단용 스크립트
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import numpy as np

logging.basicConfig(level=logging.DEBUG)

# 1. 임베딩 테스트
print("=" * 50)
print("1. 임베딩 모델 테스트")
print("=" * 50)

embeddings = OpenAIEmbeddings(
    api_key="YOUR_API_KEY",
    model="text-embedding-3-small"  # 현재 사용 중인 모델
)

test_text = "교육만족도"
test_embedding = embeddings.embed_query(test_text)
print(f"✅ 임베딩 차원: {len(test_embedding)}")
print(f"✅ 임베딩 타입: {type(test_embedding)}")
print(f"✅ 샘플 값: {test_embedding[:5]}")

# 2. FAISS 인덱스 직접 로드
print("\n" + "=" * 50)
print("2. FAISS 인덱스 로드 테스트")
print("=" * 50)

try:
    vectorstore_path = Path("vectorstores/vectorstore_unified_satisfaction")
    
    # 파일 존재 확인
    faiss_file = vectorstore_path / "satisfaction_index.faiss"
    pkl_file = vectorstore_path / "satisfaction_index.pkl"
    
    print(f"FAISS 파일 존재: {faiss_file.exists()} - {faiss_file}")
    print(f"PKL 파일 존재: {pkl_file.exists()} - {pkl_file}")
    
    # FAISS 로드
    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        index_name="satisfaction_index",
        allow_dangerous_deserialization=True
    )
    
    print(f"✅ FAISS 로드 성공!")
    print(f"✅ 문서 수: {len(vectorstore.docstore._dict)}")
    
    # 3. 검색 테스트
    print("\n" + "=" * 50)
    print("3. 검색 테스트")
    print("=" * 50)
    
    # 인덱스 차원 확인
    if hasattr(vectorstore, 'index'):
        index = vectorstore.index
        print(f"인덱스 차원: {index.d if hasattr(index, 'd') else 'Unknown'}")
        print(f"인덱스 타입: {type(index)}")
    
    # 실제 검색
    query = "교육만족도"
    print(f"\n쿼리: '{query}'")
    
    try:
        # 쿼리 임베딩 생성
        query_embedding = embeddings.embed_query(query)
        print(f"쿼리 임베딩 차원: {len(query_embedding)}")
        
        # similarity_search_with_score 테스트
        results = vectorstore.similarity_search_with_score(query, k=3)
        print(f"✅ 검색 성공! 결과 수: {len(results)}")
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n결과 {i}:")
            print(f"  점수: {score:.4f}")
            print(f"  내용: {doc.page_content[:100]}...")
            
    except Exception as search_error:
        print(f"❌ 검색 실패!")
        print(f"에러 타입: {type(search_error).__name__}")
        print(f"에러 메시지: {str(search_error)}")
        
        # 상세 디버깅
        import traceback
        traceback.print_exc()
        
        # 차원 불일치 확인
        if hasattr(vectorstore, 'index') and hasattr(vectorstore.index, 'd'):
            index_dim = vectorstore.index.d
            query_dim = len(query_embedding)
            print(f"\n⚠️ 차원 확인:")
            print(f"  인덱스 차원: {index_dim}")
            print(f"  쿼리 차원: {query_dim}")
            
            if index_dim != query_dim:
                print(f"  ❌ 차원 불일치! {index_dim} != {query_dim}")
                print(f"\n💡 해결 방법:")
                print(f"  1. 인덱스 재생성 필요")
                print(f"  2. 또는 올바른 임베딩 모델 사용")
                
                # 어떤 모델을 사용해야 하는지 추측
                if index_dim == 1536:
                    print(f"  → 인덱스는 'text-embedding-ada-002'로 생성된 것으로 보임")
                elif index_dim == 3072:
                    print(f"  → 인덱스는 'text-embedding-3-large'로 생성된 것으로 보임")
                elif index_dim == 1536:
                    print(f"  → 인덱스는 'text-embedding-3-small'로 생성된 것으로 보임")

except Exception as e:
    print(f"❌ FAISS 로드 실패!")
    print(f"에러: {e}")
    import traceback
    traceback.print_exc()

# 4. 대체 임베딩 모델 테스트
print("\n" + "=" * 50)
print("4. 대체 임베딩 모델 테스트")
print("=" * 50)

models_to_test = [
    "text-embedding-ada-002",  # 구 모델 (1536 차원)
    "text-embedding-3-small",   # 신 모델 (1536 차원)
    "text-embedding-3-large"    # 신 모델 (3072 차원)
]

for model_name in models_to_test:
    try:
        test_embeddings = OpenAIEmbeddings(
            api_key="YOUR_API_KEY",
            model=model_name
        )
        test_vec = test_embeddings.embed_query("test")
        print(f"✅ {model_name}: {len(test_vec)} 차원")
    except Exception as e:
        print(f"❌ {model_name}: {e}")
