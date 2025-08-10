#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - 벡터스토어 구축 메인 스크립트 (수정됨)

🔧 주요 수정사항:
✅ IndexManager와 완전 동일한 파일명 매핑 적용
✅ 벡터스토어 저장 로직 개선
✅ 파일 존재 확인 및 검증 강화
✅ BM25 인덱스 생성 및 저장 로직 추가
"""

import os
import sys
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import gc

# ✅ BM25 인덱스 생성을 위해 필요한 라이브러리 추가
import pickle
from rank_bm25 import BM25Okapi

# 프로젝트 루트 경로 설정
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

# 프로젝트 모듈 임포트
try:
    from utils.config import config
    from utils.textifier import TextChunk
    from utils.index_manager import get_index_manager, IndexManager
    from modules.base_loader import BaseLoader
    
    # 도메인별 로더들
    from modules.loader_satisfaction import SatisfactionLoader
    from modules.loader_general import GeneralLoader
    from modules.loader_publish import PublishLoader
    from modules.loader_cyber import CyberLoader
    from modules.loader_menu import MenuLoader
    from modules.loader_notice import NoticeLoader
    
except ImportError as e:
    print(f"❌ 필수 모듈 import 실패: {e}")
    print("requirements.txt의 모든 의존성이 설치되었는지 확인해주세요.")
    sys.exit(1)

# 외부 라이브러리 (선택적 의존성)
try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️ FAISS 라이브러리를 찾을 수 없습니다. 일부 기능이 제한됩니다.")
    FAISS_AVAILABLE = False

# 로깅 설정
Path(ROOT_DIR / "logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT_DIR / "logs" / "data_ingestion.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ================================================================
# 1. 수정된 벡터스토어 구축 클래스
# ================================================================

class VectorStoreBuilder:
    """
    벡터스토어 구축 메인 클래스 (IndexManager 호환성 수정)
    """
    
    def __init__(self):
        """VectorStoreBuilder 초기화"""
        self.start_time = time.time()
        self.results = {}
        self.embeddings = None
        self.total_chunks = 0
        self.total_documents = 0
        
        # 도메인별 로더 등록
        self.loaders = {
            "satisfaction": SatisfactionLoader,
            "general": GeneralLoader, 
            "publish": PublishLoader,
            "cyber": CyberLoader,
            "menu": MenuLoader,
            "notice": NoticeLoader
        }
        
        logger.info("🚀 VectorStoreBuilder 초기화 완료 (IndexManager 호환성 강화)")
        logger.info(f"📁 프로젝트 루트: {ROOT_DIR}")
        logger.info(f"📊 처리 대상 도메인: {list(self.loaders.keys())}")
    
    def setup_environment(self) -> bool:
        """환경 설정 및 의존성 검사"""
        try:
            logger.info("🔧 환경 설정 시작...")
            
            # 1. 필수 디렉터리 생성
            required_dirs = [
                config.VECTORSTORE_DIR,
                config.CACHE_DIR,
                config.LOGS_DIR,
                config.DATA_DIR
            ]
            
            for dir_path in required_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.debug(f"📁 디렉터리 확인: {dir_path}")
            
            # 2. FAISS 의존성 확인
            if not FAISS_AVAILABLE:
                logger.error("❌ FAISS 라이브러리가 필요합니다.")
                return False
            
            # 3. 임베딩 모델 초기화
            try:
                self.embeddings = OpenAIEmbeddings(
                    model=config.EMBEDDING_MODEL,
                    dimensions=config.EMBEDDING_DIMENSION
                )
                logger.info(f"✅ 임베딩 모델 초기화: {config.EMBEDDING_MODEL}")
            except Exception as e:
                logger.error(f"❌ 임베딩 모델 초기화 실패: {e}")
                return False
            
            # 4. 데이터 디렉터리 확인
            if not config.DATA_DIR.exists():
                logger.error(f"❌ 데이터 디렉터리를 찾을 수 없습니다: {config.DATA_DIR}")
                return False
            
            logger.info("✅ 환경 설정 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 환경 설정 실패: {e}")
            return False
    
    def build_all_vectorstores(self) -> Dict[str, bool]:
        """모든 도메인의 벡터스토어 구축"""
        logger.info("🔨 전체 벡터스토어 구축 시작...")
        
        overall_start = time.time()
        
        for domain, loader_class in self.loaders.items():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔨 {domain.upper()} 도메인 처리 시작...")
                logger.info(f"{'='*60}")
                
                success = self.build_domain_vectorstore(domain, loader_class)
                self.results[domain] = success
                
                if success:
                    logger.info(f"✅ {domain} 도메인 처리 완료")
                else:
                    logger.warning(f"⚠️ {domain} 도메인 처리 실패 (계속 진행)")
                
                # 메모리 정리
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ {domain} 도메인 처리 중 예외: {e}")
                logger.debug(traceback.format_exc())
                self.results[domain] = False
        
        # 전체 결과 요약
        total_time = time.time() - overall_start
        success_count = sum(1 for success in self.results.values() if success)
        total_count = len(self.results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🏁 전체 벡터스토어 구축 완료!")
        logger.info(f"📊 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        logger.info(f"⏱️ 총 소요시간: {total_time:.2f}초")
        logger.info(f"📄 총 문서 수: {self.total_documents}")
        logger.info(f"🔪 총 청크 수: {self.total_chunks}")
        logger.info(f"{'='*60}")
        
        return self.results
    
    def build_domain_vectorstore(self, domain: str, loader_class) -> bool:
        """개별 도메인 벡터스토어 구축"""
        domain_start = time.time()
        
        try:
            # 1. 로더 인스턴스 생성
            logger.info(f"📚 {domain} 로더 초기화...")
            loader = loader_class()
            
            # 2. 도메인 데이터 처리
            logger.info(f"🔍 {domain} 데이터 처리 중...")
            chunks = loader.process_domain_data()
            
            if not chunks:
                logger.warning(f"⚠️ {domain} 도메인에서 처리할 데이터가 없습니다")
                return False
            
            logger.info(f"📄 {domain}: {len(chunks)}개 청크 생성됨")
            self.total_chunks += len(chunks)
            
            # 3. ✅ IndexManager와 동일한 경로 및 파일명 사용
            vectorstore_path = self._get_vectorstore_path(domain)
            # domain 인자를 _create_vectorstores 메서드에 전달
            success = self._create_vectorstores(chunks, vectorstore_path, domain)
            
            if success:
                domain_time = time.time() - domain_start
                logger.info(f"✅ {domain} 벡터스토어 생성 완료 ({domain_time:.2f}s)")
                return True
            else:
                logger.error(f"❌ {domain} 벡터스토어 생성 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ {domain} 벡터스토어 구축 실패: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def _get_vectorstore_path(self, domain: str) -> Path:
        """✅ IndexManager와 완전히 동일한 경로 반환"""
        # IndexManager._get_domain_configs()와 완전 동일
        vectorstore_base = config.ROOT_DIR / "vectorstores"
        
        path_mapping = {
            "satisfaction": vectorstore_base / "vectorstore_unified_satisfaction",
            "general": vectorstore_base / "vectorstore_general",
            "menu": vectorstore_base / "vectorstore_menu", 
            "cyber": vectorstore_base / "vectorstore_cyber",
            "publish": vectorstore_base / "vectorstore_unified_publish",
            "notice": vectorstore_base / "vectorstore_notice"
        }
        
        return path_mapping.get(domain, vectorstore_base / f"vectorstore_{domain}")
    
    def _create_vectorstores(self, chunks: List[TextChunk], output_path: Path, domain: str) -> bool:
        """✅ BM25 인덱스와 FAISS 벡터스토어 모두 생성"""
        try:
            logger.info(f"🔧 FAISS 및 BM25 인덱스 생성 중: {output_path}")
            
            # 1. 텍스트 추출
            texts = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            logger.info(f"📝 텍스트 데이터: {len(texts)}개")
            
            # ✅ 2. BM25 인덱스 생성 및 저장 (추가된 부분)
            logger.info(f"🔧 BM25 인덱스 생성 중...")
            tokenized_texts = [text.split(" ") for text in texts]
            bm25_index = BM25Okapi(tokenized_texts)
            
            bm25_path = output_path / f"{domain}_index.bm25"
            with open(bm25_path, "wb") as f:
                pickle.dump((bm25_index, metadatas), f) # 메타데이터도 함께 저장
            logger.info(f"✅ BM25 인덱스 저장 완료: {bm25_path}")

            # 3. FAISS 벡터스토어 생성 및 저장 (기존 로직)
            batch_size = 100
            vectorstore = None
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                
                logger.info(f"⚡ 배치 처리: {i+1}-{min(i+batch_size, len(texts))}/{len(texts)}")
                
                if vectorstore is None:
                    vectorstore = FAISS.from_texts(
                        texts=batch_texts,
                        embedding=self.embeddings,
                        metadatas=batch_metadatas
                    )
                else:
                    batch_vectorstore = FAISS.from_texts(
                        texts=batch_texts,
                        embedding=self.embeddings,
                        metadatas=batch_metadatas
                    )
                    vectorstore.merge_from(batch_vectorstore)
                    
                gc.collect()
            
            output_path.mkdir(parents=True, exist_ok=True)
            index_name = f"{domain}_index"
            logger.info(f"💾 FAISS 벡터스토어 저장: {output_path}/{index_name}")
            
            vectorstore.save_local(
                folder_path=str(output_path),
                index_name=index_name
            )
            
            # 4. 파일 생성 확인
            faiss_file = output_path / f"{index_name}.faiss"
            pkl_file = output_path / f"{index_name}.pkl"
            
            logger.info(f"🔍 파일 생성 확인:")
            logger.info(f"  - FAISS: {faiss_file} ({'✅ 존재' if faiss_file.exists() else '❌ 없음'})")
            logger.info(f"  - PKL: {pkl_file} ({'✅ 존재' if pkl_file.exists() else '❌ 없음'})")
            
            # ✅ BM25 파일도 함께 확인
            bm25_file = output_path / f"{domain}_index.bm25"
            logger.info(f"  - BM25: {bm25_file} ({'✅ 존재' if bm25_file.exists() else '❌ 없음'})")
            
            if faiss_file.exists() and pkl_file.exists() and bm25_file.exists():
                faiss_size = faiss_file.stat().st_size / (1024*1024)  # MB
                pkl_size = pkl_file.stat().st_size / (1024*1024)     # MB
                bm25_size = bm25_file.stat().st_size / (1024*1024)   # MB
                logger.info(f"✅ 벡터스토어 저장 완료: FAISS {faiss_size:.1f}MB, PKL {pkl_size:.1f}MB, BM25 {bm25_size:.1f}MB")
                self.total_documents += len(texts)
                return True
            else:
                logger.error(f"❌ 벡터스토어 파일 생성 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ 벡터스토어 생성 실패: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def verify_vectorstores(self) -> Dict[str, Any]:
        """✅ 생성된 벡터스토어 IndexManager 호환성 검증 (개선)"""
        logger.info("🔍 벡터스토어 검증 시작...")
        
        verification_results = {}
        
        try:
            # IndexManager 인스턴스 생성
            index_manager = get_index_manager()
            
            # 각 도메인별 검증
            for domain in self.loaders.keys():
                try:
                    logger.info(f"🔍 {domain} 벡터스토어 검증 중...")
                    
                    # 1. 파일 존재 확인
                    vectorstore_path = self._get_vectorstore_path(domain)
                    faiss_file = vectorstore_path / f"{domain}_index.faiss"
                    pkl_file = vectorstore_path / f"{domain}_index.pkl"
                    # ✅ BM25 파일 경로 추가
                    bm25_file = vectorstore_path / f"{domain}_index.bm25"

                    files_exist = faiss_file.exists() and pkl_file.exists() and bm25_file.exists()
                    
                    logger.info(f"   파일 존재: FAISS {'✅' if faiss_file.exists() else '❌'}, PKL {'✅' if pkl_file.exists() else '❌'}, BM25 {'✅' if bm25_file.exists() else '❌'}")
                    
                    if not files_exist:
                        verification_results[domain] = {
                            "loaded": False,
                            "search_test": False,
                            "status": f"❌ 파일 없음: {vectorstore_path}",
                            "files_exist": False
                        }
                        continue
                    
                    # 2. IndexManager를 통한 로드 테스트
                    vectorstore = index_manager.get_vectorstore(domain)
                    
                    if vectorstore is not None:
                        # 3. 간단한 검색 테스트
                        test_results = vectorstore.similarity_search("테스트", k=1)
                        verification_results[domain] = {
                            "loaded": True,
                            "search_test": len(test_results) > 0,
                            "status": "✅ 성공",
                            "files_exist": True,
                            "documents_count": len(test_results)
                        }
                        logger.info(f"✅ {domain} 검증 성공")
                    else:
                        verification_results[domain] = {
                            "loaded": False,
                            "search_test": False,
                            "status": "❌ IndexManager 로드 실패",
                            "files_exist": True
                        }
                        logger.warning(f"⚠️ {domain} 검증 실패: IndexManager 로드 불가")
                        
                except Exception as e:
                    verification_results[domain] = {
                        "loaded": False,
                        "search_test": False,
                        "status": f"❌ 오류: {str(e)[:50]}",
                        "files_exist": False
                    }
                    logger.error(f"❌ {domain} 검증 중 오류: {e}")
            
            # 전체 헬스체크
            health = index_manager.health_check()
            verification_results["overall_health"] = health
            
            # 검증 결과 요약
            successful_domains = sum(1 for result in verification_results.values() 
                                   if isinstance(result, dict) and result.get("loaded", False))
            total_domains = len(self.loaders)
            
            logger.info(f"🏁 검증 완료: {successful_domains}/{total_domains} 도메인 성공")
            
            return verification_results
            
        except Exception as e:
            logger.error(f"❌ 벡터스토어 검증 실패: {e}")
            return {"error": str(e)}
    
    def generate_report(self, verification_results: Dict[str, Any]) -> str:
        """구축 결과 리포트 생성"""
        total_time = time.time() - self.start_time
        
        report = f"""
{'='*80}
🌟 BYEOLI_TALK_AT_GNH_app 벡터스토어 구축 완료 리포트 (수정됨)
{'='*80}

📊 구축 결과:
"""
        
        for domain, success in self.results.items():
            status = "✅ 성공" if success else "❌ 실패"
            report += f"  - {domain.ljust(12)}: {status}\n"
        
        successful_builds = sum(1 for success in self.results.values() if success)
        total_builds = len(self.results)
        
        report += f"""
📈 성공률: {successful_builds}/{total_builds} ({successful_builds/total_builds*100:.1f}%)
⏱️ 총 소요시간: {total_time:.2f}초
📄 총 문서 수: {self.total_documents:,}개
🔪 총 청크 수: {self.total_chunks:,}개

🔍 IndexManager 호환성 검증:
"""
        
        for domain, result in verification_results.items():
            if domain != "overall_health" and isinstance(result, dict):
                report += f"  - {domain.ljust(12)}: {result.get('status', '❓ 알 수 없음')}\n"
        
        if "overall_health" in verification_results:
            health = verification_results["overall_health"]
            report += f"""
🏥 전체 시스템 상태: {health.get('overall_health', '알 수 없음')}
📚 로드된 도메인: {health.get('loaded_domains', '0/0')}
📊 총 인덱스 문서: {health.get('total_documents', 0):,}개

"""
        
        # 성능 평가
        if total_time <= 300:  # 5분
            performance = "🚀 우수"
        elif total_time <= 600:  # 10분
            performance = "👍 양호"
        else:
            performance = "⚠️ 개선 필요"
        
        report += f"""📋 성능 평가:
  - 구축 시간: {performance} ({total_time:.1f}초)
  - 메모리 사용: {'🟢 효율적' if self.total_chunks < 50000 else '🟡 보통'}
  - 성공률: {'🟢 우수' if successful_builds/total_builds >= 0.8 else '🟡 보통'}

🎯 다음 단계:
  1. app.py 실행으로 Streamlit UI 테스트
  2. 각 도메인별 질의응답 기능 확인
  3. 성능 모니터링 및 최적화

💡 참고사항:
  - 벡터스토어는 {config.VECTORSTORE_DIR} 에 저장됨
  - IndexManager 자동 로드 및 핫스왑 지원
  - 로그는 {ROOT_DIR}/logs/data_ingestion.log 에서 확인 가능

{'='*80}
🌟 BYEOLI 챗봇 벡터스토어 구축 완료! (IndexManager 호환성 확인됨)
{'='*80}
"""
        
        return report

# ================================================================
# 2. 메인 실행 함수 (변경 없음)
# ================================================================

def main():
    """메인 실행 함수"""
    print("🌟 BYEOLI_TALK_AT_GNH_app 벡터스토어 구축 시작! (IndexManager 호환성 수정됨)")
    print(f"📍 프로젝트 위치: {ROOT_DIR}")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 1. VectorStoreBuilder 초기화
    builder = VectorStoreBuilder()
    
    # 2. 환경 설정
    if not builder.setup_environment():
        logger.error("❌ 환경 설정 실패. 프로그램을 종료합니다.")
        return False
    
    # 3. 벡터스토어 구축
    build_results = builder.build_all_vectorstores()
    
    # 4. 검증
    verification_results = builder.verify_vectorstores()
    
    # 5. 리포트 생성 및 출력
    report = builder.generate_report(verification_results)
    print(report)
    
    # 6. 리포트 파일 저장
    try:
        report_path = ROOT_DIR / "logs" / f"build_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"📝 리포트 저장됨: {report_path}")
    except Exception as e:
        logger.warning(f"⚠️ 리포트 저장 실패: {e}")
    
    # 7. 성공 여부 반환
    success_rate = sum(1 for success in build_results.values() if success) / len(build_results)
    return success_rate >= 0.8

# ================================================================
# 3. 개발/테스트 함수들 (변경 없음)
# ================================================================

def test_single_domain(domain: str):
    """개별 도메인 테스트"""
    print(f"🧪 {domain} 도메인 단독 테스트 시작...")
    
    builder = VectorStoreBuilder()
    
    if not builder.setup_environment():
        print(f"❌ {domain} 환경 설정 실패")
        return False
    
    if domain not in builder.loaders:
        print(f"❌ 알 수 없는 도메인: {domain}")
        return False
    
    loader_class = builder.loaders[domain]
    success = builder.build_domain_vectorstore(domain, loader_class)
    
    if success:
        print(f"✅ {domain} 도메인 테스트 성공!")
    else:
        print(f"❌ {domain} 도메인 테스트 실패")
    
    return success

def quick_health_check():
    """빠른 상태 확인"""
    print("🏥 IndexManager 헬스체크...")
    
    try:
        from utils.index_manager import index_health_check
        health = index_health_check()
        
        print(f"시스템 상태: {health.get('overall_health', '알 수 없음')}")
        print(f"로드된 도메인: {health.get('loaded_domains', '0/0')}")
        print(f"총 문서 수: {health.get('total_documents', 0):,}개")
        
        return health.get('overall_health') == 'healthy'
        
    except Exception as e:
        print(f"❌ 헬스체크 실패: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BYEOLI 챗봇 벡터스토어 구축 (수정됨)")
    parser.add_argument("--domain", type=str, help="특정 도메인만 처리")
    parser.add_argument("--health-check", action="store_true", help="헬스체크만 실행")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.health_check:
            # 헬스체크만 실행
            success = quick_health_check()
            sys.exit(0 if success else 1)
        elif args.domain:
            # 특정 도메인만 처리
            success = test_single_domain(args.domain)
            sys.exit(0 if success else 1)
        else:
            # 전체 벡터스토어 구축
            success = main()
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
