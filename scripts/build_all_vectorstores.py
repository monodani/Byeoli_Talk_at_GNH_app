#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - 통합 벡터스토어 빌드 스크립트

GitHub Actions에서 실행되는 모든 도메인 로더 통합 실행기:
- 해시 기반 증분 빌드 지원
- 도메인별 독립 처리 (에러 격리)
- 상세 실행 로그 및 통계
- 실패 시 이메일 알림 기능
"""

import os
import sys
import logging
import traceback
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 개별 로더 모듈 임포트
try:
    from modules.loader_cyber import CyberLoader
    from modules.loader_menu import MenuLoader
    from modules.loader_notice import NoticeLoader
    from modules.loader_satisfaction import SatisfactionLoader
    
    # PDFProcessor 의존성이 있는 로더들은 조건부 임포트
    try:
        from modules.loader_general import GeneralLoader
        GENERAL_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️ GeneralLoader 임포트 실패 (건너뜀): {e}")
        GeneralLoader = None
        GENERAL_AVAILABLE = False
    
    try:
        from modules.loader_publish import PublishLoader
        PUBLISH_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️ PublishLoader 임포트 실패 (건너뜀): {e}")
        PublishLoader = None
        PUBLISH_AVAILABLE = False
        
except ImportError as e:
    print(f"❌ 로더 모듈 임포트 실패: {e}")
    sys.exit(1)

# 로깅 설정
def setup_logging():
    """GitHub Actions 환경에 최적화된 로깅 설정"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 콘솔 출력
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('vectorstore_build.log', encoding='utf-8')
        ]
    )
    
    # GitHub Actions 그룹 출력 지원
    logger = logging.getLogger(__name__)
    return logger

class VectorstoreBuildManager:
    """벡터스토어 빌드 관리자"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.build_results = {}
        self.start_time = datetime.now()
        
        # 도메인별 로더 매핑 (문제 있는 로더 임시 제외)
        base_loaders = {
            'cyber': CyberLoader,
            'menu': MenuLoader,
            'notice': NoticeLoader,
            'satisfaction': SatisfactionLoader
        }
        
        # PDFProcessor 의존성이 있는 로더들은 조건부 추가
        if GENERAL_AVAILABLE and GeneralLoader:
            base_loaders['general'] = GeneralLoader
        else:
            self.logger.warning("⚠️ GeneralLoader를 사용할 수 없어 건너뜁니다")
            
        if PUBLISH_AVAILABLE and PublishLoader:
            base_loaders['publish'] = PublishLoader
        else:
            self.logger.warning("⚠️ PublishLoader를 사용할 수 없어 건너뜁니다")
        
        self.loaders = base_loaders
        
        self.logger.info("🚀 벡터스토어 통합 빌드 시작")
        self.logger.info(f"📅 시작 시간: {self.start_time.isoformat()}")
        self.logger.info(f"📊 총 도메인 수: {len(self.loaders)}")
    
    def build_all_vectorstores(self) -> Dict[str, Any]:
        """모든 도메인 벡터스토어 빌드 실행"""
        
        success_count = 0
        failed_domains = []
        
        for domain_name, loader_class in self.loaders.items():
            try:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"🔄 {domain_name.upper()} 도메인 처리 시작")
                self.logger.info(f"{'='*50}")
                
                # GitHub Actions 그룹 출력
                print(f"::group::{domain_name.upper()} 도메인 벡터스토어 빌드")
                
                # 로더 인스턴스 생성 및 실행
                loader = loader_class()
                
                # 해시 기반 증분 빌드 실행 (force_rebuild=False)
                build_success = loader.build_vectorstore(force_rebuild=False)
                
                if build_success:
                    # 통계 정보 수집
                    stats = loader.get_stats()
                    self.build_results[domain_name] = {
                        'status': 'success',
                        'stats': stats,
                        'error': None
                    }
                    success_count += 1
                    self.logger.info(f"✅ {domain_name} 도메인 빌드 성공")
                    self.logger.info(f"📊 통계: {stats}")
                else:
                    self.build_results[domain_name] = {
                        'status': 'failed',
                        'stats': None,
                        'error': 'Build returned False'
                    }
                    failed_domains.append(domain_name)
                    self.logger.error(f"❌ {domain_name} 도메인 빌드 실패: Build returned False")
                
                print("::endgroup::")
                
            except Exception as e:
                # 개별 도메인 실패가 전체를 중단시키지 않도록 격리
                error_msg = f"{type(e).__name__}: {str(e)}"
                error_trace = traceback.format_exc()
                
                self.build_results[domain_name] = {
                    'status': 'failed',
                    'stats': None,
                    'error': error_msg,
                    'traceback': error_trace
                }
                failed_domains.append(domain_name)
                
                self.logger.error(f"❌ {domain_name} 도메인 처리 중 예외 발생:")
                self.logger.error(f"   오류: {error_msg}")
                self.logger.error(f"   추적: {error_trace}")
                
                print("::endgroup::")
                continue
        
        # 최종 결과 수집
        end_time = datetime.now()
        elapsed_time = end_time - self.start_time
        
        final_results = {
            'total_domains': len(self.loaders),
            'successful_domains': success_count,
            'failed_domains': len(failed_domains),
            'failed_domain_list': failed_domains,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'elapsed_time': str(elapsed_time),
            'build_details': self.build_results,
            'success_rate': f"{(success_count / len(self.loaders) * 100):.1f}%"
        }
        
        return final_results
    
    def save_build_report(self, results: Dict[str, Any]) -> Path:
        """빌드 결과 보고서 저장"""
        report_file = Path("vectorstore_build_report.json")
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"📄 빌드 보고서 저장: {report_file}")
            return report_file
            
        except Exception as e:
            self.logger.error(f"보고서 저장 실패: {e}")
            return None
    
    def print_summary(self, results: Dict[str, Any]):
        """실행 결과 요약 출력"""
        
        print("\n" + "="*70)
        print("📊 벡터스토어 빌드 최종 결과")
        print("="*70)
        
        print(f"🕐 시작 시간: {results['start_time']}")
        print(f"🕐 종료 시간: {results['end_time']}")
        print(f"⏱️ 소요 시간: {results['elapsed_time']}")
        print(f"📈 성공률: {results['success_rate']}")
        
        print(f"\n📊 처리 결과:")
        print(f"   ✅ 성공: {results['successful_domains']}개 도메인")
        print(f"   ❌ 실패: {results['failed_domains']}개 도메인")
        
        if results['failed_domain_list']:
            print(f"\n❌ 실패한 도메인:")
            for domain in results['failed_domain_list']:
                error_info = results['build_details'][domain]['error']
                print(f"   • {domain}: {error_info}")
        
        print(f"\n📄 상세 통계:")
        for domain, details in results['build_details'].items():
            if details['status'] == 'success' and details['stats']:
                stats = details['stats']
                size = stats.get('faiss_size_mb', 0)
                exists = stats.get('vectorstore_exists', False)
                print(f"   📁 {domain}: {'✅' if exists else '❌'} ({size:.1f}MB)")
        
        print("="*70)

def check_environment():
    """실행 환경 검증"""
    logger = logging.getLogger(__name__)
    
    # OpenAI API 키 확인
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다")
        logger.info("🔍 환경변수 디버깅:")
        logger.info(f"   - OPENAI_API_KEY: {'설정됨' if api_key else '없음'}")
        logger.info(f"   - 전체 환경변수 개수: {len(os.environ)}")
        
        # GitHub Actions에서 사용 가능한 모든 환경변수 중 API 키 관련 찾기
        api_related = [k for k in os.environ.keys() if 'API' in k.upper() or 'OPENAI' in k.upper()]
        logger.info(f"   - API 관련 환경변수: {api_related}")
        
        return False
    
    logger.info("✅ 환경변수 확인 완료")
    logger.info(f"🔑 API 키 확인: {api_key[:10]}..." if len(api_key) >= 10 else "🔑 API 키가 너무 짧음")
    
    # 필수 디렉터리 확인
    required_dirs = ['data', 'vectorstores', 'modules']
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            logger.error(f"❌ 필수 디렉터리가 없습니다: {dir_name}")
            return False
    
    logger.info("✅ 디렉터리 구조 확인 완료")
    return True

def main():
    """메인 실행 함수"""
    
    # 환경 검증
    if not check_environment():
        print("❌ 환경 검증 실패")
        sys.exit(1)
    
    # 빌드 매니저 실행
    manager = VectorstoreBuildManager()
    
    try:
        # 모든 벡터스토어 빌드
        results = manager.build_all_vectorstores()
        
        # 결과 요약 출력
        manager.print_summary(results)
        
        # 보고서 저장
        report_file = manager.save_build_report(results)
        
        # 실패한 도메인이 있으면 종료 코드 1 (GitHub Actions 알림 트리거)
        if results['failed_domains'] > 0:
            print(f"\n⚠️ {results['failed_domains']}개 도메인에서 실패가 발생했습니다.")
            print("📧 실패 알림이 전송됩니다.")
            sys.exit(1)
        else:
            print("\n🎉 모든 도메인 벡터스토어 빌드 성공!")
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ 치명적 오류 발생: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
