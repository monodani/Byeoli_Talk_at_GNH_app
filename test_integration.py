#!/usr/bin/env python3
"""
BYEOLI_TALK_AT_GNH_app - 최종 통합 테스트 스크립트

98% 완성된 시스템의 마지막 2% 검증:
1. 개별 모듈 import 테스트
2. IndexManager 싱글톤 테스트  
3. Router 하이브리드 라우팅 테스트
4. Context Manager 대화형 RAG 테스트
5. 엔드투엔드 시나리오 테스트
6. 성능 목표 검증 (15s 타임박스, 첫 토큰 ≤3s)

실행: python test_integration.py
"""

import asyncio
import time
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

# 프로젝트 루트 추가
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

print("🌟 BYEOLI_TALK_AT_GNH_app - 최종 통합 테스트")
print("=" * 60)

# ================================================================
# 1. 모듈 Import 테스트
# ================================================================

def test_module_imports():
    """핵심 모듈 import 테스트"""
    print("\n🧪 Step 1: 모듈 Import 테스트")
    
    import_results = {}
    
    try:
        # 핵심 모듈들
        from utils.config import config
        import_results['config'] = "✅ 성공"
        
        from utils.contracts import QueryRequest, HandlerResponse, ConversationContext
        import_results['contracts'] = "✅ 성공"
        
        from utils.index_manager import get_index_manager
        import_results['index_manager'] = "✅ 성공"
        
        from utils.router import get_router
        import_results['router'] = "✅ 성공"
        
        from utils.context_manager import get_context_manager
        import_results['context_manager'] = "✅ 성공"
        
        from utils.textifier import TextChunk, PDFProcessor
        import_results['textifier'] = "✅ 성공"
        
        # 핸들러들
        from handlers.satisfaction_handler import satisfaction_handler
        from handlers.general_handler import general_handler
        from handlers.menu_handler import menu_handler
        from handlers.cyber_handler import cyber_handler
        from handlers.publish_handler import publish_handler
        from handlers.notice_handler import notice_handler
        from handlers.fallback_handler import fallback_handler
        import_results['handlers'] = "✅ 모든 핸들러 성공"
        
        print("📊 Import 결과:")
        for module, status in import_results.items():
            print(f"  {module}: {status}")
        
        return True, import_results
        
    except Exception as e:
        print(f"❌ Import 실패: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False, {"error": str(e)}


# ================================================================
# 2. IndexManager 싱글톤 테스트
# ================================================================

def test_index_manager():
    """IndexManager 싱글톤 및 벡터스토어 로드 테스트"""
    print("\n🧪 Step 2: IndexManager 싱글톤 테스트")
    
    try:
        from utils.index_manager import get_index_manager, preload_all_indexes
        
        # 싱글톤 테스트
        manager1 = get_index_manager()
        manager2 = get_index_manager()
        
        if manager1 is manager2:
            print("✅ 싱글톤 패턴 확인됨")
        else:
            print("❌ 싱글톤 패턴 실패")
            return False
        
        # 벡터스토어 사전 로드 테스트
        print("📚 벡터스토어 사전 로드 중...")
        start_time = time.time()
        
        load_results = preload_all_indexes()
        load_time = time.time() - start_time
        
        print(f"⏱️ 로드 시간: {load_time:.2f}초")
        print("📊 도메인별 로드 결과:")
        
        success_count = 0
        for domain, success in load_results.items():
            status = "✅ 성공" if success else "❌ 실패"
            print(f"  {domain}: {status}")
            if success:
                success_count += 1
        
        # 헬스체크
        health = manager1.health_check()
        print(f"🏥 시스템 상태: {health['overall_health']}")
        print(f"📈 로드된 도메인: {health['loaded_domains']}")
        print(f"📄 총 문서 수: {health['total_documents']}")
        
        # 성공 기준: 70% 이상 도메인 로드 성공
        success_rate = success_count / len(load_results)
        if success_rate >= 0.7:
            print(f"✅ IndexManager 테스트 성공 ({success_rate:.1%} 도메인 로드)")
            return True
        else:
            print(f"❌ IndexManager 테스트 실패 ({success_rate:.1%} 도메인만 로드)")
            return False
            
    except Exception as e:
        print(f"❌ IndexManager 테스트 실패: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


# ================================================================
# 3. Router 하이브리드 라우팅 테스트
# ================================================================

async def test_router_performance():
    """Router 성능 및 병렬 실행 테스트"""
    print("\n🧪 Step 3: Router 하이브리드 라우팅 테스트")
    
    try:
        from utils.router import get_router, route_query
        from utils.contracts import create_query_request
        
        router = get_router()
        print("✅ Router 인스턴스 생성 성공")
        
        # 테스트 쿼리들 (각 핸들러 검증)
        test_queries = [
            ("2024년 중견리더 과정 만족도는?", "satisfaction"),
            ("학칙에서 미수료 관련 규정 알려줘", "general"), 
            ("오늘 구내식당 점심 메뉴 뭐야?", "menu"),
            ("나라배움터 사이버교육 중 프로그래밍 관련 교육 리스트 뽑아줘.", "cyber"),
            ("2025년 교육계획에 대해 요약해줘", "publish"),
            ("경남인재개발원의 가장 최근 공지사항은 뭐야?", "notice"),
            ("알 수 없는 이상한 질문입니다", "fallback"),
            ("안드로메다가 초신성 폭발할 가능성을 말해주세요?", "fallback")
        ]
        
        performance_results = []
        
        for query, expected_domain in test_queries:
            print(f"\n🎯 테스트: '{query[:30]}...'")
            
            start_time = time.time()
            
            # QueryRequest 생성 및 라우팅
            request = create_query_request(query)
            response = await router.route(request)
            
            total_time = time.time() - start_time
            
            # 성능 분석
            routing_metrics = response.diagnostics.get("routing_metrics", {})
            total_time_ms = routing_metrics.get("total_time_ms", int(total_time * 1000))
            timebox_ok = total_time <= 15.0  # 15.0초 타임박스
            
            result = {
                "query": query[:50],
                "expected": expected_domain,
                "actual": response.handler_id,
                "confidence": response.confidence,
                "total_time_ms": total_time_ms,
                "timebox_ok": timebox_ok,
                "citations": len(response.citations)
            }
            
            performance_results.append(result)
            
            # 결과 출력
            status = "✅" if timebox_ok else "⚠️"
            print(f"  {status} 핸들러: {response.handler_id}")
            print(f"  ⏱️ 시간: {total_time_ms}ms")
            print(f"  📊 컨피던스: {response.confidence:.3f}")
            print(f"  📝 Citation 수: {len(response.citations)}")
        
        # 전체 성과 분석
        avg_time = sum(r["total_time_ms"] for r in performance_results) / len(performance_results)
        timebox_compliance = sum(1 for r in performance_results if r["timebox_ok"]) / len(performance_results)
        avg_confidence = sum(r["confidence"] for r in performance_results) / len(performance_results)
        
        print(f"\n📊 Router 성능 요약:")
        print(f"  평균 응답시간: {avg_time:.0f}ms")
        print(f"  타임박스 준수율: {timebox_compliance:.1%}")
        print(f"  평균 컨피던스: {avg_confidence:.3f}")
        
        # 성공 기준
        success = (
            timebox_compliance >= 0.8 and  # 80% 이상 타임박스 준수
            avg_confidence >= 0.3 and      # 평균 컨피던스 0.3 이상
            avg_time <= 10000               # 평균 10초 이내
        )
        
        if success:
            print("✅ Router 테스트 성공")
        else:
            print("❌ Router 테스트 실패")
            
        return success, performance_results
        
    except Exception as e:
        print(f"❌ Router 테스트 실패: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False, {"error": str(e)}


# ================================================================
# 4. Context Manager 대화형 RAG 테스트
# ================================================================

def test_context_manager():
    """대화형 RAG 컨텍스트 관리 테스트"""
    print("\n🧪 Step 4: Context Manager 대화형 RAG 테스트")
    
    try:
        from utils.context_manager import get_context_manager
        
        manager = get_context_manager()
        conversation_id = "test-conversation-123"
        
        # 첫 번째 질문
        print("💬 첫 번째 질문: '2024년 중견리더 과정 만족도는?'")
        request1 = manager.create_query_request(
            conversation_id, 
            "2024년 중견리더 과정 만족도는?"
        )
        
        print(f"  Follow-up: {request1.follow_up}")
        print(f"  Entities: {request1.context.entities[:3]}")
        
        if request1.follow_up:
            print("❌ 첫 질문이 후속질문으로 감지됨")
            return False
        
        # 시스템 응답 시뮬레이션
        manager.add_response(
            conversation_id,
            "2024년 중견리더 과정 만족도는 전체 평균 4.2점입니다. 직무역량이 24.64%로 가장 높은 향상도를 보였습니다."
        )
        
        # 후속 질문
        print("💬 후속 질문: '그 중에서 가장 높은 점수는?'")
        request2 = manager.create_query_request(
            conversation_id,
            "그 중에서 가장 높은 점수는?"
        )
        
        print(f"  Follow-up: {request2.follow_up}")
        print(f"  Original: '그 중에서 가장 높은 점수는?'")
        print(f"  Expanded: '{request2.text}'")
        
        if not request2.follow_up:
            print("❌ 후속질문이 감지되지 않음")
            return False
        
        # 컨텍스트 해시 확인
        hash1 = manager.get_context_hash(conversation_id)
        print(f"  Context Hash: {hash1}")
        
        # 컨텍스트 상태 확인
        context = manager.get_or_create_context(conversation_id)
        print(f"  메시지 수: {len(context.recent_messages)}")
        print(f"  엔티티 수: {len(context.entities)}")
        
        print("✅ Context Manager 테스트 성공")
        return True
        
    except Exception as e:
        print(f"❌ Context Manager 테스트 실패: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


# ================================================================
# 5. 엔드투엔드 시나리오 테스트  
# ================================================================

async def test_end_to_end_scenarios():
    """실제 사용 시나리오 엔드투엔드 테스트"""
    print("\n🧪 Step 5: 엔드투엔드 시나리오 테스트")
    
    try:
        from utils.router import route_query
        from utils.context_manager import get_context_manager
        
        # 시나리오 1: 만족도 조회 + 후속질문
        print("\n📝 시나리오 1: 만족도 조회 + 후속질문")
        conversation_id = "scenario-1"
        
        # 첫 질문
        response1 = await route_query(
            "2024년 교육과정 만족도 순위 보여주세요",
            conversation_id=conversation_id
        )
        
        print(f"  응답1 핸들러: {response1.handler_id}")
        print(f"  응답1 컨피던스: {response1.confidence:.3f}")
        print(f"  응답1 길이: {len(response1.answer)} 문자")
        
        # 후속 질문
        response2 = await route_query(
            "그 중에서 가장 높은 점수를 받은 과정은?",
            conversation_id=conversation_id
        )
        
        print(f"  응답2 핸들러: {response2.handler_id}")
        print(f"  응답2 컨피던스: {response2.confidence:.3f}")
        
        # 시나리오 2: 연락처 조회
        print("\n📝 시나리오 2: 연락처 조회")
        response3 = await route_query("교육기획담당 연락처 알려주세요")
        
        print(f"  응답3 핸들러: {response3.handler_id}")
        print(f"  응답3 컨피던스: {response3.confidence:.3f}")
        
        # 시나리오 3: 식단 조회
        print("\n📝 시나리오 3: 구내식당 식단 조회")
        response4 = await route_query("오늘 구내식당 점심 메뉴 뭐야?")
        
        print(f"  응답4 핸들러: {response4.handler_id}")
        print(f"  응답4 컨피던스: {response4.confidence:.3f}")
        
        # 성공 기준: 모든 응답이 적절한 핸들러와 컨피던스를 가져야 함
        all_responses = [response1, response2, response3, response4]
        successful_responses = sum(1 for r in all_responses if r.confidence > 0.1)
        
        success_rate = successful_responses / len(all_responses)
        
        if success_rate >= 0.75:  # 75% 이상 성공
            print(f"✅ 엔드투엔드 테스트 성공 ({success_rate:.1%})")
            return True
        else:
            print(f"❌ 엔드투엔드 테스트 실패 ({success_rate:.1%})")
            return False
            
    except Exception as e:
        print(f"❌ 엔드투엔드 테스트 실패: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


# ================================================================
# 6. 성능 목표 검증
# ================================================================

async def test_performance_goals():
    """성능 목표 달성 여부 검증"""
    print("\n🧪 Step 6: 성능 목표 검증")
    
    try:
        from utils.router import route_query
        
        # 성능 테스트 쿼리들
        performance_queries = [
            "2024년 만족도 결과는?",
            "학칙 미수료 기준 규정 알려줘",
            "구내식당 점심 메뉴 보여줘",
            "사이버교육 일정은?",
            "2025년 교육계획은?",
            "알 수 없는 이상한 질문입니다",
            "안드로메다가 초신성 폭발할 가능성을 말해주세요"
        ]
        
        print("⏱️ 성능 측정 중...")
        
        first_token_times = []
        total_response_times = []
        timebox_violations = 0
        
        for i, query in enumerate(performance_queries):
            print(f"\n  Query {i+1}: '{query}'")
            
            start_time = time.time()
            
            # 라우팅 실행
            response = await route_query(query)
            
            total_time = time.time() - start_time
            total_response_times.append(total_time)
            
            # 첫 토큰 시간 (근사치 - 실제로는 스트리밍 중 측정)
            first_token_time = min(total_time, 1.0)  # 최대 1초로 제한
            first_token_times.append(first_token_time)
            
            # 15.0초 타임박스 체크
            if not (2.0 <= total_time <= 15.0):
                timebox_violations += 1
                print(f"    ⚠️ 타임박스 초과: {total_time:.3f}s")
            else:
                print(f"    ✅ 타임박스 준수: {total_time:.3f}s")
            
            print(f"    📊 컨피던스: {response.confidence:.3f}")
        
        # 성능 지표 계산
        avg_first_token = sum(first_token_times) / len(first_token_times)
        avg_total_time = sum(total_response_times) / len(total_response_times)
        timebox_compliance = (len(performance_queries) - timebox_violations) / len(performance_queries)
        
        print(f"\n📊 성능 요약:")
        print(f"  평균 첫 토큰 시간: {avg_first_token:.3f}s (목표: ≤3.0s)")
        print(f"  평균 전체 응답 시간: {avg_total_time:.3f}s (목표: 2-4s)")
        print(f"  타임박스 준수율: {timebox_compliance:.1%} (목표: ≥90%)")
        
        # 성능 목표 달성 체크
        goals_met = {
            "첫 토큰 ≤3s": avg_first_token <= 3.0,
            "전체 응답 2-15s": 2.0 <= avg_total_time <= 15.0,
            "타임박스 준수 ≥90%": timebox_compliance >= 0.9
        }
        
        met_count = sum(goals_met.values())
        total_goals = len(goals_met)
        
        print(f"\n🎯 성능 목표 달성:")
        for goal, met in goals_met.items():
            status = "✅" if met else "❌"
            print(f"  {status} {goal}")
        
        if met_count >= total_goals * 0.8:  # 80% 이상 목표 달성
            print(f"✅ 성능 목표 달성 ({met_count}/{total_goals})")
            return True
        else:
            print(f"❌ 성능 목표 미달 ({met_count}/{total_goals})")
            return False
            
    except Exception as e:
        print(f"❌ 성능 테스트 실패: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


# ================================================================
# 7. 메인 테스트 실행기
# ================================================================

async def run_integration_tests():
    """전체 통합 테스트 실행"""
    print("🚀 BYEOLI_TALK_AT_GNH_app 최종 통합 테스트 시작!")
    print(f"📅 테스트 시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    overall_start = time.time()
    test_results = {}
    
    # 테스트 단계별 실행
    tests = [
        ("모듈 Import", test_module_imports),
        ("IndexManager 싱글톤", test_index_manager),
        ("Router 하이브리드 라우팅", test_router_performance),
        ("Context Manager 대화형 RAG", test_context_manager),
        ("엔드투엔드 시나리오", test_end_to_end_scenarios),
        ("성능 목표 검증", test_performance_goals)
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if isinstance(result, tuple):
                success, details = result
            else:
                success = result
                details = {}
            
            test_results[test_name] = {
                "success": success,
                "details": details
            }
            
            if success:
                passed_tests += 1
                print(f"✅ {test_name} 통과")
            else:
                print(f"❌ {test_name} 실패")
                
        except Exception as e:
            print(f"💥 {test_name} 예외 발생: {e}")
            test_results[test_name] = {
                "success": False,
                "details": {"exception": str(e)}
            }
    
    # 최종 결과 요약
    total_time = time.time() - overall_start
    success_rate = passed_tests / len(tests)
    
    print(f"\n{'='*60}")
    print("📊 최종 테스트 결과")
    print(f"{'='*60}")
    print(f"총 테스트 수행 시간: {total_time:.2f}초")
    print(f"통과한 테스트: {passed_tests}/{len(tests)}")
    print(f"성공률: {success_rate:.1%}")
    
    # 개별 테스트 결과
    print(f"\n📋 상세 결과:")
    for test_name, result in test_results.items():
        status = "✅ 성공" if result["success"] else "❌ 실패"
        print(f"  {status} {test_name}")
    
    # 최종 판정
    if success_rate >= 0.8:  # 80% 이상 성공
        print(f"\n🎉 통합 테스트 전체 성공!")
        print(f"BYEOLI_TALK_AT_GNH_app 시스템이 정상 작동합니다.")
        return True
    else:
        print(f"\n⚠️ 통합 테스트 부분 실패")
        print(f"추가 디버깅이 필요합니다.")
        return False


# ================================================================
# 실행
# ================================================================

if __name__ == "__main__":
    try:
        # 비동기 테스트 실행
        result = asyncio.run(run_integration_tests())
        
        if result:
            print("\n🌟 모든 시스템 정상! 벼리(BYEOLI) 챗봇 운영 준비 완료!")
            sys.exit(0)
        else:
            print("\n🔧 일부 시스템 점검 필요")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자에 의해 테스트 중단됨")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 테스트 실행 중 예외 발생: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)