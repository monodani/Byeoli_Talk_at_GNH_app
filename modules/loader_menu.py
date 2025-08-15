#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - 식단표 이미지 로더 (하이브리드 방식)

주요 특징:
- ChatGPT API를 활용한 이미지 → 텍스트 변환 (기존 방식 유지)
- BaseLoader 패턴 완전 준수
- 주차별 캐시 시스템 (menu_YYYYWW.txt)
- 6시간 TTL 적용
- 요일별/식사별 최적화된 청킹
"""

import logging
import json
import base64
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from calendar import timegm

# 프로젝트 모듈 임포트
from modules.base_loader import BaseLoader
from utils.textifier import TextChunk
from utils.config import config

# 외부 라이브러리
import requests
from PIL import Image

# 로깅 설정
logger = logging.getLogger(__name__)

# ================================================================
# 1. ChatGPT API 통합 클래스
# ================================================================

class ChatGPTImageProcessor:
    """
    ChatGPT API를 활용한 이미지 처리 전용 클래스
    - 식단표 이미지 → 구조화된 JSON 변환
    - API 키 환경변수 관리
    - 에러 처리 및 폴백
    """
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4o-mini"  # 이미지 처리 최적화 모델
        
    def _get_api_key(self) -> str:
        """환경변수에서 OpenAI API 키 안전하게 로드"""
        # config.get() 대신 속성에 직접 접근하도록 수정
        api_key = config.OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다. "
                ".env 파일 또는 환경변수에 API 키를 설정해주세요."
            )
        return api_key
    
    def _encode_image_to_base64(self, image_path: Path) -> str:
        """이미지 파일을 base64로 인코딩"""
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"이미지 인코딩 실패: {e}")
            raise
    
    def extract_menu_from_image(self, image_path: Path) -> Dict[str, Any]:
        """
        ChatGPT API로 식단표 이미지를 구조화된 JSON으로 변환
        
        Returns:
            Dict[str, Any]: {
                "월요일": {"조식": ["메뉴1", "메뉴2"], "중식": [...], "석식": [...]},
                "화요일": {...},
                ...
            }
        """
        try:
            logger.info(f"🤖 ChatGPT API로 식단표 파싱 시작: {image_path}")
            
            # 1. 이미지 base64 인코딩
            base64_image = self._encode_image_to_base64(image_path)
            
            # 2. ChatGPT API 호출
            menu_data = self._call_chatgpt_api(base64_image)
            
            logger.info(f"✅ ChatGPT API 파싱 완료: {len(menu_data)}개 요일")
            return menu_data
            
        except Exception as e:
            logger.error(f"ChatGPT API 처리 실패: {e}")
            return self._create_fallback_menu()
    
    def _call_chatgpt_api(self, base64_image: str) -> Dict[str, Any]:
        """ChatGPT API 실제 호출"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 프롬프트: 식단표 구조화 전용
        prompt = """
다음은 경상남도인재개발원의 주간 식단표 이미지입니다.
이 이미지를 분석하여 요일별, 식사별로 메뉴를 정확히 추출해주세요.

출력 형식은 반드시 다음 JSON 구조를 따라주세요:
{
    "월요일": {
        "조식": ["메뉴1", "메뉴2", ...],
        "중식": ["메뉴1", "메뉴2", ...], 
        "석식": ["메뉴1", "메뉴2", ...]
    },
    "화요일": { ... },
    "수요일": { ... },
    "목요일": { ... },
    "금요일": { ... }
}

주의사항:
1. 메뉴명은 정확히 추출하되, 불분명한 경우 "메뉴 확인 필요"로 표시
2. 빈 항목은 빈 배열 []로 표시
3. 한글 요일명 사용 (월요일, 화요일, ...)
4. 식사 구분은 "조식", "중식", "석식"으로 통일
5. JSON 형식만 응답하고 다른 설명은 추가하지 마세요
"""
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"  # 고해상도 분석
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1  # 일관성을 위해 낮은 temperature
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        # 응답 파싱
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # JSON 추출 및 파싱
        try:
            # content에서 JSON 부분만 추출
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                menu_data = json.loads(json_str)
                return menu_data
            else:
                raise ValueError("JSON 형식을 찾을 수 없습니다")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            logger.debug(f"원본 응답: {content}")
            return self._create_fallback_menu()
    
    def _create_fallback_menu(self) -> Dict[str, Any]:
        """API 실패 시 기본 메뉴 생성"""
        fallback_menu = {
            "월요일": {
                "조식": ["식단 정보 로드 실패"],
                "중식": ["원본 이미지를 확인해주세요"],
                "석식": ["관리자에게 문의하세요"]
            },
            "화요일": {"조식": [], "중식": [], "석식": []},
            "수요일": {"조식": [], "중식": [], "석식": []},
            "목요일": {"조식": [], "중식": [], "석식": []},
            "금요일": {"조식": [], "중식": [], "석식": []}
        }
        logger.warning("폴백 메뉴 데이터 생성됨")
        return fallback_menu

# ================================================================
# 2. BaseLoader 패턴 준수 메인 로더
# ================================================================

class MenuLoader(BaseLoader):
    """
    BaseLoader 패턴을 준수하는 식단표 로더
    - ChatGPT API 활용한 이미지 파싱
    - 주차별 캐시 관리 시스템
    - 6시간 TTL 적용
    - 요일별/식사별 최적화 청킹
    """
    
    def __init__(self):
        super().__init__(
            domain="menu",
            source_dir=config.ROOT_DIR / "data" / "menu",
            vectorstore_dir=config.ROOT_DIR / "vectorstores" / "vectorstore_menu",
            index_name="menu_index"
        )
        self.chatgpt_processor = ChatGPTImageProcessor()
        self.cache_ttl = 21600  # 6시간 (초)
        
    def process_domain_data(self) -> List[TextChunk]:
        """
        BaseLoader 인터페이스 구현: 식단표 데이터 처리
        """
        all_chunks = []
        menu_image = self.source_dir / "menu.png"
        
        if not menu_image.exists():
            logger.warning(f"식단표 이미지를 찾을 수 없습니다: {menu_image}")
            return all_chunks
        
        try:
            logger.info(f"🍽️ 식단표 처리 시작: {menu_image}")
            
            # 1. 캐시된 메뉴 데이터 확인 또는 새로 추출
            menu_data = self._get_or_extract_menu_data(menu_image)
            
            # 2. 메뉴 데이터를 TextChunk로 변환
            if menu_data:
                chunks = self._create_menu_chunks(menu_data)
                all_chunks.extend(chunks)
                logger.info(f"✅ 식단표 처리 완료: {len(chunks)}개 청크 생성")
            else:
                logger.warning("메뉴 데이터가 비어있습니다")
            
        except Exception as e:
            logger.error(f"식단표 처리 실패: {e}")
            # 비상 폴백 청크 생성
            fallback_chunk = self._create_emergency_fallback_chunk()
            if fallback_chunk:
                all_chunks.append(fallback_chunk)
        
        return all_chunks
    
    def _get_or_extract_menu_data(self, image_path: Path) -> Dict[str, Any]:
        """캐시 확인 후 필요시 ChatGPT API로 메뉴 추출"""
        
        # 1. 현재 주차 계산
        current_week = self._get_current_week_string()
        cache_file = self.source_dir / f"menu_{current_week}.txt"
        
        # 2. 캐시 유효성 확인
        if cache_file.exists() and self._is_cache_valid(cache_file, image_path):
            logger.info(f"📁 캐시된 식단 데이터 사용: {cache_file}")
            return self._load_cached_menu(cache_file)
        
        # 3. 새로 추출 필요 시 ChatGPT API 호출
        logger.info("🤖 ChatGPT API로 새 식단 데이터 추출")
        menu_data = self.chatgpt_processor.extract_menu_from_image(image_path)
        
        # 4. 캐시 저장
        if menu_data:
            self._save_menu_cache(cache_file, menu_data)
            logger.info(f"💾 식단 데이터 캐시 저장: {cache_file}")
        
        return menu_data
    
    def _get_current_week_string(self) -> str:
        """현재 주차를 YYYYWW 형식으로 반환"""
        now = datetime.now()
        year = now.year
        # ISO 주차 계산 (월요일 시작)
        week = now.isocalendar()[1]
        return f"{year}W{week:02d}"
    
    def _is_cache_valid(self, cache_file: Path, image_path: Path) -> bool:
        """캐시 파일의 유효성 검사"""
        try:
            # 파일 수정 시간 확인
            cache_mtime = cache_file.stat().st_mtime
            image_mtime = image_path.stat().st_mtime
            current_time = datetime.now().timestamp()
            
            # 1. 이미지가 캐시보다 최신인지 확인
            if image_mtime > cache_mtime:
                logger.info("이미지 파일이 캐시보다 최신입니다")
                return False
            
            # 2. TTL 만료 확인 (6시간)
            if current_time - cache_mtime > self.cache_ttl:
                logger.info("캐시 TTL 만료 (6시간)")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"캐시 유효성 검사 실패: {e}")
            return False
    
    def _load_cached_menu(self, cache_file: Path) -> Dict[str, Any]:
        """캐시된 메뉴 데이터 로드"""
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                content = f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"캐시 로드 실패: {e}")
            return {}
    
    def _save_menu_cache(self, cache_file: Path, menu_data: Dict[str, Any]) -> None:
        """메뉴 데이터를 캐시 파일로 저장"""
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(menu_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
    
    def _create_menu_chunks(self, menu_data: Dict[str, Any]) -> List[TextChunk]:
        """메뉴 데이터를 검색 최적화된 TextChunk로 변환"""
        chunks = []
        current_week = self._get_current_week_string()
        
        # 전체 주간 메뉴 요약 청크 (높은 우선순위)
        weekly_summary = self._create_weekly_summary_chunk(menu_data, current_week)
        if weekly_summary:
            chunks.append(weekly_summary)
        
        # 요일별/식사별 개별 청크
        for day, meals in menu_data.items():
            for meal_type, menu_items in meals.items():
                if menu_items:  # 빈 메뉴는 제외
                    chunk = self._create_meal_chunk(day, meal_type, menu_items, current_week)
                    if chunk:
                        chunks.append(chunk)
        
        return chunks
    
    def _create_weekly_summary_chunk(self, menu_data: Dict[str, Any], week: str) -> Optional[TextChunk]:
        """주간 식단 전체 요약 청크 생성"""
        try:
            summary_lines = [f"📅 {week} 주간 식단표"]
            
            for day, meals in menu_data.items():
                day_summary = f"▫️ {day}: "
                meal_summaries = []
                
                for meal_type, menu_items in meals.items():
                    if menu_items:
                        main_menu = menu_items[0] if menu_items else "메뉴 없음"
                        meal_summaries.append(f"{meal_type}({main_menu})")
                
                day_summary += ", ".join(meal_summaries) if meal_summaries else "식단 정보 없음"
                summary_lines.append(day_summary)
            
            summary_text = "\n".join(summary_lines)
            
            return TextChunk(
                text=summary_text,
                metadata={
                    'source_file': 'menu.png',
                    'week': week,
                    'chunk_type': 'weekly_summary',
                    'cache_ttl': self.cache_ttl,
                    'priority': 'high',
                    'source_id': f'menu/menu.png#{week}_summary',
                    'processing_date': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"주간 요약 청크 생성 실패: {e}")
            return None
    
    def _create_meal_chunk(self, day: str, meal_type: str, menu_items: List[str], week: str) -> Optional[TextChunk]:
        """개별 식사 청크 생성"""
        try:
            # 메뉴 항목들을 자연어로 구성
            menu_text = ", ".join(menu_items)
            
            chunk_text = f"""🍽️ {day} {meal_type}

메뉴: {menu_text}

#{day} #{meal_type} #식단 #경남인재개발원 #{week}"""
            
            return TextChunk(
                text=chunk_text,
                source_id=f'menu/menu.png#{week}_{day}_{meal_type}',
                metadata={
                    'source_file': 'menu.png',
                    'day': day,
                    'meal_type': meal_type,
                    'week': week,
                    'menu_count': len(menu_items),
                    'chunk_type': 'meal_detail',
                    'cache_ttl': self.cache_ttl,
                    'source_id': f'menu/menu.png#{current_week}_emergency',
                    'processing_date': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"식사 청크 생성 실패 ({day} {meal_type}): {e}")
            return None
    
    def _create_emergency_fallback_chunk(self) -> Optional[TextChunk]:
        """최후의 비상 폴백 청크 생성"""
        try:
            current_week = self._get_current_week_string()
            
            fallback_text = f"""⚠️ {current_week} 식단표 정보 로드 실패

죄송합니다. 현재 식단표 정보를 불러올 수 없습니다.
다음을 확인해주세요:

1. 원본 이미지 파일 (menu.png) 존재 여부
2. OpenAI API 키 설정 상태
3. 네트워크 연결 상태

정확한 식단 정보는 경남인재개발원 구내식당에 직접 문의해주세요.
📞 문의: 055-254-2100 (총무담당)"""
            
            return TextChunk(
                text=fallback_text,
                source_id=f'menu/menu.png#{week}_{day}_{meal_type}',
                metadata={
                    'source_file': 'menu.png',
                    'week': current_week,
                    'chunk_type': 'emergency_fallback',
                    'quality_level': 'fallback',
                    'cache_ttl': 3600,  # 1시간 TTL (짧게)
                    'source_id': f'menu/menu.png#{current_week}_emergency',
                    'processing_date': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"비상 폴백 청크 생성 실패: {e}")
            return None

# ================================================================
# 3. 유틸리티 함수
# ================================================================

def validate_menu_image(image_path: Path) -> bool:
    """식단표 이미지 파일 유효성 검사"""
    try:
        if not image_path.exists():
            logger.error(f"이미지 파일이 존재하지 않습니다: {image_path}")
            return False
        
        # 이미지 파일 검증
        with Image.open(image_path) as img:
            # 최소 크기 검증 (너무 작으면 OCR 정확도 떨어짐)
            if img.width < 300 or img.height < 300:
                logger.warning(f"이미지 크기가 너무 작습니다: {img.width}x{img.height}")
                return False
            
            # 파일 크기 검증 (너무 크면 API 호출 실패 가능)
            file_size = image_path.stat().st_size / (1024 * 1024)  # MB
            if file_size > 20:  # 20MB 제한
                logger.warning(f"이미지 파일이 너무 큽니다: {file_size:.2f}MB")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"이미지 검증 실패: {e}")
        return False

# ================================================================
# 4. 모듈 진입점
# ================================================================

def main():
    """개발/테스트용 진입점"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 이미지 파일 검증
        menu_image = config.ROOT_DIR / "data" / "menu" / "menu.png"
        if not validate_menu_image(menu_image):
            logger.error("❌ 이미지 파일 검증 실패")
            return
        
        # 로더 실행
        loader = MenuLoader()
        loader.load()  # BaseLoader의 표준 인터페이스 사용
        
        logger.info("✅ 식단표 로더 실행 완료")
        
    except Exception as e:
        logger.error(f"❌ 로더 실행 실패: {e}")

if __name__ == '__main__':
    main()
