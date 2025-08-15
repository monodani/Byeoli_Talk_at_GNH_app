#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - 개선된 스마트 공지사항 로더 (BaseLoader 패턴 적용)

주요 개선사항:
- BaseLoader 표준 패턴 준수
- 플러그인 기반 동적 파싱 시스템 유지
- 설정 파일 경로 표준화 (/schemas)
- 캐시 TTL 6시간 적용
- 해시 기반 증분 빌드 지원
- Citation 모델 유효성 검사 오류 수정 (context 길이 제한)
"""

import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from abc import ABC, abstractmethod

# 프로젝트 모듈 임포트
from modules.base_loader import BaseLoader
from utils.textifier import TextChunk
from utils.config import config

# 로깅 설정
logger = logging.getLogger(__name__)

# ================================================================
# 1. 동적 파싱을 위한 플러그인 구조 (개선됨)
# ================================================================

class NoticeParser(ABC):
    """
    공지사항 파서의 기본 추상 클래스.
    모든 파서는 이 클래스를 상속받아 자동 등록됩니다.
    """
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'TOPIC_TYPE') and cls.TOPIC_TYPE:
            NoticeParser._registry[cls.TOPIC_TYPE] = cls

    @abstractmethod
    def can_parse(self, title: str, text: str, patterns: Dict[str, Any]) -> bool:
        """이 파서가 해당 공지사항을 처리할 수 있는지 판단"""
        pass

    @abstractmethod
    def parse(self, notice_text: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """공지사항 텍스트를 파싱하여 구조화된 데이터를 반환"""
        pass
    
    @abstractmethod
    def create_chunks(self, parsed_notice: Dict[str, Any], notice_number: int) -> List[TextChunk]:
        """파싱된 데이터를 기반으로 RAG용 청크를 생성"""
        pass

# ================================================================
# 2. 구체적인 파서 구현 (TTL 캐시 메타데이터 포함)
# ================================================================

class EvaluationNoticeParser(NoticeParser):
    """평가 관련 공지사항 전문 파서"""
    TOPIC_TYPE = "evaluation"
    
    def can_parse(self, title: str, text: str, patterns: Dict[str, Any]) -> bool:
        keywords = patterns.get('topic_patterns', {}).get(self.TOPIC_TYPE, {}).get('keywords', [])
        combined_text = (title + " " + text).lower()
        return any(keyword.lower() in combined_text for keyword in keywords)

    def parse(self, notice_text: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """평가 관련 주요 정보(마감기한, 점수, 제출방법) 추출"""
        parsed = {}
        
        # 마감기한 추출 (다양한 패턴 지원)
        deadline_patterns = [
            r'제출기한\s*[:：]\s*([^\n]+)',
            r'마감일\s*[:：]\s*([^\n]+)',
            r'(?:까지|이내)\s*제출'
        ]
        for pattern in deadline_patterns:
            match = re.search(pattern, notice_text)
            if match:
                parsed['deadline'] = match.group(1).strip()
                break
        
        # 점수 정보 추출
        score_match = re.search(r'(\d+)\s*점\s*만점', notice_text)
        if score_match:
            parsed['max_score'] = int(score_match.group(1))
        
        # 제출방법 추출
        submit_method = re.search(r'제출방법\s*[:：]\s*([^\n]+)', notice_text)
        if submit_method:
            parsed['submit_method'] = submit_method.group(1).strip()
        
        return parsed
        
    def create_chunks(self, parsed_notice: Dict[str, Any], notice_number: int) -> List[TextChunk]:
        """평가 공지사항 청크 생성 (캐시 TTL 6시간 적용)"""
        title = parsed_notice.get('title', '제목 없음')
        full_text = parsed_notice.get('full_text', '')
        
        # Pydantic Citation 모델의 context 필드 유효성 검사 오류를 방지하기 위해 
        # content의 길이를 200자로 제한합니다.
        truncated_content = full_text[:200] + '...' if len(full_text) > 200 else full_text
        
        # 기본 메타데이터 (캐시 TTL 6시간)
        base_metadata = {
            'source_file': 'notice.txt',
            'notice_number': notice_number,
            'notice_title': title,
            'topic_type': self.TOPIC_TYPE,
            'cache_ttl': 21600,  # 6시간 TTL
            'processing_date': datetime.now().isoformat(),
            'source_id': f'notice/notice.txt#section_{notice_number}',
            # ✅ context 길이 제한 로직을 적용한 content 필드
            'content': truncated_content
        }
        
        chunks = []
        
        # 1. 메인 요약 청크
        deadline_info = f"마감기한: {parsed_notice.get('deadline', '별도 명시 없음')}"
        score_info = f"점수: {parsed_notice.get('max_score', '미명시')}점" if parsed_notice.get('max_score') else ""
        
        summary = f"[{title}] 평가 관련 공지사항입니다. {deadline_info}. {score_info}"
        
        main_chunk = TextChunk(
            text=summary.strip(),
            source_id=base_metadata['source_id'],
            metadata={**base_metadata, 'chunk_type': 'summary', 'priority': 'high'}
        )
        chunks.append(main_chunk)
        
        # 2. 세부 정보 청크
        if parsed_notice.get('submit_method'):
            detail_chunk = TextChunk(
                text=f"[{title} - 제출방법]\n\n{parsed_notice.get('submit_method')}",
                source_id=base_metadata['source_id'],
                metadata={**base_metadata, 'chunk_type': 'details'}
            )
            chunks.append(detail_chunk)
        
        # 3. 원문 전체 청크
        full_chunk = TextChunk(
            text=f"[{title} - 원문]\n\n{full_text}",
            source_id=base_metadata['source_id'],
            metadata={**base_metadata, 'chunk_type': 'full_text'}
        )
        chunks.append(full_chunk)
        
        return chunks


class EnrollmentNoticeParser(NoticeParser):
    """입교 준비사항 관련 공지사항 전문 파서"""
    TOPIC_TYPE = "enrollment"

    def can_parse(self, title: str, text: str, patterns: Dict[str, Any]) -> bool:
        keywords = patterns.get('topic_patterns', {}).get(self.TOPIC_TYPE, {}).get('keywords', [])
        combined_text = (title + " " + text).lower()
        return any(keyword.lower() in combined_text for keyword in keywords)

    def parse(self, notice_text: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """체크리스트와 준비물 목록 추출"""
        parsed = {}
        
        # 체크리스트 항목 추출 (여러 형식 지원)
        checklist_patterns = [
            r'(?:\d+\.|\-|○|●|▪)\s*([^\n]+)',
            r'(?:준비물|지참물)\s*[:：]\s*([^\n]+)'
        ]
        
        checklist_items = []
        for pattern in checklist_patterns:
            items = re.findall(pattern, notice_text)
            checklist_items.extend([item.strip() for item in items if item.strip()])
        
        if checklist_items:
            parsed['checklist'] = list(set(checklist_items))  # 중복 제거
        
        # 연락처 정보 추출
        contact_match = re.search(r'문의\s*[:：]\s*([^\n]+)', notice_text)
        if contact_match:
            parsed['contact'] = contact_match.group(1).strip()
        
        return parsed

    def create_chunks(self, parsed_notice: Dict[str, Any], notice_number: int) -> List[TextChunk]:
        """입교 공지사항 청크 생성"""
        title = parsed_notice.get('title', '제목 없음')
        full_text = parsed_notice.get('full_text', '')
        
        # Pydantic Citation 모델의 context 필드 유효성 검사 오류를 방지하기 위해 
        # content의 길이를 200자로 제한합니다.
        truncated_content = full_text[:200] + '...' if len(full_text) > 200 else full_text
        
        base_metadata = {
            'source_file': 'notice.txt',
            'notice_number': notice_number,
            'notice_title': title,
            'topic_type': self.TOPIC_TYPE,
            'cache_ttl': 21600,  # 6시간 TTL
            'processing_date': datetime.now().isoformat(),
            'source_id': f'notice/notice.txt#section_{notice_number}',
            # ✅ context 길이 제한 로직을 적용한 content 필드
            'content': truncated_content
        }
        
        chunks = []
        
        # 1. 메인 요약 청크
        summary = f"[{title}] 입교 준비사항에 대한 안내입니다. 체크리스트를 확인해주세요."
        main_chunk = TextChunk(
            text=summary,
            metadata={**base_metadata, 'chunk_type': 'summary', 'priority': 'high'}
        )
        chunks.append(main_chunk)

        # 2. 체크리스트 전용 청크
        if parsed_notice.get('checklist'):
            checklist_text = "\n".join(f"• {item}" for item in parsed_notice.get('checklist', []))
            checklist_chunk = TextChunk(
                text=f"[{title} - 준비사항 체크리스트]\n\n{checklist_text}",
                metadata={**base_metadata, 'chunk_type': 'checklist'}
            )
            chunks.append(checklist_chunk)

        # 3. 연락처 정보 청크
        if parsed_notice.get('contact'):
            contact_chunk = TextChunk(
                text=f"[{title} - 문의처]\n\n{parsed_notice.get('contact')}",
                metadata={**base_metadata, 'chunk_type': 'contact'}
            )
            chunks.append(contact_chunk)

        return chunks


class RecruitmentNoticeParser(NoticeParser):
    """모집 공고 관련 전문 파서"""
    TOPIC_TYPE = "recruitment"

    def can_parse(self, title: str, text: str, patterns: Dict[str, Any]) -> bool:
        keywords = patterns.get('topic_patterns', {}).get(self.TOPIC_TYPE, {}).get('keywords', ['모집', '신청', '접수'])
        combined_text = (title + " " + text).lower()
        return any(keyword.lower() in combined_text for keyword in keywords)

    def parse(self, notice_text: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """모집 기간, 대상, 방법 추출"""
        parsed = {}
        
        # 모집기간 추출
        period_match = re.search(r'(?:모집기간|신청기간)\s*[:：]\s*([^\n]+)', notice_text)
        if period_match:
            parsed['recruitment_period'] = period_match.group(1).strip()
        
        # 모집대상 추출
        target_match = re.search(r'(?:모집대상|신청대상)\s*[:：]\s*([^\n]+)', notice_text)
        if target_match:
            parsed['target'] = target_match.group(1).strip()
        
        return parsed

    def create_chunks(self, parsed_notice: Dict[str, Any], notice_number: int) -> List[TextChunk]:
        """모집 공고 청크 생성"""
        title = parsed_notice.get('title', '제목 없음')
        full_text = parsed_notice.get('full_text', '')
        
        # Pydantic Citation 모델의 context 필드 유효성 검사 오류를 방지하기 위해 
        # content의 길이를 200자로 제한합니다.
        truncated_content = full_text[:200] + '...' if len(full_text) > 200 else full_text
        
        base_metadata = {
            'source_file': 'notice.txt',
            'notice_number': notice_number,
            'notice_title': title,
            'topic_type': self.TOPIC_TYPE,
            'cache_ttl': 21600,  # 6시간 TTL
            'processing_date': datetime.now().isoformat(),
            'source_id': f'notice/notice.txt#section_{notice_number}',
            # ✅ context 길이 제한 로직을 적용한 content 필드
            'content': truncated_content
        }
        
        summary_parts = [f"[{title}] 모집 공고입니다."]
        if parsed_notice.get('recruitment_period'):
            summary_parts.append(f"모집기간: {parsed_notice.get('recruitment_period')}")
        if parsed_notice.get('target'):
            summary_parts.append(f"대상: {parsed_notice.get('target')}")
        
        summary = " ".join(summary_parts)
        
        return [TextChunk(
            text=summary,
            metadata={**base_metadata, 'chunk_type': 'summary', 'priority': 'high'}
        )]


class FallbackNoticeParser(NoticeParser):
    """범용 폴백 파서 (모든 공지사항 처리 가능)"""
    TOPIC_TYPE = "general"
    
    def can_parse(self, title: str, text: str, patterns: Dict[str, Any]) -> bool:
        return True  # 항상 처리 가능 (최후의 수단)

    def parse(self, notice_text: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """기본 파싱: 제목과 본문 분리"""
        return {"full_text": notice_text}
        
    def create_chunks(self, parsed_notice: Dict[str, Any], notice_number: int) -> List[TextChunk]:
        """원문을 기본 청크로 생성"""
        title = parsed_notice.get('title', '제목 없음')
        full_text = parsed_notice.get('full_text', '')
        
        # Pydantic Citation 모델의 context 필드 유효성 검사 오류를 방지하기 위해 
        # content의 길이를 200자로 제한합니다.
        truncated_content = full_text[:200] + '...' if len(full_text) > 200 else full_text
        
        metadata = {
            'source_file': 'notice.txt',
            'notice_number': notice_number,
            'notice_title': title,
            'topic_type': 'general',
            'cache_ttl': 21600,  # 6시간 TTL
            'processing_date': datetime.now().isoformat(),
            'source_id': f'notice/notice.txt#section_{notice_number}',
            'chunk_type': 'general',
            # ✅ context 길이 제한 로직을 적용한 content 필드
            'content': truncated_content
        }
        
        return [TextChunk(text=f"[{title}]\n\n{full_text}", metadata=metadata)]

# ================================================================
# 3. BaseLoader 패턴을 준수하는 메인 로더
# ================================================================

class NoticeLoader(BaseLoader):
    """
    BaseLoader 패턴을 준수하는 공지사항 로더
    - 플러그인 기반 동적 파싱 시스템
    - 해시 기반 증분 빌드 지원
    - 캐시 TTL 6시간 적용
    """
    
    def __init__(self):
        super().__init__(
            domain="notice",
            source_dir=config.ROOT_DIR / "data" / "notice",
            vectorstore_dir=config.ROOT_DIR / "vectorstores" / "vectorstore_notice",
            index_name="notice_index"
        )
        self.patterns_config = self._load_patterns_config()
        self.parsers = NoticeParser._registry
        logger.info(f"✨ 등록된 파서: {list(self.parsers.keys())}")

    def _load_patterns_config(self) -> Dict[str, Any]:
        """schemas 디렉토리에서 패턴 설정 로드"""
        config_path = config.ROOT_DIR / "schemas" / "notice_patterns.json"
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"패턴 설정 로드 실패: {e}")
        
        # 기본 설정 (패턴 파일이 없을 경우)
        default_config = {
            "topic_patterns": {
                "evaluation": {
                    "keywords": ["평가", "과제", "제출기한", "마감일", "점수"],
                    "priority": 25
                },
                "enrollment": {
                    "keywords": ["입교", "교육생", "준비물", "체크리스트", "지참"],
                    "priority": 20
                },
                "recruitment": {
                    "keywords": ["모집", "신청", "접수", "선발"],
                    "priority": 18
                },
                "general": {
                    "keywords": ["공지", "안내", "알림"],
                    "priority": 10
                }
            }
        }
        
        # 기본 설정 파일 자동 생성
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            logger.info(f"기본 패턴 설정 파일 생성: {config_path}")
        except Exception as e:
            logger.warning(f"패턴 설정 파일 생성 실패: {e}")
        
        return default_config

    def process_domain_data(self) -> List[TextChunk]:
        """
        BaseLoader 인터페이스 구현: 공지사항 데이터 처리
        """
        all_chunks = []
        notice_file = self.source_dir / "notice.txt"
        
        if not notice_file.exists():
            logger.warning(f"공지사항 파일을 찾을 수 없습니다: {notice_file}")
            return all_chunks

        try:
            logger.info(f"🧠 스마트 공지사항 처리 시작: {notice_file}")
            
            # 파일을 섹션별로 분할 처리
            with open(notice_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            sections = [section.strip() for section in content.split('---') if section.strip()]

            for idx, section in enumerate(sections, 1):
                try:
                    # 1. 제목 추출
                    title = self._extract_title(section)
                    
                    # 2. 최적 파서 선택
                    parser = self._select_best_parser(title, section)
                    
                    # 3. 파싱 및 청크 생성
                    parsed_notice = parser.parse(section, self.patterns_config)
                    parsed_notice['title'] = title
                    parsed_notice['full_text'] = section
                    
                    chunks = parser.create_chunks(parsed_notice, idx)
                    all_chunks.extend(chunks)
                    
                    logger.info(f"📋 공지사항 #{idx} ({parser.TOPIC_TYPE}) 처리 완료: {len(chunks)}개 청크")

                except Exception as e:
                    logger.error(f"공지사항 #{idx} 처리 중 오류: {e}")
                    # 폴백 처리
                    fallback_chunk = self._create_emergency_fallback(section, idx)
                    if fallback_chunk:
                        all_chunks.append(fallback_chunk)
            
            logger.info(f"✅ 전체 처리 완료: {len(all_chunks)}개 청크 생성")
            
        except Exception as e:
            logger.error(f"공지사항 파일 처리 실패: {e}")
            
        return all_chunks

    def _extract_title(self, text: str) -> str:
        """다양한 형식의 제목 추출"""
        lines = text.strip().split('\n')
        if not lines:
            return "제목 없음"
        
        first_line = lines[0].strip()
        
        # 대괄호 패턴 우선 추출
        bracket_match = re.search(r'\[(.*?)\]', first_line)
        if bracket_match:
            return bracket_match.group(1).strip()
        
        # 첫 번째 줄을 제목으로 사용 (50자 제한)
        return first_line[:50] if len(first_line) > 50 else first_line

    def _select_best_parser(self, title: str, text: str) -> NoticeParser:
        """가장 적합한 파서를 점수 기반으로 선택"""
        best_parser = None
        best_score = -1
        
        # 등록된 모든 파서를 점수순으로 평가
        for topic_type, parser_cls in self.parsers.items():
            try:
                parser_instance = parser_cls()
                if parser_instance.can_parse(title, text, self.patterns_config):
                    priority = self.patterns_config.get('topic_patterns', {}).get(topic_type, {}).get('priority', 0)
                    if priority > best_score:
                        best_score = priority
                        best_parser = parser_instance
            except Exception as e:
                logger.warning(f"파서 {topic_type} 평가 중 오류: {e}")
        
        # 적합한 파서가 없으면 FallbackNoticeParser 사용
        if best_parser is None:
            best_parser = FallbackNoticeParser()
            logger.debug("FallbackNoticeParser 사용")
        
        return best_parser

    def _create_emergency_fallback(self, text: str, notice_number: int) -> Optional[TextChunk]:
        """최후의 비상 폴백 청크 생성"""
        try:
            title = self._extract_title(text)
            # 500자 -> 200자로 변경
            content = text[:200] + "..." if len(text) > 200 else text
            
            fallback_text = f"""
[공지사항 #{notice_number}] {title}

{content}

⚠️ 이 공지사항은 파싱 중 오류가 발생하여 기본 처리되었습니다.
정확한 정보는 원본 파일을 확인해주세요.
""".strip()
            
            return TextChunk(
                text=fallback_text,
                metadata={
                    'source_file': 'notice.txt',
                    'notice_number': notice_number,
                    'notice_title': title,
                    'topic_type': 'emergency_fallback',
                    'quality_level': 'fallback',
                    'cache_ttl': 21600,  # 6시간 TTL
                    'processing_date': datetime.now().isoformat(),
                    'source_id': f'notice/notice.txt#section_{notice_number}',
                    # ✅ context 길이 제한 로직을 적용한 content 필드
                    'content': content
                }
            )
        except Exception as e:
            logger.error(f"비상 폴백 생성 실패: {e}")
            return None

# ================================================================
# 4. 모듈 진입점
# ================================================================

def main():
    """개발/테스트용 진입점"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    loader = NoticeLoader()
    
    # BaseLoader의 표준 인터페이스 사용
    try:
        loader.load()  # FAISS 인덱스까지 자동 생성
        logger.info("✅ 공지사항 로더 실행 완료")
    except Exception as e:
        logger.error(f"❌ 로더 실행 실패: {e}")

if __name__ == '__main__':
    main()
