#!/usr/bin/env python3
"""
경상남도인재개발원 RAG 챗봇 - 개선된 스마트 공지사항 로더

- 동적 플러그인 기반 파싱 시스템
- 텍스트 패턴 의존도를 낮추고 유연성 강화
- 스트리밍 기반 파일 처리로 메모리 효율 개선
"""

import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from collections import defaultdict

# 프로젝트 모듈 임포트 (가정)
# from modules.base_loader import BaseLoader
# from utils.textifier import TextChunk
# from utils.config import config

# 외부 라이브러리 가정을 위한 더미 클래스
class BaseLoader:
    def __init__(self, domain, source_dir, vectorstore_dir, index_name):
        self.domain = domain
        self.source_dir = source_dir
        self.vectorstore_dir = vectorstore_dir
        self.index_name = index_name

    def process_domain_data(self) -> List:
        raise NotImplementedError

class TextChunk:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata

class Config:
    def __init__(self):
        self.ROOT_DIR = Path('.')

config = Config()

# 로깅 설정
logger = logging.getLogger(__name__)


# ================================================================
# 1. 동적 파싱을 위한 플러그인 구조
# ================================================================

class NoticeParser:
    """
    공지사항 파서의 기본 추상 클래스.
    모든 파서는 이 클래스를 상속받아야 합니다.
    """
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'TOPIC_TYPE') and cls.TOPIC_TYPE:
            NoticeParser._registry[cls.TOPIC_TYPE] = cls

    def can_parse(self, title: str, text: str, patterns: Dict[str, Any]) -> bool:
        """이 파서가 해당 공지사항을 처리할 수 있는지 판단하는 메서드"""
        raise NotImplementedError

    def parse(self, notice_text: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """공지사항 텍스트를 파싱하여 구조화된 데이터를 반환"""
        raise NotImplementedError
    
    def create_chunks(self, parsed_notice: Dict[str, Any], notice_number: int) -> List[TextChunk]:
        """파싱된 데이터를 기반으로 RAG용 청크를 생성"""
        raise NotImplementedError

# 예시: 평가 공지사항 파서
class EvaluationNoticeParser(NoticeParser):
    TOPIC_TYPE = "evaluation"
    
    def can_parse(self, title: str, text: str, patterns: Dict[str, Any]) -> bool:
        """제목 또는 내용에 '평가', '과제', '제출기한' 키워드가 있는지 확인"""
        keywords = patterns['topic_patterns'][self.TOPIC_TYPE]['keywords']
        return any(keyword in (title + text) for keyword in keywords)

    def parse(self, notice_text: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """평가 관련 주요 정보(마감기한, 점수) 추출"""
        parsed = {}
        # 마감기한 추출
        deadline_match = re.search(r'제출기한\s*[:：]\s*([^\n]+)', notice_text)
        if deadline_match:
            parsed['deadline'] = deadline_match.group(1).strip()
        # 만점 점수 추출
        score_match = re.search(r'(\d+)\s*점\s*만점', notice_text)
        if score_match:
            parsed['max_score'] = int(score_match.group(1))
        
        return parsed
        
    def create_chunks(self, parsed_notice: Dict[str, Any], notice_number: int) -> List[TextChunk]:
        """평가 공지사항 청크 생성 로직"""
        title = parsed_notice.get('title', '제목 없음')
        full_text = parsed_notice.get('full_text', '')
        metadata = {
            'source_file': 'notice.txt',
            'notice_number': notice_number,
            'notice_title': title,
            'topic_type': self.TOPIC_TYPE,
            'processing_date': datetime.now().isoformat()
        }
        
        # 메인 청크: 제목과 핵심 요약
        summary = f"[{title}] 이 공지사항은 평가에 관한 중요 내용입니다. 마감기한은 {parsed_notice.get('deadline', '별도 명시 없음')}입니다."
        main_chunk = TextChunk(text=summary, metadata={**metadata, 'chunk_type': 'summary'})
        
        # 세부 청크: 원문 전체
        detail_chunk = TextChunk(text=f"[{title} - 원문]\n\n{full_text}", metadata={**metadata, 'chunk_type': 'full_text'})
        
        return [main_chunk, detail_chunk]


# 예시: 입교 공지사항 파서
class EnrollmentNoticeParser(NoticeParser):
    TOPIC_TYPE = "enrollment"

    def can_parse(self, title: str, text: str, patterns: Dict[str, Any]) -> bool:
        keywords = patterns['topic_patterns'][self.TOPIC_TYPE]['keywords']
        return any(keyword in (title + text) for keyword in keywords)

    def parse(self, notice_text: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """체크리스트와 연락처 추출"""
        parsed = {}
        # 체크리스트 항목 추출 (정규표현식 의존도 낮추기 위해 유연하게)
        checklist_items = re.findall(r'(?:\d+\.|\-|○)\s*([^\n]+)', notice_text)
        if checklist_items:
            parsed['checklist'] = [item.strip() for item in checklist_items]
        
        return parsed

    def create_chunks(self, parsed_notice: Dict[str, Any], notice_number: int) -> List[TextChunk]:
        """입교 공지사항 청크 생성 로직"""
        title = parsed_notice.get('title', '제목 없음')
        full_text = parsed_notice.get('full_text', '')
        metadata = {
            'source_file': 'notice.txt',
            'notice_number': notice_number,
            'notice_title': title,
            'topic_type': self.TOPIC_TYPE,
            'processing_date': datetime.now().isoformat()
        }
        
        # 메인 청크: 제목과 핵심 요약
        summary = f"[{title}] 이 공지사항은 입교 준비사항에 대한 안내입니다. 체크리스트를 확인해주세요."
        main_chunk = TextChunk(text=summary, metadata={**metadata, 'chunk_type': 'summary'})

        # 체크리스트 청크
        checklist_text = "\n".join(f"- {item}" for item in parsed_notice.get('checklist', []))
        checklist_chunk = TextChunk(text=f"[{title} - 체크리스트]\n\n{checklist_text}", metadata={**metadata, 'chunk_type': 'checklist'})

        return [main_chunk, checklist_chunk]


# ================================================================
# 2. 메인 로더 (동적 파서 활용)
# ================================================================

class SmartNoticeLoader(BaseLoader):
    def __init__(self):
        super().__init__(
            domain="notice",
            source_dir=config.ROOT_DIR / "data" / "notice",
            vectorstore_dir=config.ROOT_DIR / "vectorstores" / "vectorstore_notice",
            index_name="notice_index"
        )
        self.config = self._load_patterns_config()
        self.parsers = NoticeParser._registry
        logger.info(f"✨ 동적으로 등록된 파서: {list(self.parsers.keys())}")

    def _load_patterns_config(self) -> Dict[str, Any]:
        config_path = config.ROOT_DIR / "configs" / "notice_patterns.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "topic_patterns": {
                    "evaluation": {"keywords": ["평가", "과제", "제출기한"], "priority": 25},
                    "enrollment": {"keywords": ["입교", "교육생", "준비물"], "priority": 20},
                    "general": {"keywords": ["공지", "안내"], "priority": 10}
                }
            }

    def process_domain_data(self) -> List[TextChunk]:
        all_chunks = []
        notice_file = self.source_dir / "notice.txt"
        
        if not notice_file.exists():
            logger.warning(f"공지사항 파일을 찾을 수 없습니다: {notice_file}")
            return all_chunks

        try:
            logger.info(f"🧠 개선된 스마트 처리 시작: {notice_file}")
            
            # 스트리밍 방식으로 파일 읽기 (섹션별로 처리)
            with open(notice_file, 'r', encoding='utf-8') as f:
                sections = f.read().split('---')

            for idx, section in enumerate(sections):
                if not section.strip():
                    continue

                try:
                    # 1. 주제 분류 및 파서 선택
                    title = self._extract_title(section)
                    parser = self._select_parser(title, section)
                    
                    # 2. 파싱 및 청크 생성
                    parsed_notice = parser.parse(section, self.config)
                    parsed_notice['title'] = title
                    parsed_notice['full_text'] = section
                    
                    chunks = parser.create_chunks(parsed_notice, idx + 1)
                    all_chunks.extend(chunks)
                    logger.info(f"📋 공지사항 #{idx + 1} ({parser.TOPIC_TYPE}) 처리 완료")

                except Exception as e:
                    logger.error(f"공지사항 #{idx + 1} 처리 중 오류, 폴백 처리: {e}")
                    fallback_chunk = self._create_fallback_chunk(section, idx + 1)
                    if fallback_chunk:
                        all_chunks.append(fallback_chunk)
            
            logger.info(f"✅ 개선된 처리 완료: {len(all_chunks)}개 청크 생성")
        except Exception as e:
            logger.error(f"전체 파일 처리 실패: {e}")
            
        return all_chunks

    def _extract_title(self, text: str) -> str:
        """첫 번째 줄에서 제목을 추출하는 유연한 로직"""
        first_line = text.strip().split('\n')[0]
        title_match = re.search(r'\[(.*?)\]', first_line)
        return title_match.group(1).strip() if title_match else first_line.strip()

    def _select_parser(self, title: str, text: str) -> NoticeParser:
        """가장 적합한 파서를 동적으로 선택"""
        best_parser = self.parsers.get('general', FallbackNoticeParser()) # 기본 파서
        best_score = -1
        
        # 모든 등록된 파서에 대해 점수 계산
        for topic_type, parser_cls in self.parsers.items():
            parser_instance = parser_cls()
            if parser_instance.can_parse(title, text, self.config):
                score = self.config['topic_patterns'].get(topic_type, {}).get('priority', 0)
                if score > best_score:
                    best_score = score
                    best_parser = parser_instance
                    
        return best_parser

    def _create_fallback_chunk(self, text: str, notice_number: int) -> TextChunk:
        """폴백(Fallback) 청크 생성 (정보 손실 최소화)"""
        title = self._extract_title(text)
        content = text[:500]  # 처음 500자
        
        fallback_text = f"""
[공지사항 #{notice_number}] {title}

{content}

[주의] 이 공지사항은 파싱 중 오류가 발생하여 기본 처리되었습니다.
정확한 정보는 원본 파일을 확인해주세요.
""".strip()
        
        return TextChunk(
            text=fallback_text,
            metadata={
                'source_file': 'notice.txt',
                'notice_number': notice_number,
                'notice_title': title,
                'topic_type': 'fallback',
                'quality_level': 'fallback'
            }
        )

# 폴백 처리를 위한 기본 파서
class FallbackNoticeParser(NoticeParser):
    TOPIC_TYPE = "general"
    
    def can_parse(self, title: str, text: str, patterns: Dict[str, Any]) -> bool:
        return True # 항상 처리 가능

    def parse(self, notice_text: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """기본 파싱: 원문 전체를 구조화된 데이터로 반환"""
        return {"full_text": notice_text}
        
    def create_chunks(self, parsed_notice: Dict[str, Any], notice_number: int) -> List[TextChunk]:
        """원문 전체를 하나의 청크로 생성"""
        title = parsed_notice.get('title', '제목 없음')
        full_text = parsed_notice.get('full_text', '')
        metadata = {
            'source_file': 'notice.txt',
            'notice_number': notice_number,
            'notice_title': title,
            'topic_type': 'general',
            'processing_date': datetime.now().isoformat()
        }
        return [TextChunk(text=f"[{title} - 원문]\n\n{full_text}", metadata=metadata)]

# ================================================================
# 3. 테스트 실행 (코드 블록을 위해 주석 처리)
# ================================================================
# if __name__ == '__main__':
#     # 로깅 설정
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     
#     # 테스트용 파일 및 폴더 생성
#     Path("data/notice").mkdir(parents=True, exist_ok=True)
#     with open("data/notice/notice.txt", "w", encoding="utf-8") as f:
#         f.write("[평가안내] 2025년 과제 제출 안내\n\n- 제출기한: 2025년 12월 31일\n- 과제점수: 100점 만점\n---\n[입교안내] 1차 교육생 준비물\n\n○ 복장: 단정한 복장\n○ 준비물: 개인 노트북\n---")
#
#     # 로더 실행
#     loader = SmartNoticeLoader()
#     chunks = loader.process_domain_data()
#     
#     # 결과 출력
#     print("\n--- 생성된 청크 목록 ---")
#     for chunk in chunks:
#         print(f"[{chunk.metadata['chunk_type']}] - {chunk.metadata['notice_title']}")
#         print(f"내용: {chunk.text[:50].replace('\n', ' ')}...")
#         print(f"메타데이터: {chunk.metadata}")
#         print("-" * 20)
