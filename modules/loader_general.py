"""
General domain data loader for Byeoli_Talk_at_GNH_app.
Processes operation_test.pdf, hakchik.pdf, and task_telephone.csv files.
Preserves existing Colab templates while implementing BaseLoader pattern.
"""

import os
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

from modules.base_loader import BaseLoader, TextChunk
from utils.config import config


class GeneralLoader(BaseLoader):
    """
    General domain loader inheriting from BaseLoader.
    Handles PDF documents and contact CSV with domain-specific templates.
    """
    
    def __init__(self):
        super().__init__(
            loader_id="general",
            source_dir="data/general", 
            target_dir="vectorstores/vectorstore_general",
            schema_dir="schemas"
        )
        
    def get_file_patterns(self) -> List[str]:
        """Return file patterns to process in general domain."""
        return ["*.pdf", "task_telephone.csv"]
    
    def get_schema_files(self) -> Dict[str, str]:
        """Return schema validation files for CSV data."""
        return {
            "task_telephone.csv": "general.schema.json"
        }
    
    def process_domain_data(self, file_chunks: Dict[str, List[TextChunk]]) -> List[TextChunk]:
        """
        Process general domain data with file-specific logic.
        
        Args:
            file_chunks: Dictionary mapping filenames to their text chunks
            
        Returns:
            List of processed TextChunks with appropriate metadata
        """
        processed_chunks = []
        
        # Process PDF files (operation_test.pdf, hakchik.pdf)
        pdf_chunks = self._process_pdf_chunks(file_chunks)
        processed_chunks.extend(pdf_chunks)
        
        # Process CSV file (task_telephone.csv) 
        csv_chunks = self._process_csv_chunks(file_chunks)
        processed_chunks.extend(csv_chunks)
        
        self.logger.info(f"Processed {len(processed_chunks)} total chunks from general domain")
        return processed_chunks
    
    def _process_pdf_chunks(self, file_chunks: Dict[str, List[TextChunk]]) -> List[TextChunk]:
        """Process PDF files with general document metadata."""
        pdf_chunks = []
        
        for filename, chunks in file_chunks.items():
            if not filename.endswith('.pdf'):
                continue
                
            self.logger.info(f"Processing PDF file: {filename}")
            
            # Determine PDF category
            if filename == "hakchik.pdf":
                category = "regulations"
                doc_type = "통합규정문서"  # 학칙+감점기준+전결규정
            elif filename == "operation_test.pdf":
                category = "operations"
                doc_type = "운영평가계획"
            else:
                category = "general"
                doc_type = "일반문서"
            
            for i, chunk in enumerate(chunks):
                # Enhance chunk with general PDF metadata
                enhanced_chunk = TextChunk(
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "source_file": filename,
                        "file_type": "pdf",
                        "category": category,
                        "doc_type": doc_type,
                        "domain": "general",
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                pdf_chunks.append(enhanced_chunk)
                
        self.logger.info(f"Generated {len(pdf_chunks)} PDF chunks")
        return pdf_chunks
    
    def _process_csv_chunks(self, file_chunks: Dict[str, List[TextChunk]]) -> List[TextChunk]:
        """Process task_telephone.csv with preserved Colab template."""
        csv_chunks = []
        
        # Look for task_telephone.csv
        if "task_telephone.csv" not in file_chunks:
            self.logger.warning("task_telephone.csv not found in file_chunks")
            return csv_chunks
            
        self.logger.info("Processing task_telephone.csv with preserved template")
        
        # Read CSV file directly to apply template
        csv_path = os.path.join(self.source_path, "task_telephone.csv")
        
        try:
            # Try UTF-8 first, fallback to other encodings if needed
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                self.logger.warning("UTF-8 encoding failed, trying cp949...")
                df = pd.read_csv(csv_path, encoding='cp949')
            except UnicodeDecodeError:
                self.logger.warning("cp949 encoding failed, trying euc-kr...")
                df = pd.read_csv(csv_path, encoding='euc-kr')
                
            # Apply preserved Colab template
            for idx, row in df.iterrows():
                # Use EXACT template from existing Colab code
                chunk_text = (
                    f"담당업무: {row['담당업무']}\n"
                    f"  - 담당자: {row['부서']} {row['직책']}\n"
                    f"  - 연락처: {row['전화번호']}\n"
                )
                
                chunk = TextChunk(
                    content=chunk_text,
                    metadata={
                        "source_file": "task_telephone.csv",
                        "file_type": "csv", 
                        "category": "contact",
                        "doc_type": "업무담당자연락처",
                        "domain": "general",
                        "row_index": idx,
                        "department": str(row['부서']),
                        "position": str(row['직책']),
                        "phone": str(row['전화번호']),
                        "task_area": str(row['담당업무'])
                    }
                )
                csv_chunks.append(chunk)
                
            self.logger.info(f"Generated {len(csv_chunks)} contact chunks using preserved template")
            
        except Exception as e:
            self.logger.error(f"Failed to process task_telephone.csv: {e}")
            self.logger.error("Please check file encoding and column names")
            # Continue without failing entire process
            
        return csv_chunks
    
    def _create_hakchik_special_chunks(self, content: str) -> List[str]:
        """
        Special processing for hakchik.pdf (regulations document).
        TODO: Add table extraction logic here if needed.
        """
        # For now, use standard chunking
        # Future enhancement: extract and restructure table data
        from utils.textifier import split_text_into_chunks
        return split_text_into_chunks(content)
    
    def validate_processed_data(self, chunks: List[TextChunk]) -> bool:
        """
        Validate processed general domain data.
        
        Args:
            chunks: List of processed TextChunks
            
        Returns:
            bool: True if validation passes
        """
        if not chunks:
            self.logger.warning("No chunks generated for general domain")
            return False
            
        # Check required categories are present
        categories = set(chunk.metadata.get('category', '') for chunk in chunks)
        self.logger.info(f"Found categories: {categories}")
        
        # Check for contact information chunks
        contact_chunks = [c for c in chunks if c.metadata.get('category') == 'contact']
        if contact_chunks:
            self.logger.info(f"Successfully processed {len(contact_chunks)} contact entries")
        
        # Check for regulation chunks 
        reg_chunks = [c for c in chunks if c.metadata.get('category') == 'regulations']
        if reg_chunks:
            self.logger.info(f"Successfully processed {len(reg_chunks)} regulation chunks")
            
        return True


def main():
    """Main execution function for standalone running."""
    try:
        loader = GeneralLoader()
        success = loader.build_index()
        
        if success:
            print("✅ General domain vectorstore built successfully!")
            return 0
        else:
            print("❌ Failed to build general domain vectorstore")
            return 1
            
    except Exception as e:
        print(f"❌ Fatal error in general loader: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
