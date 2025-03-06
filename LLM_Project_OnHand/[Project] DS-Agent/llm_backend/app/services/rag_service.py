from typing import Dict, List, Optional
from pathlib import Path
import asyncio
from app.core.config import settings
from .embedding_service import EmbeddingService
import PyPDF2
from docx import Document
import re
import os

class RAGService:
    def __init__(self):
        self.supported_types = {
            # 文本文件
            ".txt": self._process_text,
            ".md": self._process_text,
            # PDF 文件
            ".pdf": self._process_pdf,
            # Word 文件
            ".doc": self._process_word,
            ".docx": self._process_word,
            # 其他类型按需添加
        }
        self.embedding_service = EmbeddingService()
        # 确保必要的目录存在
        self.indexes_dir = Path("indexes")
        self.indexes_dir.mkdir(exist_ok=True)  # 创建 indexes 目录
        self.uploads_dir = Path("uploads")
        self.uploads_dir.mkdir(exist_ok=True)  # 创建 uploads 目录
    
    def _split_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将文本分割成适合的块"""
        # 简单的按句子分割
        sentences = re.split(r'[。！？.!?]', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += sentence + "。"
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    async def process_file(self, file_info: Dict) -> Dict:
        """处理上传的文件"""
        try:
            # 处理文件并创建向量索引
            result = await self.embedding_service.create_embeddings(
                file_info["path"],
                str(self.indexes_dir)  # 传入 indexes 目录路径
            )
            
            return {
                "status": "success",
                "index_id": result["index_id"],
                "chunks": result["chunks"]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"创建向量失败: {str(e)}"
            }
    
    async def _process_text(self, file_path: Path) -> List[str]:
        """处理文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return [text]
        except Exception as e:
            raise Exception(f"处理文本文件失败: {str(e)}")
    
    async def _process_pdf(self, file_path: Path) -> List[str]:
        """处理 PDF 文件"""
        try:
            text_content = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text_content.append(page.extract_text())
            return text_content
        except Exception as e:
            raise Exception(f"处理 PDF 文件失败: {str(e)}")
    
    async def _process_word(self, file_path: Path) -> List[str]:
        """处理 Word 文件"""
        try:
            doc = Document(file_path)
            text_content = []
            for para in doc.paragraphs:
                if para.text.strip():
                    text_content.append(para.text)
            return text_content
        except Exception as e:
            raise Exception(f"处理 Word 文件失败: {str(e)}") 