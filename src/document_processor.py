"""Load PDF/TXT files and split them into chunks (FR1–FR3, FR6)."""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.config import ConfigError


class DocumentProcessor:
    """Turns the files in `documents/` into small chunks the retriever can search."""

    def __init__(
        self,
        data_directory: str = config.DATA_DIRECTORY,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ) -> None:
        self.data_directory = Path(data_directory)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_documents(self) -> List[Document]:
        """Load every PDF (one Document per page) and TXT file, including subfolders."""
        if not self.data_directory.is_dir():
            raise ConfigError(
                f"Documents folder not found: {self.data_directory}. "
                "Create it and put PDF or TXT files in it."
            )

        pdf_files = sorted(self.data_directory.rglob("*.pdf"))
        txt_files = sorted(self.data_directory.rglob("*.txt"))
        if not pdf_files and not txt_files:
            raise ConfigError(
                f"No PDF or TXT files found in {self.data_directory}. "
                "Add at least one file and run again."
            )

        documents: List[Document] = []
        for path in pdf_files:
            pages = PyPDFLoader(str(path)).load()
            # Scanned PDFs have no text layer -> warn instead of silently indexing nothing
            if not any(p.page_content.strip() for p in pages):
                print(f"⚠️  Warning: no text found in {path.name} (probably a scanned PDF).")
            documents.extend(pages)
        for path in txt_files:
            documents.extend(TextLoader(str(path), autodetect_encoding=True).load())

        print(f"📄 Loaded {len(pdf_files)} PDF and {len(txt_files)} TXT file(s) "
              f"({len(documents)} documents/pages).")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split into overlapping chunks; metadata (source, page) is copied to each chunk."""
        chunks = [c for c in self.splitter.split_documents(documents) if c.page_content.strip()]
        print(f"✂️  Created {len(chunks)} chunks.")
        return chunks
