from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
from PyPDF2 import PdfReader
from config.settings import settings

def process_data():
    if not os.path.exists(settings.PDF_FOLDER):
        raise FileNotFoundError(f"PDF folder not found: {settings.PDF_FOLDER}")
    loader = PyPDFDirectoryLoader(settings.PDF_FOLDER)
    docs = loader.load()

    processed_docs = []
    for file in os.listdir(settings.PDF_FOLDER):
        if file.lower().endswith('.pdf'):
            pdf_path = os.path.join(settings.PDF_FOLDER, file)
            for doc in docs:
                if doc.metadata.get("source") == pdf_path:
                    page_content = doc.page_content or ""
                    page_num = doc.metadata.get("page", 1)
                    if page_content.strip():
                        processed_docs.append(Document(page_content=page_content, metadata={"source": pdf_path, "page": page_num}))
                    else:
                        pdf_reader = PdfReader(pdf_path)
                        for page_num in range(len(pdf_reader.pages)):
                            py_pdf_text = pdf_reader.pages[page_num].extract_text() or ""
                            if py_pdf_text.strip():
                                processed_docs.append(Document(page_content=py_pdf_text, metadata={"source": pdf_path, "page": page_num + 1}))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_documents(processed_docs)

# Process data on import
chunks = process_data()