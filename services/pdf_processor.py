import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_and_chunk(content: bytes, filename: str, doc_id: str) -> list:
    """Extract text from PDF bytes and split into chunks with metadata."""
    doc = fitz.open(stream=content, filetype="pdf")

    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text().strip()
        if text:
            pages.append({"text": text, "page": page_num + 1})

    doc.close()

    if not pages:
        raise ValueError(
            "No extractable text found. This may be a scanned PDF. "
            "Please use a PDF with selectable text."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_index = 0
    for page_data in pages:
        raw_chunks = splitter.split_text(page_data["text"])
        for raw in raw_chunks:
            text = raw.strip()
            if len(text) < 30:
                continue
            chunks.append({
                "id": f"{doc_id}_{chunk_index}",
                "text": text,
                "metadata": {
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page_data["page"],
                    "chunk_index": chunk_index,
                },
            })
            chunk_index += 1

    return chunks
