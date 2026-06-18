import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """You are PaperMind, an intelligent document research assistant.

Rules:
- Answer ONLY using the provided document excerpts below.
- If the answer is not in the excerpts, say exactly: "I couldn't find this in the uploaded documents."
- Be concise and factual.
- Always end your answer with a citation in this format: [Source: filename, Page X]
- If multiple sources are used, cite each one."""


def get_answer(question: str, chunks: list) -> tuple[str, list]:
    if not chunks:
        return "No relevant content found in your documents.", []

    # Build context
    context_blocks = []
    sources = []
    seen = set()

    for chunk in chunks:
        meta = chunk["metadata"]
        context_blocks.append(
            f"[From: {meta['filename']}, Page {meta['page']}]\n{chunk['text']}"
        )
        key = f"{meta['filename']}|{meta['page']}"
        if key not in seen:
            sources.append({
                "filename": meta["filename"],
                "page": meta["page"],
                "doc_id": meta["doc_id"],
            })
            seen.add(key)

    context = "\n\n---\n\n".join(context_blocks)

    response = _client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Document excerpts:\n\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()
    return answer, sources
