import hashlib
import re
from backend.app.rag.schemas import PolicyChunk

CHUNKING_VERSION = "section-v1"
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150


def chunk_documents(documents):
    chunks = []
    for doc in documents:
        sections = []
        heading, body = doc.title, []
        for line in doc.body.splitlines():
            match = re.match(r"^#{1,6}\s+(.+)$", line)
            if match:
                if "\n".join(body).strip():
                    sections.append((heading, "\n".join(body).strip()))
                heading, body = match[1], []
            else:
                body.append(line)
        if "\n".join(body).strip():
            sections.append((heading, "\n".join(body).strip()))
        for section_number, (heading, body) in enumerate(sections, 1):
            prefix = f"{doc.title}\n{heading}\n"
            room = MAX_CHUNK_CHARS-len(prefix)
            if room <= CHUNK_OVERLAP_CHARS:
                raise ValueError("Policy heading is too long")
            start, part = 0, 0
            while start < len(body):
                end = min(start+room, len(body))
                if end < len(body):
                    paragraph = body.rfind("\n\n", start+room//2, end)
                    if paragraph != -1:
                        end = paragraph
                text = prefix+body[start:end].strip()
                digest = hashlib.sha256(text.encode()).hexdigest()
                section_id = f"s{section_number:03}"
                chunks.append(PolicyChunk(**doc.model_dump(exclude={"body", "content_hash"}),
                    chunk_id=f"{doc.doc_id}:{section_id}:{part}:{digest[:20]}", section_id=section_id,
                    section_title=heading, chunk_index=part, text=text, content_hash=digest))
                if end == len(body):
                    break
                start, part = end-CHUNK_OVERLAP_CHARS, part+1
    if not chunks:
        raise ValueError("Corpus has no nonempty sections")
    return chunks
