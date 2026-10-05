"""Strict flat front matter, trusted corpus root, no YAML or query-controlled paths."""
import hashlib
from pathlib import Path
from backend.app.rag.schemas import PolicyDocument

CORPUS_ROOT = Path(__file__).resolve().parents[3] / "knowledge_base" / "policies"
METADATA = {"doc_id", "title", "version", "effective_date", "department", "policy_type"}


def load_documents(root=CORPUS_ROOT):
    root = Path(root).resolve()
    documents = []
    for path in sorted(root.glob("*.md")):
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("Policy source escapes corpus root")
        source = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        lines = source.splitlines()
        if not lines or lines[0] != "---" or "---" not in lines[1:]:
            raise ValueError("Missing controlled front matter")
        end = lines.index("---", 1)
        metadata = {}
        for line in lines[1:end]:
            key, separator, value = line.partition(":")
            if not separator or key not in METADATA or key in metadata or not value.strip():
                raise ValueError("Invalid or duplicate policy metadata")
            metadata[key] = value.strip()
        if set(metadata) != METADATA:
            raise ValueError("Incomplete policy metadata")
        documents.append(PolicyDocument(**metadata, source_path="knowledge_base/policies/"+path.name,
            content_hash=hashlib.sha256(source.encode()).hexdigest(), body="\n".join(lines[end+1:]).strip()))
    if not documents or len({d.doc_id for d in documents}) != len(documents):
        raise ValueError("Empty corpus or duplicate document IDs")
    return documents
