import re
from rank_bm25 import BM25Okapi

BM25_TOP_K = 8


def tokenize(text):
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(run)
        tokens.extend(run[i:i+2] for i in range(len(run)-1))
    return tokens


class LexicalIndex:
    def __init__(self, chunks):
        self.ids = [c.chunk_id for c in chunks]
        self.index = BM25Okapi([tokenize(c.doc_id+" "+c.text) for c in chunks])

    def search(self, query, top_k=BM25_TOP_K):
        pairs = zip(self.ids, self.index.get_scores(tokenize(query)))
        ranked = sorted(((cid, float(score)) for cid, score in pairs if score > 0), key=lambda x: (-x[1], x[0]))[:top_k]
        return [dict(chunk_id=cid, score=score, rank=i) for i, (cid, score) in enumerate(ranked, 1)]
