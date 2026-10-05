RRF_K = 60
HYBRID_TOP_K = 5


def reciprocal_rank_fusion(vector, lexical, top_k=HYBRID_TOP_K):
    scores, best = {}, {}
    for results in (vector, lexical):
        seen = set()
        for result in results:
            cid, rank = result["chunk_id"], result["rank"]
            if cid in seen:
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0)+1/(RRF_K+rank)
            best[cid] = min(best.get(cid, rank), rank)
    return [dict(chunk_id=cid, rrf_score=scores[cid]) for cid in
            sorted(scores, key=lambda cid: (-scores[cid], best[cid], cid))[:top_k]]
