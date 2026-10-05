from time import perf_counter
from backend.app.rag.schemas import RetrievalError
from backend.app.tools.schemas import ToolResult, ToolError, ToolMeta


def search_sales_policy(retriever, args):
    started = perf_counter()
    try:
        if retriever is None:
            from backend.app.rag.retriever import PolicyRetriever
            retriever = PolicyRetriever()
        data = retriever.search(args.query, args.top_k)
        return ToolResult(success=True, tool_name="search_sales_policy", data=data,
            summary=f"Retrieved {len(data.hits)} policy sections using HYBRID search; policy is not business fact evidence.",
            evidence_ids=[h.citation.policy_evidence_id for h in data.hits], error=None,
            meta=ToolMeta(latency_ms=(perf_counter()-started)*1000, returned_count=len(data.hits)))
    except RetrievalError as error:
        code, retryable = error.code, error.retryable
    except Exception:
        code, retryable = "RETRIEVAL_ERROR", False
    return ToolResult(success=False, tool_name="search_sales_policy", data=None, summary=None, evidence_ids=[],
        error=ToolError(code=code, message="Policy retrieval failed: "+code, retryable=retryable),
        meta=ToolMeta(latency_ms=(perf_counter()-started)*1000))
