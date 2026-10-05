"""Six frozen READ tools plus write metadata; direct write invocation is forbidden."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType

from pydantic import BaseModel, ValidationError
from sqlalchemy import Engine
from sqlalchemy.exc import DBAPIError, DisconnectionError, TimeoutError

from backend.app.data.repositories.analytics import AnalyticsRepository, EntityNotFound, NoData
from backend.app.observability.logger import get_logger
from backend.app.observability.collector import observe
from backend.app.tools.base import SessionFactory, ToolPayload, read_session, readonly_engine, readonly_factory
from backend.app.tools.customer import analyze_customer_performance
from backend.app.tools.orders import query_orders
from backend.app.tools.product import analyze_product_performance
from backend.app.tools.region import analyze_region_performance
from backend.app.tools.sales import get_sales_overview
from backend.app.tools.policy import search_sales_policy
from backend.app.rag.schemas import PolicySearchInput
from backend.app.tools.write_schemas import CRMTaskInput, BusinessAlertInput
from backend.app.tools.schemas import (
    CustomerInput, DateInput, OrdersInput, ProductInput, RegionInput, SalesInput,
    ToolError, ToolMeta, ToolMetadata, ToolResult,
)

logger = get_logger("tools")


@dataclass(frozen=True)
class Definition:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[..., ToolPayload | ToolResult]
    permission_level: str = "READ"


DEFINITIONS = (
    Definition("get_sales_overview", "Use first to measure company or region revenue, profit, margin, orders and customers against the previous equal-length period. Does not identify causes or individual orders.", SalesInput, get_sales_overview),
    Definition("analyze_region_performance", "Compare all regions' sales and profit changes for an inclusive date range to locate weak performance. Returns aggregates ranked by profit change, not order details or causal diagnoses.", RegionInput, analyze_region_performance),
    Definition("analyze_product_performance", "Compare product revenue, profit, units, average discounts and scope-wide revenue shares, optionally within a region or for one product. Supports mix analysis; does not determine policy violations or return orders.", ProductInput, analyze_product_performance),
    Definition("analyze_customer_performance", "Compare customer sales, profit, units and average discounts by inclusive dates, optionally by region or customer ID. Returns customer-level aggregates, not individual transactions or recommended actions.", CustomerInput, analyze_customer_performance),
    Definition("query_orders", "Retrieve up to 50 dated order records using customer, product, region, discount and approval filters. Returns exact match count and truncation; use after aggregate analysis, not to compute company totals.", OrdersInput, query_orders),
    Definition("search_sales_policy", "Search synthetic enterprise sales policy sections about discount approvals, margin governance, special pricing, regional promotions, order review and CRM follow-up. Returns policy citations, not business measurements or proof of order violations.", PolicySearchInput, search_sales_policy),
    Definition("create_crm_task", "Propose one local CRM follow-up for a customer supported by cited business evidence. Requires human approval; cannot be executed through the investigation registry or notify an external CRM.", CRMTaskInput, None, "WRITE_APPROVAL_REQUIRED"),
    Definition("create_business_alert", "Propose one local business risk alert for an evidenced customer, product, region or company. Requires human approval and only inserts into the local demo alert table; sends no external notification.", BusinessAlertInput, None, "WRITE_APPROVAL_REQUIRED"),
)


class ToolRegistry:
    def __init__(self, session_factory: SessionFactory, owned_engine: Engine | None = None, *, raise_unexpected_errors: bool = False, policy_retriever=None):
        self._factory = session_factory
        self._owned_engine = owned_engine
        self._raise_unexpected_errors = raise_unexpected_errors
        self._policy_retriever = policy_retriever
        self._definitions = MappingProxyType({item.name: item for item in DEFINITIONS})

    def list_tools(self, permission=None) -> list[ToolMetadata]:
        return [ToolMetadata(name=item.name, description=item.description, permission_level=item.permission_level,
                             input_schema=item.input_model.model_json_schema())
                for item in self._definitions.values() if permission is None or item.permission_level == permission]

    def get_tool(self, name: str) -> ToolMetadata | None:
        # Return public metadata, never a callable or internal session dependency.
        return next((metadata for metadata in self.list_tools() if metadata.name == name), None)

    def invoke(self, name: str, arguments: dict) -> ToolResult:
        started = perf_counter()

        def failure(code, message) -> ToolResult:
            result = ToolResult(success=False, tool_name=name if isinstance(name, str) else "<invalid>",
                                data=None, summary=None, evidence_ids=[],
                                error=ToolError(code=code, message=message, retryable=code == "DATABASE_ERROR"),
                                meta=ToolMeta(latency_ms=(perf_counter()-started)*1000))
            # Never log untrusted tool names, parameters, exception SQL or results.
            logger.info("tool failure code=%s latency_ms=%.3f", code, result.meta.latency_ms)
            return result

        if not isinstance(name, str) or name not in self._definitions:
            return failure("TOOL_NOT_FOUND", "Tool is not registered")
        definition = self._definitions[name]
        if definition.permission_level != "READ":
            observe("emit", "WRITE_BLOCKED", node="write", tool_name=name, permission=definition.permission_level,
                    error_code="APPROVAL_REQUIRED", approval_existed=False, write_executed=False)
            return failure("APPROVAL_REQUIRED", "Use the server-side human approval boundary; direct writes are forbidden")
        try:
            validated = definition.input_model.model_validate(arguments)
        except ValidationError:
            return failure("INVALID_ARGUMENT", "Arguments do not match the tool input schema or range constraints")
        if name == "search_sales_policy":
            return search_sales_policy(self._policy_retriever, validated)
        try:
            with read_session(self._factory) as session:
                payload = definition.handler(AnalyticsRepository(session), validated)
            result = ToolResult(success=True, tool_name=name, data=payload.data, summary=payload.summary,
                                evidence_ids=payload.evidence_ids, error=None,
                                meta=ToolMeta(latency_ms=(perf_counter()-started)*1000, **payload.meta))
        except EntityNotFound as error:
            return failure("ENTITY_NOT_FOUND", str(error))
        except NoData as error:
            return failure("NO_DATA", str(error))
        except (DBAPIError, DisconnectionError, TimeoutError):
            logger.error("Database operation failed for registered tool=%s", definition.name)
            return failure("DATABASE_ERROR", "Database is unavailable or the query could not be completed")
        except Exception as error:
            # Explicit bug channel, not a retryable business error. Strict test
            # mode re-raises the original exception; public mode never leaks it.
            logger.error("Unexpected tool defect tool=%s type=%s", definition.name, type(error).__name__)
            if self._raise_unexpected_errors:
                raise
            return failure("INTERNAL_ERROR", "An unexpected tool defect occurred")
        logger.info("tool=%s success=true latency_ms=%.3f", definition.name, result.meta.latency_ms)
        return result

    def close(self) -> None:
        if self._owned_engine is not None:
            self._owned_engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def build_default_registry(session_factory: SessionFactory | None = None, *, raise_unexpected_errors: bool = False, policy_retriever=None) -> ToolRegistry:
    if session_factory is not None:
        return ToolRegistry(session_factory, raise_unexpected_errors=raise_unexpected_errors, policy_retriever=policy_retriever)
    engine = readonly_engine()
    return ToolRegistry(readonly_factory(engine), owned_engine=engine, raise_unexpected_errors=raise_unexpected_errors, policy_retriever=policy_retriever)
