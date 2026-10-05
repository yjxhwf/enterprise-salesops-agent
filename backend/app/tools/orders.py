from backend.app.data.repositories.analytics import AnalyticsRepository
from backend.app.tools.base import ToolPayload
from backend.app.tools.schemas import OrdersData, OrdersInput


def query_orders(repository: AnalyticsRepository, arguments: OrdersInput) -> ToolPayload:
    filters = arguments.model_dump(exclude={"start_date", "end_date"})
    rows, total = repository.orders(arguments.start_date, arguments.end_date, **filters)
    return ToolPayload(OrdersData(orders=rows), f"Returned {len(rows)} of {total} matching orders; sorted by date descending, then order ID ascending.",
                       [f"order:{row['order_id']}" for row in rows],
                       {"current_period": arguments.current_period(), "returned_count": len(rows), "total_matches": total, "truncated": len(rows) < total})
