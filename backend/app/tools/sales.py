from backend.app.data.repositories.analytics import AnalyticsRepository
from backend.app.tools.base import ToolPayload, aggregate_evidence, comparison_dates, period_meta
from backend.app.tools.schemas import OverviewData, SalesInput


def get_sales_overview(repository: AnalyticsRepository, arguments: SalesInput) -> ToolPayload:
    data = OverviewData.model_validate(repository.overview(*comparison_dates(arguments), region=arguments.region))
    current = data.current
    margin = f"{current.margin * 100:.2f}%" if current.margin is not None else "unavailable"
    summary = (f"{arguments.start_date} to {arguments.end_date}: revenue {current.revenue:.2f}, "
               f"profit {current.profit:.2f}, margin {margin}, orders {current.order_count}.")
    return ToolPayload(data, summary, aggregate_evidence("get_sales_overview", arguments),
                       period_meta(arguments, data.comparison_available))
