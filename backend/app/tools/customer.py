from backend.app.data.repositories.analytics import AnalyticsRepository
from backend.app.tools.base import ToolPayload, aggregate_evidence, comparison_dates, period_meta
from backend.app.tools.schemas import CustomerData, CustomerInput


def analyze_customer_performance(repository: AnalyticsRepository, arguments: CustomerInput) -> ToolPayload:
    rows, available = repository.breakdown("customer", *comparison_dates(arguments), region=arguments.region, customer_id=arguments.customer_id)
    return ToolPayload(CustomerData(customers=rows),
                       f"{len(rows)} customers; sorted by absolute profit change ascending, then customer ID.",
                       aggregate_evidence("analyze_customer_performance", arguments), period_meta(arguments, available))
