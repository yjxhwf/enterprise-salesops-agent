from backend.app.data.repositories.analytics import AnalyticsRepository
from backend.app.tools.base import ToolPayload, aggregate_evidence, comparison_dates, period_meta
from backend.app.tools.schemas import ProductData, ProductInput


def analyze_product_performance(repository: AnalyticsRepository, arguments: ProductInput) -> ToolPayload:
    rows, available = repository.breakdown("product", *comparison_dates(arguments), region=arguments.region, product_id=arguments.product_id)
    return ToolPayload(ProductData(products=rows),
                       f"{len(rows)} products; revenue shares use all products in the date/region scope. Sorted by profit change ascending, then product ID.",
                       aggregate_evidence("analyze_product_performance", arguments), period_meta(arguments, available))
