from backend.app.data.repositories.analytics import AnalyticsRepository
from backend.app.tools.base import ToolPayload, aggregate_evidence, comparison_dates, period_meta
from backend.app.tools.schemas import RegionData, RegionInput


def analyze_region_performance(repository: AnalyticsRepository, arguments: RegionInput) -> ToolPayload:
    rows, available = repository.breakdown("region", *comparison_dates(arguments))
    return ToolPayload(RegionData(regions=rows),
                       f"{len(rows)} regions; sorted by absolute profit change ascending, then region code.",
                       aggregate_evidence("analyze_region_performance", arguments), period_meta(arguments, available))
