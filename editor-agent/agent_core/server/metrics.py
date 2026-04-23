from prometheus_client import Counter, Histogram

# Metrics definitions
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total", 
    "Total number of HTTP requests", 
    ["method", "endpoint", "status"]
)

LLM_LATENCY_SECONDS = Histogram(
    "llm_latency_seconds", 
    "Latency of LLM inference requests in seconds",
    ["provider", "model"]
)

WORKSPACE_SCAN_DURATION_SECONDS = Histogram(
    "workspace_scan_duration_seconds", 
    "Duration of workspace scanning in seconds"
)
