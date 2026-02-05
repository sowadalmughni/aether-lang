# Telemetry Configuration

This runtime uses **OpenTelemetry 0.22** with the **OTLP** exporter.
The deprecated `opentelemetry-jaeger` crate has been removed.

## Supported Stack Versions
We strictly pin these versions to avoid trait bound errors (`E0277`):

* `opentelemetry` = **0.22.0**
* `opentelemetry_sdk` = **0.22.1**
* `tracing-opentelemetry` = **0.23.0**
* `opentelemetry-otlp` = **0.15.0**

## OTLP Configuration
To enable OTLP tracing, set endpoint environment variables (defaulting to `http://localhost:4317`):

```bash
# Standard OTLP variables (supported by most recent SDKs)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Legacy Jaeger-specific env vars (mapped to OTLP endpoint)
export JAEGER_COLLECTOR_ENDPOINT="http://localhost:4317"
```

## Why not use `opentelemetry-jaeger`?
The `opentelemetry-jaeger` crate is deprecated and incompatible with newer Tokio/Tor runtimes. We use `opentelemetry-otlp` which can export to Jaeger, Zipkin, or any OTLP-compliant collector.

## Troublehsooting
If you see build errors like `trait bound OpenTelemetryLayer ... is not satisfied` or `method init exists ... but its trait bounds were not satisfied`:

1. Check for duplicate versions:
   ```bash
   cargo tree -d | grep -E "opentelemetry|tracing-opentelemetry"
   ```
   You should only see ONE version for each.

2. Ensure `flags` match. `opentelemetry` 0.22 requires `trace` feature.

3. Verify no transitive dependencies are pulling in `opentelemetry` 0.21 or 0.23.
