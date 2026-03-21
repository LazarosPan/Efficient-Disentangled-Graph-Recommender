# Running Scripts

Use `uv run <script>` for maintenance and utility tasks.

## Preflight

Validate setup before formal runs:
```bash
uv run preflight --dry-run                    # Preview plan
uv run preflight                              # Run representative checks
uv run preflight --reset-sqlite-after         # Clear DB after success
```

## Database Management

Reset experiment database:
```bash
uv run reset-experiment-db
```

Query results:
```bash
uv run query-results --help
```

## Verification

Check environment setup:
```bash
uv run verify-setup
```

Verify pipeline end-to-end:
```bash
uv run verify-pipeline
```

Check SQLite integrity:
```bash
uv run verify-sqlite
```

## Data and Visualization

Download datasets:
```bash
uv run download-pyg-datasets
```

Generate figures:
```bash
uv run visualize-results
```