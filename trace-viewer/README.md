# SkyDiscover Evolution Viewer

Analyst-facing web viewer for one SkyDiscover `evolution_trace.json`. It shows:

- candidate and best-so-far performance by iteration;
- successful and failed LLM calls, including provider reasoning rounds;
- Tavily request configuration, timing, complete provider responses, and errors;
- the compact title/URL/content payload returned to the LLM;
- prompts, final responses, program lineage, and unattached calls that did not
  produce a Program.

The viewer uses `evolution_trace.json` as its only run-data source. The exporter
reads all of these locations:

```text
programs[].llm_reasoning.calls[]
programs[].llm_reasoning_content
programs[].web_search_results.tavily_searches[]
unattached_llm_reasoning.calls[]
unattached_web_search_results.tavily_searches[]
```

Raw reasoning response records and raw Tavily provider responses are preserved;
the viewer derives compact display text without replacing those source objects.
When an older canonical trace contains only `llm_tool_result`, the result cards
show that stored model-facing payload and label the raw provider response as
unavailable. The viewer never issues a replacement search.

## Export a run

From the SkyDiscover repository root:

```bash
uv run python scripts/export_evolution_viewer.py \
  outputs/path/to/run \
  --output trace-viewer/data/trace.json
```

The selected run directory must contain a run-level `evolution_trace.json`.
If it does not, resume or rerun SkyDiscover first so the trace is written.

## Run locally

Pass either a run directory or one of its `checkpoint_*` paths:

```bash
scripts/run_trace_viewer_local.sh outputs/path/to/run
```

The script normalizes a checkpoint path to its run directory, refreshes the
viewer data, builds the app, and starts it at `http://127.0.0.1:4173`.

Useful environment variables:

```text
PORT=4173
BIND_HOST=127.0.0.1
REFRESH_TRACE=true
DEV_MODE=false
SKIP_BUILD=false
```

Set `REFRESH_TRACE=false` to keep the currently exported viewer JSON. For a
remote machine, keep the loopback default and use SSH port forwarding, or set
`BIND_HOST=0.0.0.0` only on a trusted network.
