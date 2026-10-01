"use client";

import { useEffect, useMemo, useState } from "react";
import { compactId, duration, latency, score } from "@/lib/format";
import type {
  AttemptTrace,
  IterationTrace,
  SearchResult,
  SearchTrace,
} from "@/lib/types";
import type { Selection } from "./trace-timeline";

type Props = {
  selection: Selection;
  iterations: IterationTrace[];
  attempts: AttemptTrace[];
  searches: SearchTrace[];
  onSearchSelect: (id: string) => void;
};

type IterationTab = "rationale" | "response" | "prompt";

export function TraceInspector({
  selection,
  iterations,
  attempts,
  searches,
  onSearchSelect,
}: Props) {
  const [tab, setTab] = useState<IterationTab>("rationale");
  const [selectedAttemptId, setSelectedAttemptId] = useState<string | null>(null);

  useEffect(() => setTab("rationale"), [selection]);
  useEffect(() => {
    if (selection.kind !== "iteration") return;
    const related = attempts.filter(
      (item) => item.iteration === selection.iteration,
    );
    setSelectedAttemptId(related.at(-1)?.id ?? null);
  }, [selection, attempts]);

  const iterationMap = useMemo(
    () => new Map(iterations.map((item) => [item.iteration, item])),
    [iterations],
  );
  const attemptMap = useMemo(
    () => new Map(attempts.map((item) => [item.id, item])),
    [attempts],
  );
  const searchMap = useMemo(
    () => new Map(searches.map((item) => [item.id, item])),
    [searches],
  );

  if (selection.kind === "search") {
    const searchItem = searchMap.get(selection.id);
    const iteration =
      searchItem?.iteration == null
        ? undefined
        : iterationMap.get(searchItem.iteration);
    const attempt = searchItem?.attempt_id
      ? attemptMap.get(searchItem.attempt_id)
      : undefined;
    return searchItem ? (
      <SearchInspector
        searchItem={searchItem}
        attempt={attempt}
        iteration={iteration}
      />
    ) : null;
  }

  const iteration = iterationMap.get(selection.iteration);
  if (!iteration) return null;

  const relatedAttempts = iteration.attempt_ids
    .map((id) => attemptMap.get(id))
    .filter((item): item is AttemptTrace => Boolean(item));
  const activeAttempt =
    relatedAttempts.find((item) => item.id === selectedAttemptId) ??
    relatedAttempts.at(-1);
  const relatedSearches = (activeAttempt?.search_ids ?? iteration.search_ids)
    .map((id) => searchMap.get(id))
    .filter((item): item is SearchTrace => Boolean(item));
  const rationale =
    activeAttempt?.reasoning ||
    activeAttempt?.visible_rationale ||
    iteration.reasoning ||
    iteration.visible_rationale;
  const reasoningCapture =
    activeAttempt?.reasoning_capture ?? iteration.reasoning_capture;
  const hasProviderReasoning = reasoningCapture === "provider_reasoning";
  const response = activeAttempt?.response || iteration.llm_response;
  const prompt = activeAttempt?.prompt || iteration.prompt;
  const isFailed = iteration.status === "failed";
  const isBaseline = iteration.status === "baseline";

  return (
    <section className="inspector-panel">
      <div className="inspector-head">
        <div>
          <span className="inspector-type">
            {isBaseline ? "Baseline" : `Iteration ${iteration.iteration}`}
          </span>
          <h2>
            {isFailed
              ? "No candidate accepted"
              : iteration.is_new_best && !isBaseline
                ? "New performance frontier"
                : "Candidate inspection"}
          </h2>
        </div>
        <span
          className={`outcome-badge ${
            isFailed
              ? "error"
              : iteration.is_new_best
                ? "improved"
                : "exploration"
          }`}
        >
          {isFailed
            ? "failed"
            : iteration.is_new_best
              ? "best-so-far"
              : iteration.status}
        </span>
      </div>

      <div className="inspector-kpis">
        <Metric
          label="Candidate"
          value={iteration.score == null ? "not accepted" : score(iteration.score)}
          tone={isFailed ? "negative" : undefined}
        />
        <Metric label="Retained best" value={score(iteration.best_so_far)} accent />
        <Metric label="LLM attempts" value={String(relatedAttempts.length || "—")} />
        <Metric label="Wall time" value={duration(iteration.duration_seconds)} />
      </div>

      <div className="provenance-strip">
        <span>
          Program <code>{compactId(iteration.program_id)}</code>
        </span>
        <span>
          Retained <code>{compactId(iteration.retained_program_id)}</code>
        </span>
        <span>LLM {duration(iteration.llm_seconds)}</span>
        <span>{iteration.search_ids.length} Tavily calls</span>
      </div>

      {isFailed ? (
        <div className="failure-summary">
          <span>{failureLabel(iteration.failure_kind)}</span>
          <div>
            <strong>Iteration {iteration.iteration} exhausted all retries</strong>
            <p>{iteration.failure_reason || "The iteration failed without a message."}</p>
          </div>
        </div>
      ) : null}

      {iteration.changes ? (
        <div className="change-summary">
          <span>CHANGE SUMMARY</span>
          <p>{iteration.changes}</p>
        </div>
      ) : null}

      {relatedAttempts.length > 0 ? (
        <div className="attempt-section">
          <div className="subsection-title">
            <span>RETRY ATTEMPTS</span>
            <b>{relatedAttempts.length} total</b>
          </div>
          <div className="attempt-switcher">
            {relatedAttempts.map((attempt) => (
              <button
                className={activeAttempt?.id === attempt.id ? "active" : ""}
                key={attempt.id}
                onClick={() => setSelectedAttemptId(attempt.id)}
              >
                <span>ATTEMPT {attempt.attempt}</span>
                <strong>{failureLabel(attempt.failure_kind)}</strong>
                <small>
                  {latency(attempt.duration_ms)} · web ×{attempt.tavily_searches} ·{" "}
                  {attempt.returned_results} results
                </small>
              </button>
            ))}
          </div>
        </div>
      ) : null}

      {relatedSearches.length > 0 ? (
        <div className="related-searches">
          <div className="subsection-title">
            <span>
              WEB EVIDENCE
              {activeAttempt ? ` · ATTEMPT ${activeAttempt.attempt}` : ""}
            </span>
            <b>{relatedSearches.length} calls</b>
          </div>
          <div className="search-chip-list">
            {relatedSearches.map((searchItem, index) => (
              <button key={searchItem.id} onClick={() => onSearchSelect(searchItem.id)}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{searchItem.query}</strong>
                <small>
                  round {(searchItem.round ?? 0) + 1} ·{" "}
                  {searchItem.result_count ?? 0} returned
                </small>
              </button>
            ))}
          </div>
        </div>
      ) : !isBaseline ? (
        <div className="no-search-note">
          <span>No web call in this attempt</span>
          Select another retry to inspect its retrieval activity.
        </div>
      ) : null}

      <div className="inspector-tabs" role="tablist">
        {(
          [
            ["rationale", "Reasoning"],
            ["response", "Final response"],
            ["prompt", "Prompt"],
          ] as const
        ).map(([value, label]) => (
          <button
            role="tab"
            aria-selected={tab === value}
            className={tab === value ? "active" : ""}
            key={value}
            onClick={() => setTab(value)}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="inspector-content">
        {tab === "rationale" ? (
          <>
            <div className="content-label">
              <span>
                {hasProviderReasoning
                  ? "CAPTURED MODEL REASONING"
                  : reasoningCapture === "visible_response_rationale_only"
                    ? "VISIBLE RESPONSE RATIONALE"
                    : "REASONING UNAVAILABLE"}
              </span>
              <b
                className={
                  hasProviderReasoning ? "capture-complete" : "capture-partial"
                }
              >
                {hasProviderReasoning
                  ? "provider"
                  : reasoningCapture === "visible_response_rationale_only"
                    ? "fallback"
                    : "unavailable"}
              </b>
            </div>
            {rationale ? (
              <div className="prose-trace">{rationale}</div>
            ) : (
              <EmptyContent message="No reasoning was captured for this selection." />
            )}
            {activeAttempt?.responses.length ? (
              <details className="response-details">
                <summary>
                  Raw provider response rounds · {activeAttempt.responses.length}
                </summary>
                <pre className="trace-code">
                  {JSON.stringify(activeAttempt.responses, null, 2)}
                </pre>
              </details>
            ) : null}
            {activeAttempt ? (
              <details className="response-details">
                <summary>Raw LLM call, request &amp; tool executions</summary>
                <pre className="trace-code">
                  {JSON.stringify(activeAttempt.raw_call, null, 2)}
                </pre>
              </details>
            ) : null}
          </>
        ) : null}

        {tab === "response" ? (
          response ? (
            <pre className="trace-code">{response}</pre>
          ) : (
            <EmptyContent message="The baseline has no LLM response." />
          )
        ) : null}

        {tab === "prompt" ? (
          prompt.system || prompt.user ? (
            <div className="prompt-stack">
              <details>
                <summary>System prompt</summary>
                <pre className="trace-code">{prompt.system}</pre>
              </details>
              <details open>
                <summary>User prompt</summary>
                <pre className="trace-code">{prompt.user}</pre>
              </details>
            </div>
          ) : (
            <EmptyContent message="The baseline has no generation prompt." />
          )
        ) : null}
      </div>
    </section>
  );
}

function SearchInspector({
  searchItem,
  attempt,
  iteration,
}: {
  searchItem: SearchTrace;
  attempt?: AttemptTrace;
  iteration?: IterationTrace;
}) {
  const llmToolPayload =
    searchItem.llm_tool_result &&
    typeof searchItem.llm_tool_result === "object" &&
    !Array.isArray(searchItem.llm_tool_result)
      ? searchItem.llm_tool_result
      : null;
  const displayPayload = searchItem.result ?? llmToolPayload;
  const results = Array.isArray(displayPayload?.results)
    ? (displayPayload.results as SearchResult[])
    : [];
  const answer =
    typeof displayPayload?.answer === "string" ? displayPayload.answer : null;
  const configured = searchItem.settings.configured_max_results;
  const requested = searchItem.settings.requested_max_results;
  const effective = searchItem.settings.effective_max_results;
  const rawResultError = displayPayload?.error ?? searchItem.error;
  const resultError =
    typeof rawResultError === "string"
      ? rawResultError
      : rawResultError == null
        ? null
        : JSON.stringify(rawResultError);
  const rationale = attempt?.reasoning || attempt?.visible_rationale;

  return (
    <section className="inspector-panel search-inspector">
      <div className="inspector-head">
        <div>
          <span className="inspector-type">
            Tavily · iteration {searchItem.iteration ?? "—"} · attempt{" "}
            {searchItem.attempt ?? "—"} · round {(searchItem.round ?? 0) + 1}
          </span>
          <h2>Stored retrieval payload</h2>
        </div>
        <span className={`outcome-badge ${searchItem.status}`}>
          {searchItem.status}
        </span>
      </div>

      <div className="query-card">
        <span>QUERY SENT TO TAVILY</span>
        <p>{searchItem.query}</p>
      </div>

      <div className="inspector-kpis search-kpis">
        <Metric label="Configured max" value={String(configured ?? "—")} />
        <Metric label="Model requested" value={String(requested ?? "default")} />
        <Metric label="Effective max" value={String(effective ?? "—")} accent />
        <Metric label="Returned" value={String(searchItem.result_count ?? 0)} />
        <Metric label="Search depth" value={searchItem.settings.search_depth ?? "—"} />
        <Metric label="Latency" value={latency(searchItem.duration_ms)} />
      </div>

      <div className="retrieval-explainer">
        <span className="formula">
          effective max = clamp(model request ?? config default, 0, 20)
        </span>
        <p>
          The cards below come from this call&apos;s stored result object. No fresh
          search is performed when you open the viewer.
        </p>
      </div>

      <div className="payload-provenance original">
        <span>EVOLUTION TRACE</span>
        <div>
          <strong>
            {searchItem.result
              ? "Exact Tavily provider response returned during this run."
              : llmToolPayload
                ? "Stored LLM-facing Tavily result; raw provider response unavailable."
                : "The Tavily call metadata and failure state were captured during this run."}
          </strong>
          <p>
            Stored in <code>evolution_trace.json</code> and linked by LLM call ID to
            iteration {searchItem.iteration ?? "—"}, attempt {searchItem.attempt ?? "—"}.
            {llmToolPayload && !searchItem.result
              ? " These result cards reproduce what the model received; they are not labeled as a raw provider response."
              : ""}
          </p>
        </div>
        <code>{latency(searchItem.duration_ms)}</code>
      </div>

      {answer ? (
        <div className="answer-card">
          <span>TAVILY ANSWER</span>
          <p>{answer}</p>
        </div>
      ) : null}

      <div className="subsection-title results-heading">
        <span>SEARCH RESULTS</span>
        <b>{results.length} items</b>
      </div>

      {resultError ? (
        <div className="search-error">
          <div className="missing-icon" aria-hidden="true">
            !
          </div>
          <div>
            <strong>Tavily call failed</strong>
            <p>{resultError || "No provider error message was captured."}</p>
          </div>
        </div>
      ) : results.length > 0 ? (
        <div className="result-list">
          {results.map((result, index) => (
            <article className="result-card" key={`${result.url}-${index}`}>
              <div className="result-index">{String(index + 1).padStart(2, "0")}</div>
              <div>
                <div className="result-meta">
                  {result.score != null ? (
                    <span>relevance {result.score.toFixed(3)}</span>
                  ) : null}
                  {result.published_date ? <span>{result.published_date}</span> : null}
                </div>
                <h3>{result.title || "Untitled result"}</h3>
                {result.content ? <p>{result.content}</p> : null}
                {result.url ? (
                  <a href={result.url} target="_blank" rel="noreferrer">
                    {result.url}
                    <span aria-hidden="true">↗</span>
                  </a>
                ) : null}
              </div>
            </article>
          ))}
        </div>
      ) : displayPayload !== null ? (
        <EmptyContent message="Tavily completed successfully but returned no result items." />
      ) : (
        <div className="payload-missing">
          <div className="missing-icon" aria-hidden="true">
            !
          </div>
          <div>
            <strong>No Tavily provider response</strong>
            <p>
              The call record is present, but the provider did not return a response
              payload.
            </p>
          </div>
        </div>
      )}

      <details className="raw-details">
        <summary>Raw tool arguments &amp; settings</summary>
        <pre className="trace-code">
          {JSON.stringify(
            {
              arguments: searchItem.arguments,
              request: searchItem.request,
              settings: searchItem.settings,
              tool_call_id: searchItem.tool_call_id,
              llm_call_id: searchItem.call_id,
              llm_tool_result: searchItem.llm_tool_result,
              error: searchItem.error,
              program_id: searchItem.program_id,
              source_program_id: searchItem.source_program_id,
              association: searchItem.association,
              capture_status: searchItem.capture_status,
              result_provenance: searchItem.result_provenance,
            },
            null,
            2,
          )}
        </pre>
      </details>

      <details className="raw-details">
        <summary>Raw Tavily provider response JSON</summary>
        <pre className="trace-code">{JSON.stringify(searchItem.result, null, 2)}</pre>
      </details>

      {searchItem.llm_tool_result != null ? (
        <details className="raw-details">
          <summary>LLM-facing Tavily tool result JSON</summary>
          <pre className="trace-code">
            {JSON.stringify(searchItem.llm_tool_result, null, 2)}
          </pre>
        </details>
      ) : null}

      <details className="raw-details">
        <summary>Complete stored Tavily call record</summary>
        <pre className="trace-code">{JSON.stringify(searchItem.raw_record, null, 2)}</pre>
      </details>

      <div className="linked-response">
        <div className="subsection-title">
          <span>LLM ATTEMPT LINKED TO THIS SEARCH</span>
          <b>
            {attempt
              ? `iteration ${attempt.iteration} · attempt ${attempt.attempt}`
              : "not available"}
          </b>
        </div>
        {attempt ? (
          <>
            <div className={`linked-outcome ${attempt.status}`}>
              <span>{failureLabel(attempt.failure_kind)}</span>
              <p>
                {attempt.failure_reason ||
                  "The attempt completed and produced a candidate response."}
              </p>
            </div>
            {rationale ? (
              <details className="response-details">
                <summary>
                  {attempt.reasoning_capture === "provider_reasoning"
                    ? "Provider reasoning"
                    : "Visible response rationale"}{" "}
                  · {rationale.length} characters
                </summary>
                <div className="prose-trace">{rationale}</div>
              </details>
            ) : null}
            {attempt.response ? (
              <details className="response-details" open>
                <summary>
                  Final model response · {attempt.response.length} characters
                </summary>
                <pre className="trace-code">{attempt.response}</pre>
              </details>
            ) : (
              <EmptyContent message="This LLM call has no final response content." />
            )}
            {attempt.responses.length ? (
              <details className="response-details">
                <summary>Raw provider response rounds · {attempt.responses.length}</summary>
                <pre className="trace-code">
                  {JSON.stringify(attempt.responses, null, 2)}
                </pre>
              </details>
            ) : null}
            <details className="response-details">
              <summary>Raw LLM call, request &amp; tool executions</summary>
              <pre className="trace-code">
                {JSON.stringify(attempt.raw_call, null, 2)}
              </pre>
            </details>
          </>
        ) : (
          <EmptyContent
            message={
              iteration?.llm_response
                ? "A response exists at iteration level but could not be linked to this call."
                : "No LLM response could be linked to this search."
            }
          />
        )}
      </div>
    </section>
  );
}

function failureLabel(kind?: string | null) {
  if (kind === "parse_validation") return "invalid diff";
  if (kind === "evaluation") return "evaluation failed";
  if (kind === "tool") return "tool failed";
  return kind ? kind.replaceAll("_", " ") : "completed";
}

function Metric({
  label,
  value,
  accent = false,
  tone,
}: {
  label: string;
  value: string;
  accent?: boolean;
  tone?: "positive" | "negative";
}) {
  return (
    <div className={`mini-metric ${accent ? "accent" : ""} ${tone ?? ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EmptyContent({ message }: { message: string }) {
  return <div className="empty-content">{message}</div>;
}
