"use client";

import { useMemo, useState } from "react";
import rawTrace from "@/data/trace.json";
import { PerformanceChart } from "@/components/performance-chart";
import {
  TraceTimeline,
  type Selection,
} from "@/components/trace-timeline";
import { TraceInspector } from "@/components/trace-inspector";
import { duration, score } from "@/lib/format";
import type { TraceData } from "@/lib/types";

const trace = rawTrace as TraceData;

export default function Page() {
  const runFailed = trace.run.status === "failed";
  const runHasFailures = trace.summary.failed_iterations > 0;
  const providerPayloads = trace.searches.filter(
    (item) => item.result !== null,
  ).length;
  const llmFacingOnlyPayloads = trace.searches.filter(
    (item) => item.result === null && item.llm_tool_result != null,
  ).length;
  const defaultIteration =
    trace.summary.best_iteration ??
    trace.iterations.at(-1)?.iteration ??
    trace.iterations[0]?.iteration ??
    0;
  const [selection, setSelection] = useState<Selection>(
    trace.searches[0]
      ? { kind: "search", id: trace.searches[0].id }
      : { kind: "iteration", iteration: defaultIteration },
  );

  const selectedSearchId =
    selection.kind === "search" ? selection.id : undefined;
  const selectedIteration =
    selection.kind === "iteration" ? selection.iteration : undefined;
  const searchIterations = useMemo(
    () => new Set(trace.searches.map((item) => item.iteration)),
    [],
  );

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="SkyDiscover Trace Viewer home">
          <span className="brand-mark" aria-hidden="true">
            SD
          </span>
          <span>
            <b>SkyDiscover</b>
            <small>Trace viewer</small>
          </span>
        </a>
        <div className="topbar-meta">
          <span className={`live-dot ${runFailed ? "failed" : ""}`} />
          <span>Local trace · run {trace.run.status}</span>
          <i />
          <code>{trace.run.model}</code>
        </div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <div className="hero-eyebrow">
            <span>EXPERIMENT TRACE</span>
            <i />
            <span>{trace.run.provider}</span>
          </div>
          <h1>
            See what the model searched.
            <br />
            <em>{runFailed ? "Find why evolution failed." : "Read every response."}</em>
          </h1>
          <p>
            Every query, stored Tavily payload, retry, model response and run outcome
            connected in one trace for the 26-circle packing experiment.
          </p>
          <div className="hero-tags">
            <span>circle packing · n=26</span>
            <span>{trace.run.tool_config.search_depth ?? "web"} search</span>
            <span>tool choice · {trace.run.tool_config.tool_choice ?? "—"}</span>
            <span>{trace.run.iterations} iterations</span>
            <span className={runFailed ? "failed-tag" : ""}>
              {trace.run.status}
            </span>
          </div>
        </div>
        <div className="hero-score">
          <span>BEST COMBINED SCORE</span>
          <strong>{score(trace.summary.best_score, 3)}</strong>
          <div>
            <b>
              {(trace.summary.relative_lift_percent ?? 0) >= 0 ? "+" : ""}
              {(trace.summary.relative_lift_percent ?? 0).toFixed(1)}%
            </b>
            <span>from baseline</span>
          </div>
          <p>
            {trace.summary.best_iteration === 0
              ? "Baseline remained the best program"
              : `Reached at iteration ${trace.summary.best_iteration}`}
          </p>
        </div>
      </section>

      <section className="workspace">
        <div className="kpi-grid">
          <Kpi
            label={runFailed ? "Failed iterations" : "Accepted candidates"}
            value={
              runFailed
                ? `${trace.summary.failed_iterations} / ${trace.summary.attempted_iterations}`
                : String(trace.summary.accepted_candidates)
            }
            note={
              runFailed
                ? "no evolved candidate accepted"
                : runHasFailures
                  ? `${trace.summary.failed_iterations} failed iterations · beyond baseline`
                  : "beyond baseline"
            }
            tone={runFailed ? "red" : "mint"}
          />
          <Kpi
            label="LLM attempts"
            value={String(trace.summary.llm_attempts)}
            note={`${trace.summary.parse_failures} diff · ${trace.summary.evaluation_failures} evaluator failures`}
          />
          <Kpi
            label="Web searches"
            value={String(trace.summary.web_searches)}
            note={`${trace.summary.successful_web_searches} succeeded · ${trace.summary.failed_web_searches} failed`}
            tone="amber"
          />
          <Kpi
            label="Stored results"
            value={String(trace.summary.returned_results)}
            note={
              providerPayloads > 0
                ? `${providerPayloads} calls include raw provider responses`
                : `${llmFacingOnlyPayloads} calls preserve LLM-facing results`
            }
          />
          <Kpi
            label="Run time"
            value={duration(trace.run.duration_seconds)}
            note={`${searchIterations.size} search-active iterations`}
          />
        </div>

        {runFailed ? (
          <div className="run-alert">
            <div className="run-alert-icon">!</div>
            <div>
              <span>RUN DIAGNOSIS</span>
              <strong>One or more evolution iterations failed.</strong>
              <p>
                {trace.summary.parse_failures} of {trace.summary.llm_attempts} attempts
                did not produce a valid SEARCH/REPLACE diff; the remaining{" "}
                {trace.summary.evaluation_failures} failed evaluation. Tavily itself
                returned {trace.summary.returned_results} inspectable results across{" "}
                {trace.summary.successful_web_searches} successful calls, so retrieval
                and evolution outcome are shown separately below.
              </p>
            </div>
            <code>baseline retained · {score(trace.summary.best_score)}</code>
          </div>
        ) : null}

        {trace.data_quality.tool_results ? (
          <div className="quality-banner complete">
            <div className="quality-icon">✓</div>
            <div>
              <strong>
                {providerPayloads > 0
                  ? "Evolution trace loaded with Tavily provider payloads"
                  : "Evolution trace loaded with LLM-facing Tavily results"}
              </strong>
              <p>
                {providerPayloads > 0
                  ? `${providerPayloads} calls include raw provider responses; the compact title/URL/content payload passed back to the LLM is shown separately.`
                  : `${llmFacingOnlyPayloads} calls preserve the result sent to the LLM. Raw provider responses were not captured for those calls, so the viewer labels them separately.`}
                {trace.data_quality.provider_reasoning
                  ? " Provider reasoning rounds are linked to their LLM calls."
                  : " Provider reasoning was not available for this run."}
              </p>
            </div>
          </div>
        ) : (
          <div className="quality-banner">
            <div className="quality-icon">i</div>
            <div>
              <strong>Evolution trace loaded</strong>
              <p>
                No Tavily provider response payload is present. LLM calls, reasoning,
                and any failed search metadata remain available below.
              </p>
            </div>
            <span>evolution trace</span>
          </div>
        )}

        <PerformanceChart
          iterations={trace.iterations}
          searches={trace.searches}
          selectedSearchId={selectedSearchId}
          selectedIteration={selectedIteration}
          onIterationSelect={(iteration) =>
            setSelection({ kind: "iteration", iteration })
          }
          onSearchSelect={(id) => setSelection({ kind: "search", id })}
        />

        <div className="trace-grid">
          <TraceTimeline
            events={trace.timeline}
            iterations={trace.iterations}
            searches={trace.searches}
            selection={selection}
            onSelect={setSelection}
          />
          <TraceInspector
            selection={selection}
            iterations={trace.iterations}
            attempts={trace.attempts}
            searches={trace.searches}
            onSearchSelect={(id) => setSelection({ kind: "search", id })}
          />
        </div>

        <section className="analyst-notes">
          <div>
            <p className="section-kicker">Reading the run</p>
            <h2>{runFailed ? "What actually happened" : "Three signals worth separating"}</h2>
          </div>
          <article>
            <span>01</span>
            <h3>{runFailed ? "Retrieval worked" : "Candidate score"}</h3>
            <p>
              {runFailed
                ? `${trace.summary.successful_web_searches} of ${trace.summary.web_searches} Tavily calls succeeded and all ${trace.summary.returned_results} returned result cards are preserved exactly.`
                : "The gray line is the proposal evaluated on that iteration. It can fall even while the experiment retains an earlier stronger program."}
            </p>
          </article>
          <article>
            <span>02</span>
            <h3>{runFailed ? "Generation broke" : "Best-so-far"}</h3>
            <p>
              {runFailed
                ? `${trace.summary.parse_failures} attempts returned an unusable diff format; ${trace.summary.evaluation_failures} candidate reached the evaluator but failed validity.`
                : `The green staircase is the actual discovery frontier. It rose from ${score(trace.summary.baseline_score, 3)} to ${score(trace.summary.best_score, 3)}.`}
            </p>
          </article>
          <article>
            <span>03</span>
            <h3>{runFailed ? "Baseline survived" : "Retrieval timing"}</h3>
            <p>
              {runFailed
                ? `With zero accepted candidates, checkpoint ${trace.run.iterations} retained the original program at combined score ${score(trace.summary.best_score, 3)}.`
                : `Web search was active in ${searchIterations.size} iterations. The view exposes sequence, not causal proof.`}
            </p>
          </article>
        </section>
      </section>

      <footer>
        <div>
          <span className="brand-mark small">SD</span>
          <p>Generated exclusively from the run&apos;s evolution trace.</p>
        </div>
        <div className="footer-source">
          <span>DATA PROVENANCE</span>
          <code>{trace.source.evolution_trace.split("/").at(-1)}</code>
        </div>
      </footer>
    </main>
  );
}

function Kpi({
  label,
  value,
  note,
  tone,
}: {
  label: string;
  value: string;
  note: string;
  tone?: "mint" | "amber" | "red";
}) {
  return (
    <article className={`kpi ${tone ?? ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </article>
  );
}
