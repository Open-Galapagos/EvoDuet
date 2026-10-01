"use client";

import { useMemo, useState } from "react";
import { clock, score } from "@/lib/format";
import type { IterationTrace, SearchTrace, TimelineEvent } from "@/lib/types";

type Selection =
  | { kind: "iteration"; iteration: number }
  | { kind: "search"; id: string };

type Props = {
  events: TimelineEvent[];
  iterations: IterationTrace[];
  searches: SearchTrace[];
  selection: Selection;
  onSelect: (selection: Selection) => void;
};

export function TraceTimeline({
  events,
  iterations,
  searches,
  selection,
  onSelect,
}: Props) {
  const [filter, setFilter] = useState<"all" | "web" | "failures">("all");
  const iterationMap = useMemo(
    () => new Map(iterations.map((item) => [item.iteration, item])),
    [iterations],
  );
  const searchMap = useMemo(
    () => new Map(searches.map((item) => [item.id, item])),
    [searches],
  );
  const filtered = events.filter((event) => {
    if (filter === "web") return event.type === "web_search";
    if (filter === "failures") {
      return (
        event.type === "iteration" &&
        iterationMap.get(event.iteration ?? -1)?.status === "failed"
      );
    }
    return true;
  });

  return (
    <aside className="timeline-panel">
      <div className="panel-heading">
        <div>
          <p className="section-kicker">Execution trace</p>
          <h2>Run timeline</h2>
        </div>
        <span className="event-count">{filtered.length}</span>
      </div>

      <div className="segmented" aria-label="Filter trace events">
        {(["all", "web", "failures"] as const).map((value) => (
          <button
            className={filter === value ? "active" : ""}
            key={value}
            onClick={() => setFilter(value)}
          >
            {value === "all"
              ? "All"
              : value === "web"
                ? "Web only"
                : "Failures"}
          </button>
        ))}
      </div>

      <div className="timeline-list">
        {filtered.map((event, index) => {
          if (event.type === "web_search") {
            const searchItem = searchMap.get(event.search_id ?? "");
            if (!searchItem) return null;
            const active =
              selection.kind === "search" && selection.id === searchItem.id;
            return (
              <button
                className={`timeline-event web ${searchItem.status} ${
                  active ? "active" : ""
                }`}
                key={`${event.type}-${event.search_id}`}
                onClick={() => onSelect({ kind: "search", id: searchItem.id })}
              >
                <span className="event-rail">
                  <i className="event-node" />
                  {index < filtered.length - 1 ? <i className="event-line" /> : null}
                </span>
                <span className="event-body">
                  <span className="event-meta">
                    <span>WEB SEARCH</span>
                    <time>{clock(event.timestamp)}</time>
                  </span>
                  <strong>{searchItem.query}</strong>
                  <span className="event-sub">
                    Iteration {searchItem.iteration ?? "—"} · attempt{" "}
                    {searchItem.attempt ?? "—"} · round{" "}
                    {(searchItem.round ?? 0) + 1}
                    <b>max {searchItem.settings.effective_max_results ?? "—"}</b>
                    <b>{searchItem.result_count ?? "—"} returned</b>
                  </span>
                </span>
              </button>
            );
          }

          const iteration = iterationMap.get(event.iteration ?? -1);
          if (!iteration) return null;
          const active =
            selection.kind === "iteration" &&
            selection.iteration === iteration.iteration;
          return (
            <button
              className={`timeline-event iteration ${active ? "active" : ""}`}
              key={`${event.type}-${event.iteration}`}
              onClick={() =>
                onSelect({ kind: "iteration", iteration: iteration.iteration })
              }
            >
              <span className="event-rail">
                <i
                  className={`event-node ${
                    iteration.status === "failed"
                      ? "failed"
                      : iteration.is_new_best
                        ? "best"
                        : ""
                  }`}
                />
                {index < filtered.length - 1 ? <i className="event-line" /> : null}
              </span>
              <span className="event-body">
                <span className="event-meta">
                  <span>
                    {iteration.iteration === 0
                      ? "BASELINE"
                      : `ITERATION ${iteration.iteration}`}
                  </span>
                  <time>{clock(event.timestamp)}</time>
                </span>
                <strong>
                  {iteration.status === "failed"
                    ? "Iteration failed"
                    : iteration.is_new_best && iteration.iteration > 0
                    ? "New best candidate"
                    : "Candidate evaluated"}
                </strong>
                <span className="event-sub">
                  {iteration.score == null
                    ? "no candidate accepted"
                    : `score ${score(iteration.score)}`}
                  <b>best {score(iteration.best_so_far)}</b>
                  {iteration.search_ids.length > 0 ? (
                    <b className="web-pill">web ×{iteration.search_ids.length}</b>
                  ) : null}
                </span>
              </span>
            </button>
          );
        })}
      </div>
    </aside>
  );
}

export type { Selection };
