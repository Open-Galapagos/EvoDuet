"use client";

import type { IterationTrace, SearchTrace } from "@/lib/types";
import { score } from "@/lib/format";

type Props = {
  iterations: IterationTrace[];
  searches: SearchTrace[];
  selectedSearchId?: string;
  selectedIteration?: number;
  onIterationSelect: (iteration: number) => void;
  onSearchSelect: (id: string) => void;
};

const WIDTH = 980;
const HEIGHT = 360;
const MARGIN = { top: 72, right: 32, bottom: 48, left: 64 };

export function PerformanceChart({
  iterations,
  searches,
  selectedSearchId,
  selectedIteration,
  onIterationSelect,
  onSearchSelect,
}: Props) {
  const hasFailures = iterations.some((item) => item.status === "failed");
  const values = iterations
    .flatMap((item) => [item.score, item.best_so_far])
    .filter((value): value is number => value != null);
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const padding = Math.max((rawMax - rawMin) * 0.22, 0.04);
  const min = Math.max(0, rawMin - padding);
  const max = rawMax + padding;
  const maxIteration = Math.max(...iterations.map((item) => item.iteration), 1);
  const plotWidth = WIDTH - MARGIN.left - MARGIN.right;
  const plotHeight = HEIGHT - MARGIN.top - MARGIN.bottom;

  const x = (iteration: number) =>
    MARGIN.left + (iteration / maxIteration) * plotWidth;
  const y = (value: number) =>
    MARGIN.top + ((max - value) / (max - min || 1)) * plotHeight;

  const rawPath = iterations
    .filter((item) => item.score != null)
    .map(
      (item, index) =>
        `${index === 0 ? "M" : "L"} ${x(item.iteration)} ${y(item.score as number)}`,
    )
    .join(" ");

  const bestItems = iterations.filter((item) => item.best_so_far != null);
  const bestPath = bestItems
    .map((item, index) => {
      const currentX = x(item.iteration);
      const currentY = y(item.best_so_far as number);
      if (index === 0) return `M ${currentX} ${currentY}`;
      const previous = bestItems[index - 1];
      return `H ${currentX} V ${currentY}`;
    })
    .join(" ");

  const gridValues = Array.from({ length: 5 }, (_, index) => {
    const value = min + ((max - min) * index) / 4;
    return value;
  });
  const searchesByIteration = searches.reduce<Record<number, SearchTrace[]>>(
    (groups, searchItem) => {
      if (searchItem.iteration == null) return groups;
      (groups[searchItem.iteration] ||= []).push(searchItem);
      return groups;
    },
    {},
  );

  return (
    <div className="chart-shell">
      <div className="chart-heading">
        <div>
          <p className="section-kicker">Performance × retrieval</p>
          <h2>{hasFailures ? "Retained baseline × retrieval" : "Best-so-far trajectory"}</h2>
        </div>
        <div className="chart-legend" aria-label="Chart legend">
          <span>
            <i className="legend-line raw" /> Candidate
          </span>
          <span>
            <i className="legend-line best" /> Best-so-far
          </span>
          <span>
            <i className="legend-search" /> Web search
          </span>
          {hasFailures ? (
            <span>
              <i className="legend-failure" /> Failed iteration
            </span>
          ) : null}
        </div>
      </div>

      <div className="chart-scroll">
        <svg
          className="performance-chart"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label="Candidate and best-so-far score by iteration with web search markers"
        >
          <defs>
            <linearGradient id="bestGlow" x1="0" x2="1">
              <stop offset="0%" stopColor="#19a889" />
              <stop offset="100%" stopColor="#65d5b9" />
            </linearGradient>
            <filter id="softGlow" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {gridValues.map((value) => (
            <g key={value}>
              <line
                className="chart-grid"
                x1={MARGIN.left}
                y1={y(value)}
                x2={WIDTH - MARGIN.right}
                y2={y(value)}
              />
              <text
                className="axis-label y-label"
                x={MARGIN.left - 12}
                y={y(value) + 4}
              >
                {value.toFixed(2)}
              </text>
            </g>
          ))}

          {iterations.map((item) => (
            <g key={`x-${item.iteration}`}>
              <line
                className="x-guide"
                x1={x(item.iteration)}
                y1={MARGIN.top}
                x2={x(item.iteration)}
                y2={HEIGHT - MARGIN.bottom}
              />
              <text
                className="axis-label"
                x={x(item.iteration)}
                y={HEIGHT - 17}
                textAnchor="middle"
              >
                {item.iteration}
              </text>
            </g>
          ))}

          <text
            className="axis-title"
            x={MARGIN.left + plotWidth / 2}
            y={HEIGHT - 1}
            textAnchor="middle"
          >
            ITERATION
          </text>
          <text
            className="axis-title"
            transform={`translate(15 ${MARGIN.top + plotHeight / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            COMBINED SCORE
          </text>

          <path className="raw-path" d={rawPath} />
          <path className="best-path" d={bestPath} />

          {Object.entries(searchesByIteration).map(([iterationKey, items]) => {
            const iteration = Number(iterationKey);
            const chartX = x(iteration);
            const markerStep = Math.min(22, 132 / Math.max(items.length - 1, 1));
            const markerSpan = markerStep * (items.length - 1);
            const markerStart = Math.max(
              MARGIN.left + 8,
              Math.min(
                chartX - markerSpan / 2,
                WIDTH - MARGIN.right - 8 - markerSpan,
              ),
            );
            return (
              <g key={`search-band-${iteration}`}>
                <line
                  className="search-guide"
                  x1={chartX}
                  y1={MARGIN.top - 8}
                  x2={chartX}
                  y2={HEIGHT - MARGIN.bottom}
                />
                <rect
                  className="search-count-bg"
                  x={chartX - 28}
                  y={13}
                  width={56}
                  height={22}
                  rx={11}
                />
                <text
                  className="search-count"
                  x={chartX}
                  y={28}
                  textAnchor="middle"
                >
                  WEB ×{items.length}
                </text>
                {items.map((searchItem, index) => {
                  const markerX = markerStart + index * markerStep;
                  const selected = selectedSearchId === searchItem.id;
                  return (
                    <g
                      className={`search-marker ${selected ? "selected" : ""}`}
                      key={searchItem.id}
                      role="button"
                      tabIndex={0}
                      aria-label={`Web search: ${searchItem.query}`}
                      onClick={() => onSearchSelect(searchItem.id)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          onSearchSelect(searchItem.id);
                        }
                      }}
                    >
                      <circle
                        cx={markerX}
                        cy={MARGIN.top - 7}
                        r={selected ? 9 : 7}
                      />
                      <text x={markerX} y={MARGIN.top - 3} textAnchor="middle">
                        {searchItem.settings.effective_max_results ?? "?"}
                      </text>
                      <title>
                        {`${searchItem.query} · max ${
                          searchItem.settings.effective_max_results ?? "unknown"
                        } · returned ${searchItem.result_count ?? "unknown"}`}
                      </title>
                    </g>
                  );
                })}
              </g>
            );
          })}

          {iterations.map((item) => {
            if (item.score == null && item.status === "failed" && item.best_so_far != null) {
              const selected = selectedIteration === item.iteration;
              const pointX = x(item.iteration);
              const pointY = y(item.best_so_far);
              return (
                <g
                  className={`failure-point ${selected ? "selected" : ""}`}
                  key={`point-${item.iteration}`}
                  role="button"
                  tabIndex={0}
                  aria-label={`Iteration ${item.iteration} failed; retained score ${score(item.best_so_far)}`}
                  onClick={() => onIterationSelect(item.iteration)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      onIterationSelect(item.iteration);
                    }
                  }}
                >
                  <circle cx={pointX} cy={pointY} r={selected ? 11 : 9} />
                  <path
                    d={`M ${pointX - 3.5} ${pointY - 3.5} L ${pointX + 3.5} ${pointY + 3.5} M ${pointX + 3.5} ${pointY - 3.5} L ${pointX - 3.5} ${pointY + 3.5}`}
                  />
                  <title>{item.failure_reason ?? "Iteration failed"}</title>
                </g>
              );
            }
            if (item.score == null) return null;
            const selected = selectedIteration === item.iteration;
            return (
              <g
                className={`iteration-point ${selected ? "selected" : ""}`}
                key={`point-${item.iteration}`}
                role="button"
                tabIndex={0}
                aria-label={`Iteration ${item.iteration}, score ${score(item.score)}`}
                onClick={() => onIterationSelect(item.iteration)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    onIterationSelect(item.iteration);
                  }
                }}
              >
                <circle
                  className="point-halo"
                  cx={x(item.iteration)}
                  cy={y(item.score)}
                  r={selected ? 12 : 9}
                />
                <circle
                  className="point-core"
                  cx={x(item.iteration)}
                  cy={y(item.score)}
                  r={item.is_new_best ? 5 : 4}
                  filter={item.is_new_best ? "url(#softGlow)" : undefined}
                />
                {item.is_new_best && item.iteration > 0 ? (
                  <text
                    className="best-label"
                    x={x(item.iteration)}
                    y={y(item.score) - 15}
                    textAnchor="middle"
                  >
                    NEW BEST
                  </text>
                ) : null}
                <title>
                  {`Iteration ${item.iteration} · candidate ${score(
                    item.score,
                  )} · best ${score(item.best_so_far)}`}
                </title>
              </g>
            );
          })}
        </svg>
      </div>
      <p className="chart-footnote">
        Amber markers are individual Tavily calls; the number inside is effective{" "}
        <code>max_results</code>. {hasFailures ? "Red × markers show iterations that retained the baseline because no candidate was accepted." : "Click any point or search to inspect its trace."}
      </p>
    </div>
  );
}
