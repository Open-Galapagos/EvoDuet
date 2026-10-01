export type Metrics = Record<string, number | string | boolean | null>;

export type ReasoningCapture =
  | "provider_reasoning"
  | "visible_response_rationale_only"
  | "unavailable";

export type ReasoningResponseTrace = {
  round?: number | null;
  phase?: string | null;
  response_id?: string | null;
  model?: string | null;
  status?: string | null;
  finish_reason?: string | null;
  content?: string;
  reasoning?: unknown;
  reasoning_content?: unknown;
  reasoning_details?: unknown;
  usage?: Record<string, unknown>;
  tool_calls?: unknown[];
  [key: string]: unknown;
};

export type IterationTrace = {
  iteration: number;
  status: "baseline" | "accepted" | "failed" | "incomplete";
  failure_reason: string | null;
  failure_kind: string | null;
  program_id: string | null;
  retained_program_id: string | null;
  parent_id: string | null;
  timestamp: string | null;
  metrics: Metrics;
  score: number | null;
  best_so_far: number | null;
  is_new_best: boolean;
  delta_from_parent: number | null;
  duration_seconds: number | null;
  llm_seconds: number | null;
  eval_seconds: number | null;
  changes: string;
  visible_rationale: string;
  reasoning: string;
  reasoning_capture: ReasoningCapture;
  llm_response: string;
  prompt: {
    system: string;
    user: string;
  };
  search_ids: string[];
  attempt_ids: string[];
};

export type AttemptTrace = {
  id: string;
  call_id: string | null;
  iteration: number | null;
  attempt: number | null;
  timestamp: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  model: string | null;
  api_base?: string | null;
  tool_choice?: string | null;
  max_tool_rounds?: number | null;
  phase?: string | null;
  association?: string | null;
  program_id?: string | null;
  source_program_id?: string | null;
  status: "completed" | "failed";
  failure_reason: string | null;
  failure_kind: string | null;
  response: string;
  visible_rationale: string;
  reasoning: string;
  reasoning_capture: ReasoningCapture;
  rounds: number;
  usage: Record<string, number>;
  prompt: {
    system: string;
    user: string;
  };
  request: Record<string, unknown>;
  responses: ReasoningResponseTrace[];
  tool_executions: Record<string, unknown>[];
  raw_call: Record<string, unknown>;
  search_ids: string[];
  tavily_searches: number;
  returned_results: number;
};

export type SearchResult = {
  id?: string;
  title?: string;
  url?: string;
  content?: string;
  raw_content?: string | null;
  score?: number;
  published_date?: string | null;
  favicon?: string | null;
  images?: unknown[];
  [key: string]: unknown;
};

export type SearchTrace = {
  id: string;
  tool_call_id?: string | null;
  call_id?: string | null;
  attempt_id?: string | null;
  iteration: number | null;
  attempt: number | null;
  round: number | null;
  timestamp: string | null;
  completed_at?: string | null;
  duration_ms?: number | null;
  name: string;
  status: string;
  http_status?: number | null;
  query: string;
  arguments: Record<string, unknown>;
  request?: Record<string, unknown>;
  llm_tool_result?: Record<string, unknown> | string | null;
  error?: unknown;
  program_id?: string | null;
  source_program_id?: string | null;
  association?: string | null;
  raw_record: Record<string, unknown>;
  settings: {
    search_depth?: string;
    topic?: string;
    configured_max_results?: number;
    requested_max_results?: number | null;
    effective_max_results?: number;
    max_tool_rounds?: number;
    tool_choice?: string;
  };
  result_count: number | null;
  result_provenance: "evolution_trace";
  result: {
    answer?: string | null;
    results?: SearchResult[];
    response_time?: number;
    usage?: Record<string, number>;
    error?: string;
    [key: string]: unknown;
  } | null;
  capture_status:
    | "provider_response"
    | "llm_tool_result_only"
    | "metadata_only";
};

export type TimelineEvent = {
  type: "iteration" | "web_search";
  timestamp: string | null;
  iteration: number | null;
  label: string;
  score?: number | null;
  best_so_far?: number | null;
  status?: string;
  search_id?: string;
  attempt?: number | null;
  round?: number | null;
};

export type TraceData = {
  schema_version: number;
  generated_at: string;
  run: {
    title: string;
    task: string;
    model: string;
    provider: string;
    status: string;
    run_dir: string;
    started_at: string | null;
    completed_at: string | null;
    duration_seconds: number | null;
    iterations: number;
    tool_config: {
      search_depth?: string;
      configured_max_results?: number;
      max_tool_rounds?: number;
      tool_choice?: string;
    };
  };
  summary: {
    baseline_score: number | null;
    final_score: number | null;
    best_score: number | null;
    best_iteration: number | null;
    best_program_id: string | null;
    best_sum_radii: number | null;
    relative_lift_percent: number | null;
    attempted_iterations: number;
    failed_iterations: number;
    accepted_candidates: number;
    llm_attempts: number;
    parse_failures: number;
    evaluation_failures: number;
    web_searches: number;
    successful_web_searches: number;
    failed_web_searches: number;
    returned_results: number;
  };
  data_quality: {
    mode: "evolution_trace";
    tool_arguments: boolean;
    tool_results: boolean;
    provider_reasoning: boolean;
    visible_response_rationale: boolean;
    tool_results_provenance?: string;
    notes: string[];
  };
  iterations: IterationTrace[];
  attempts: AttemptTrace[];
  searches: SearchTrace[];
  timeline: TimelineEvent[];
  source: {
    evolution_trace: string;
  };
};
