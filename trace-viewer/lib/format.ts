export function score(value: number | null | undefined, digits = 4) {
  return value == null ? "—" : value.toFixed(digits);
}

export function signed(value: number | null | undefined, digits = 4) {
  if (value == null) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

export function duration(seconds: number | null | undefined) {
  if (seconds == null) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${remainder}s`;
}

export function latency(milliseconds: number | null | undefined) {
  if (milliseconds == null) return "—";
  return milliseconds < 1000
    ? `${Math.round(milliseconds)}ms`
    : `${(milliseconds / 1000).toFixed(1)}s`;
}

export function clock(timestamp: string | null | undefined) {
  if (!timestamp) return "—";
  return new Intl.DateTimeFormat("en", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date(timestamp));
}

export function compactId(value: string | null | undefined) {
  return value ? value.slice(0, 8) : "—";
}
