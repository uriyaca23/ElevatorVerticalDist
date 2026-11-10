// Display helpers shared by server and client components.

export function formatBytes(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}

export function formatDuration(sec: number | null | undefined): string {
  if (sec == null) return "—";
  const s = Math.round(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${r}s` : `${r}s`;
}

/** Wall-clock time in the recording's own timezone (falls back to Israel time). */
export function formatDateTime(iso: string | number | null | undefined, timeZone = "Asia/Jerusalem"): string {
  if (iso == null) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString("en-GB", { timeZone, dateStyle: "medium", timeStyle: "medium" });
}

export function formatClock(ms: number, timeZone = "Asia/Jerusalem"): string {
  const d = new Date(ms);
  const base = d.toLocaleTimeString("en-GB", { timeZone, hour12: false });
  return `${base}.${String(d.getUTCMilliseconds()).padStart(3, "0")}`;
}

export function deviceName(s: { device_brand: string | null; device_manufacturer: string | null; device_model: string | null }) {
  const brand = s.device_brand || s.device_manufacturer || "";
  return [brand && brand[0].toUpperCase() + brand.slice(1), s.device_model].filter(Boolean).join(" ") || "—";
}
