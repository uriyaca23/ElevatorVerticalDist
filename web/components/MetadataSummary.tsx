import { deviceName, formatBytes, formatDateTime, formatDuration } from "@/lib/format";
import type { SessionRow } from "@/lib/types";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs uppercase tracking-wide text-neutral-500">{label}</dt>
      <dd className="truncate text-sm">{children}</dd>
    </div>
  );
}

export function MetadataSummary({ session: s, actions }: { session: SessionRow; actions?: React.ReactNode }) {
  const tz = s.recording_settings?.timezone ?? "Asia/Jerusalem";
  const freqUs = s.recording_settings?.sensor_frequency_us;
  const hw = Object.entries(s.sensor_hardware ?? {}).filter(([, v]) => v);
  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-4">
      {actions && <div className="mb-3 flex justify-end gap-2">{actions}</div>}
      <dl className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3 lg:grid-cols-6">
        <Field label="User">{s.username}</Field>
        <Field label="Start">{formatDateTime(s.start_time, tz)}</Field>
        <Field label="End">{formatDateTime(s.end_time, tz)}</Field>
        <Field label="Duration">{formatDuration(s.duration_sec)}</Field>
        <Field label="Stop reason">{s.stop_reason ?? "—"}</Field>
        <Field label="Note">
          <span dir="auto">{s.recording_settings?.user_info || "—"}</span>
        </Field>
        <Field label="Device">{deviceName(s)}</Field>
        <Field label="Android">
          {s.android_version ?? "—"} {s.android_sdk ? `(SDK ${s.android_sdk})` : ""}
        </Field>
        <Field label="App version">{s.app_version ?? "—"}</Field>
        <Field label="Sample rate">{freqUs ? `${(1e6 / freqUs).toFixed(0)} Hz requested` : "—"}</Field>
        <Field label="Files">
          {s.file_count ?? "—"} · {formatBytes(s.total_bytes)}
        </Field>
        <Field label="Uploaded">{formatDateTime(s.uploaded_at, tz)}</Field>
      </dl>

      {s.recording_settings?.source === "legacy_import" && (
        <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-3 border-t border-neutral-100 pt-3 sm:grid-cols-3 lg:grid-cols-6">
          <Field label="Experiment">{s.recording_settings.exp_name}</Field>
          <Field label="Experimenter">{s.recording_settings.experimenter || "—"}</Field>
          <Field label="Location">{s.recording_settings.location || "—"}</Field>
          <Field label="Start floor">{s.recording_settings.start_floor || "—"}</Field>
          <Field label="Temperature">
            {s.recording_settings.temperature_c != null ? `${s.recording_settings.temperature_c} °C` : "—"}
          </Field>
          <Field label="Type">{s.recording_settings.experiment_type || "—"}</Field>
          {s.recording_settings.description && (
            <div className="col-span-full">
              <dt className="text-xs uppercase tracking-wide text-neutral-500">Description</dt>
              <dd className="text-sm" dir="auto">
                {s.recording_settings.description}
              </dd>
            </div>
          )}
        </dl>
      )}

      <div className="mt-4 flex flex-wrap gap-1.5">
        {Object.entries(s.files ?? {}).map(([sensor, path]) => (
          <span key={sensor} title={path} className="rounded bg-neutral-100 px-2 py-0.5 text-xs text-neutral-700">
            {sensor}
          </span>
        ))}
      </div>

      {hw.length > 0 && (
        <details className="mt-3 text-sm">
          <summary className="cursor-pointer text-neutral-500 hover:text-neutral-900">Sensor hardware</summary>
          <table className="mt-2 w-full text-xs">
            <thead className="text-left text-neutral-500">
              <tr>
                <th className="py-1 pr-3 font-medium">Sensor</th>
                <th className="py-1 pr-3 font-medium">Name</th>
                <th className="py-1 pr-3 font-medium">Vendor</th>
                <th className="py-1 pr-3 font-medium">Resolution</th>
                <th className="py-1 pr-3 font-medium">Min delay (µs)</th>
              </tr>
            </thead>
            <tbody>
              {hw.map(([sensor, v]) => (
                <tr key={sensor} className="border-t border-neutral-100">
                  <td className="py-1 pr-3">{sensor}</td>
                  <td className="py-1 pr-3">{String(v!.name ?? "")}</td>
                  <td className="py-1 pr-3">{String(v!.vendor ?? "")}</td>
                  <td className="py-1 pr-3 tabular-nums">{String(v!.resolution ?? "")}</td>
                  <td className="py-1 pr-3 tabular-nums">{String(v!.min_delay_us ?? "")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </section>
  );
}
