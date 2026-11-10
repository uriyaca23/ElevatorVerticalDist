// Types shared between server code and client components.
import type { RideType, Segment } from "@/lib/barometer";
import type { StayClass, StaySegment } from "@/lib/gpsStay";

export type { RideType, Segment, StayClass, StaySegment };

/** One row of public.shaft_sessions (written by the Skydex app, branch shaftApp). */
export interface SessionRow {
  id: number;
  username: string;
  session_tag: string;
  start_time: string | null;
  end_time: string | null;
  duration_sec: number | null;
  files: Record<string, string>;
  file_count: number | null;
  total_bytes: number | null;
  device_brand: string | null;
  device_manufacturer: string | null;
  device_model: string | null;
  device_id: string | null;
  android_version: string | null;
  android_sdk: number | null;
  app_version: string | null;
  stop_reason: string | null;
  uploaded_at: string | null;
  recording_settings: {
    timezone?: string;
    user_info?: string;
    enabled_sensors?: string[];
    sensor_frequency_us?: number;
    storage_threshold_mb?: number;
    // Set only on sessions imported from the pre-Skydex sensorLog experiments.
    source?: string;
    exp_name?: string;
    experimenter?: string;
    location?: string;
    description?: string;
    experiment_type?: string;
    temperature_c?: number | null;
    start_floor?: string;
    gramushka_path?: string | null;
  } | null;
  device_details: Record<string, unknown> | null;
  sensor_hardware: Record<string, Record<string, unknown> | null> | null;
}

/** public.shaft_session_review — web-app approval / soft-delete state. */
export interface SessionReview {
  approved_at: string | null;
  approved_by: string | null;
  deleted_at: string | null;
}

export interface SessionListItem extends SessionRow {
  label_count: number;
  approved_at: string | null;
}

/** A user-drawn ride interval (public.shaft_labels). Outside = everything unlabelled. */
export interface Label {
  start_ms: number;
  end_ms: number;
  type: RideType;
  height_diff_m: number | null;
  description?: string | null;
  labeled_by?: string | null;
  updated_at?: string | null;
}

/** Column-oriented series; `t` is seconds since `SensorPayload.t0_ms`. */
export interface Series {
  t: number[];
  [column: string]: number[];
}

export interface SensorPayload {
  t0_ms: number;
  t1_ms: number;
  barometer: (Series & { height: number[]; height_smooth: number[] }) | null;
  /** 50 Hz resampled accel + reconstructed vertical acceleration (lib/reconstruct.ts). */
  accel: (Series & { x: number[]; y: number[]; z: number[]; a_rec: number[]; a_mag_g: number[] }) | null;
  gyro: (Series & { x: number[]; y: number[]; z: number[] }) | null;
  /** Best GPS fix per second (lib/gnss.ts); accuracy is null where the provider left it empty. */
  gps: { t: number[]; lat: number[]; lon: number[]; accuracy: (number | null)[] } | null;
  /** gps_stay segmentation of `gps` (lib/gpsStay.ts); start/end_index index into it. */
  stays: StaySegment[];
  /** Per-second NMEA mean C/N0, dB-Hz (0 = GSV reported but nothing tracked). */
  nmea: { t: number[]; snr: number[] } | null;
  segments: Segment[];
  decimated: Record<string, { from: number; to: number }>;
  errors: Record<string, string>;
}
