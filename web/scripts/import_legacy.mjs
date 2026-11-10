// Import the pre-Skydex experiments (src/data/structuredData) into the Skydex
// cloud layout so Shaft Lab shows them next to new recordings.
//
//   cd web && node --env-file=.env.local scripts/import_legacy.mjs [--dry-run] [--labels | --labels-only] [exp_name ...]
//
// Per experiment:
//   - ACC/GYR/MAG/PRS.csv → Skydex-format CSVs ("timestamp" = device-local ISO
//     with +HHMM offset, quoted fields) at
//     nahori-shaft/legacy/<exp>/<folder>/<exp>_<suffix>.csv
//   - one public.shaft_sessions row, upserted on (device_id, session_tag) so
//     re-running is idempotent; experiment metadata goes in recording_settings
//   - the building's garamushka PDF, uploaded once per building and linked via
//     recording_settings.gramushka_path
//   - with --labels: gt.csv up/down rides → public.shaft_labels (needs the
//     migration + SUPABASE_SERVICE_ROLE_KEY; skips sessions that already have labels)
//   - --labels-only: just the label import for already-imported sessions (no uploads)
//
// Experiments default to the folders in src/data/rawData (the real recordings;
// structuredData also holds synthetic __corrupted variants, which are skipped).

import fs from "node:fs";
import path from "node:path";
import { cert, initializeApp } from "firebase-admin/app";
import { getStorage } from "firebase-admin/storage";
import { createClient } from "@supabase/supabase-js";

const ROOT = path.resolve(import.meta.dirname, "../..");
const DATA = path.join(ROOT, "src/data");
const TZ = "Asia/Jerusalem";

const args = process.argv.slice(2);
const DRY = args.includes("--dry-run");
const LABELS_ONLY = args.includes("--labels-only");
const LABELS = LABELS_ONLY || args.includes("--labels");
const only = args.filter((a) => !a.startsWith("--"));

// Mirrors src/data/loader/gramushka.py (_BUILDING_SLUG_TO_FOLDER / exp3 rule).
const BUILDINGS = {
  milleniumhotel: "פרימה מילניום",
  milleniumoutside: "פרימה מילניום",
  acrobuilding: "אקרו נדלן",
  beitmansour1: "בית_מנצור_1",
  beityitzchakiraanana: "בית יצחקי ב",
  barilan2herzelia: "בר אילן 2",
  haari3: "haari",
};

const SENSORS = [
  { key: "accel", src: "ACC.csv", folder: "accel", suffix: "_accel.csv", cols: ["x", "y", "z"] },
  { key: "gyro", src: "GYR.csv", folder: "gyro", suffix: "_gyro.csv", cols: ["x", "y", "z"] },
  { key: "magnetic", src: "MAG.csv", folder: "magnetic", suffix: "_magnetic.csv", cols: ["x", "y", "z"] },
  { key: "barometer", src: "PRS.csv", folder: "barometer", suffix: "_device_baro.csv", cols: ["pressure"] },
];

// ---------- helpers ----------

function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let q = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (q) {
      if (c === '"' && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else if (c === '"') q = false;
      else cur += c;
    } else if (c === '"') q = true;
    else if (c === ",") {
      out.push(cur);
      cur = "";
    } else cur += c;
  }
  out.push(cur);
  return out;
}

function readCsv(file) {
  const lines = fs.readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean);
  const header = parseCsvLine(lines[0]);
  return lines.slice(1).map((l) => {
    const f = parseCsvLine(l);
    return Object.fromEntries(header.map((h, i) => [h, f[i]]));
  });
}

/** Offset of TZ at `ms` in minutes, e.g. 180 for IDT. */
function tzOffsetMin(ms) {
  const name = new Intl.DateTimeFormat("en-US", { timeZone: TZ, timeZoneName: "longOffset" })
    .formatToParts(new Date(ms))
    .find((p) => p.type === "timeZoneName").value; // "GMT+03:00"
  const m = name.match(/GMT([+-])(\d\d):(\d\d)/);
  return m ? (m[1] === "-" ? -1 : 1) * (Number(m[2]) * 60 + Number(m[3])) : 0;
}

/** Skydex row timestamp: `2026-04-15T13:20:43.878+0300`. */
function skydexTimestamp(ms, offMin) {
  const local = new Date(ms + offMin * 60_000).toISOString().slice(0, 23);
  const sign = offMin < 0 ? "-" : "+";
  const a = Math.abs(offMin);
  return `${local}${sign}${String(Math.floor(a / 60)).padStart(2, "0")}${String(a % 60).padStart(2, "0")}`;
}

/** Skydex session tag: device-local yyyyMMddHHmm. */
function sessionTag(ms, offMin) {
  return new Date(ms + offMin * 60_000).toISOString().slice(0, 16).replace(/[-T:]/g, "");
}

// SensorManager.getAltitude(1018, p), as BarometerHandler writes it.
const androidAltitude = (p) => 44330 * (1 - Math.pow(p / 1018, 1 / 5.255));

const q = (v) => `"${v}"`;

function toSkydexCsv(rows, sensor, offMin) {
  const header = sensor.key === "barometer" ? ["timestamp", "pressure", "altitude"] : ["timestamp", ...sensor.cols];
  const out = [header.map(q).join(",")];
  for (const r of rows) {
    const ts = skydexTimestamp(Number(r.timestamp_ms), offMin);
    const vals =
      sensor.key === "barometer"
        ? [r.pressure, androidAltitude(Number(r.pressure)).toFixed(5)]
        : sensor.cols.map((c) => r[c]);
    out.push([ts, ...vals].map(q).join(","));
  }
  return out.join("\r\n") + "\r\n";
}

function parsePhone(phone) {
  const [brand, ...rest] = (phone || "").trim().split(/\s+/);
  return { brand: (brand || "").toLowerCase() || null, model: rest.join(" ") || null };
}

function gramushkaSource(exp) {
  if (exp.toLowerCase().endsWith("_exp3")) return null; // project convention: no plan for exp3
  const slug = exp.split("_").map((t) => t.toLowerCase()).find((t) => t in BUILDINGS);
  if (!slug) return null;
  const dir = path.join(DATA, "gramushka", BUILDINGS[slug]);
  const pdf = fs.existsSync(dir) && fs.readdirSync(dir).find((f) => f.toLowerCase().endsWith(".pdf"));
  if (pdf) return { slug, file: path.join(dir, pdf) };
  const raw = path.join(DATA, "rawData", exp, "gramushka.pdf");
  return fs.existsSync(raw) ? { slug, file: raw } : null;
}

// ---------- clients ----------

const bucket = getStorage(initializeApp({ credential: cert(JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT)) })).bucket(
  process.env.FIREBASE_BUCKET,
);
const db = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY,
  { auth: { persistSession: false } },
);

async function upload(dest, body, contentType) {
  if (DRY) return;
  await bucket.file(dest).save(body, { contentType, resumable: false });
}

const uploadedPlans = new Map();
async function uploadPlan(src) {
  const dest = `web-assets/gramushka/buildings/${src.slug.replace(/^milleniumoutside$/, "milleniumhotel")}.pdf`;
  if (uploadedPlans.has(dest)) return dest;
  const size = fs.statSync(src.file).size;
  const [exists] = DRY ? [false] : await bucket.file(dest).exists();
  const same = exists && Number((await bucket.file(dest).getMetadata())[0].size) === size;
  if (!same) await upload(dest, fs.readFileSync(src.file), "application/pdf");
  console.log(`  plan ${dest} (${(size / 1e6).toFixed(1)} MB${same ? ", already there" : ""})`);
  uploadedPlans.set(dest, true);
  return dest;
}

// ---------- main ----------

const experiments = only.length
  ? only
  : fs.readdirSync(path.join(DATA, "rawData")).filter((d) => fs.statSync(path.join(DATA, "rawData", d)).isDirectory());

let failed = 0;
for (const exp of experiments.sort()) {
  const dir = path.join(DATA, "structuredData/data", exp);
  if (!fs.existsSync(path.join(dir, "PRS.csv"))) {
    console.log(`skip ${exp}: no structuredData/PRS.csv`);
    continue;
  }
  try {
    console.log(`${DRY ? "[dry] " : ""}${exp}`);
    if (LABELS_ONLY) {
      const { data, error } = await db.from("shaft_sessions").select("id").eq("device_id", `legacy:${exp}`).maybeSingle();
      if (error) throw new Error(`shaft_sessions: ${error.message}`);
      if (!data) throw new Error("not imported yet — run without --labels-only first");
      if (!DRY) await importLabels(data.id, dir);
      continue;
    }
    const meta = readCsv(path.join(dir, "metadata.csv"))[0] ?? {};
    const tables = Object.fromEntries(
      SENSORS.filter((s) => fs.existsSync(path.join(dir, s.src))).map((s) => [s.key, readCsv(path.join(dir, s.src))]),
    );
    const all = Object.values(tables).flatMap((rows) => [Number(rows[0].timestamp_ms), Number(rows.at(-1).timestamp_ms)]);
    const t0 = Math.min(...all);
    const t1 = Math.max(...all);
    const off = tzOffsetMin(t0);

    const files = {};
    let totalBytes = 0;
    for (const s of SENSORS) {
      if (!tables[s.key]) continue;
      const csv = toSkydexCsv(tables[s.key], s, off);
      const dest = `legacy/${exp}/${s.folder}/${exp}${s.suffix}`;
      await upload(dest, csv, "text/csv");
      files[s.key] = dest;
      totalBytes += Buffer.byteLength(csv);
      console.log(`  ${dest} (${tables[s.key].length} rows)`);
    }

    const plan = gramushkaSource(exp);
    const gramushkaPath = plan ? await uploadPlan(plan) : null;
    const { brand, model } = parsePhone(meta.phone);
    const temperature = Number.parseFloat(meta.temperature_c);
    const row = {
      username: exp.split("_")[0],
      session_tag: sessionTag(t0, off),
      start_time: new Date(t0).toISOString(),
      end_time: new Date(t1).toISOString(),
      files,
      file_count: Object.keys(files).length,
      total_bytes: totalBytes,
      device_brand: brand,
      device_manufacturer: brand,
      device_model: model,
      device_id: `legacy:${exp}`,
      stop_reason: "legacy_import",
      recording_settings: {
        timezone: TZ,
        user_info: [meta.location, meta.description].filter(Boolean).join(" — "),
        enabled_sensors: Object.keys(files),
        source: "legacy_import",
        exp_name: exp,
        experimenter: meta.experimenter ?? null,
        location: meta.location ?? null,
        description: meta.description ?? null,
        experiment_type: meta.experiment_type ?? null,
        temperature_c: Number.isFinite(temperature) ? temperature : null,
        start_floor: meta.start_floor ?? null,
        gramushka_path: gramushkaPath,
      },
    };
    if (DRY) {
      console.log(`  row ${row.username} ${row.session_tag} ${model} ${(totalBytes / 1e6).toFixed(1)} MB`);
      continue;
    }
    const { data, error } = await db
      .from("shaft_sessions")
      .upsert(row, { onConflict: "device_id,session_tag" })
      .select("id")
      .single();
    if (error) throw new Error(`shaft_sessions: ${error.message}`);
    console.log(`  session id ${data.id}`);

    if (LABELS) await importLabels(data.id, dir);
  } catch (e) {
    failed++;
    console.error(`  FAILED: ${e.message}`);
  }
}
console.log(failed ? `${failed} experiment(s) failed` : "done");
process.exit(failed ? 1 : 0);

async function importLabels(sessionId, dir) {
  const gtFile = path.join(dir, "gt.csv");
  if (!fs.existsSync(gtFile)) return;
  const existing = await db.from("shaft_labels").select("id", { count: "exact", head: true }).eq("session_id", sessionId);
  if (existing.error) throw new Error(`shaft_labels: ${existing.error.message}`);
  if (existing.count) {
    console.log(`  labels: session already has ${existing.count}, left alone`);
    return;
  }
  const rides = readCsv(gtFile).filter((r) => r.type === "up" || r.type === "down");
  const now = new Date().toISOString();
  const { error } = await db.from("shaft_labels").insert(
    rides.map((r) => ({
      session_id: sessionId,
      start_ms: Number(r.start_ms),
      end_ms: Number(r.end_ms),
      type: r.type,
      height_diff_m: r.height_diff_m === "" ? null : Number(r.height_diff_m),
      description: r.description || "imported from gt.csv",
      labeled_by: "gt.csv",
      updated_at: now,
    })),
  );
  if (error) throw new Error(`shaft_labels insert: ${error.message}`);
  console.log(`  labels: ${rides.length} rides from gt.csv`);
}
