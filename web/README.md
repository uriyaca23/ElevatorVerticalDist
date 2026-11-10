# Shaft Lab (web)

Next.js app for experimenters to browse Skydex shaft recordings, view the
barometer / accelerometer / gyroscope traces with the barometer-predicted
elevator rides, attach the building's garamushka PDF, and label the true
ride intervals.

- Sessions: Supabase `public.shaft_sessions` (written by the Skydex app, branch `shaftApp`).
- Sensor CSVs: Firebase Storage bucket `nahori-shaft`, read server-side with the admin SDK.
- Labels: Supabase `public.shaft_labels` (`supabase/migrations/`).
- Garamushka PDFs: `nahori-shaft/web-assets/gramushka/<session_id>.pdf`.
- Predicted rides: `lib/barometer.ts`, a port of the Python pressure-filter
  segmenter, kept in parity by `tests/barometer.test.ts`.
- GPS map + stay / movement: `lib/gpsStay.ts`, a port of the sibling repo
  `gps_stay`'s `ProbabilisticStayDetector`, kept in parity by
  `tests/gpsStay.test.ts`; fixes come from the session's `loc` file (`lib/gnss.ts`).
- NMEA changepoints: per-second mean C/N0 from the `nmea` file (`lib/gnss.ts`,
  matches nahori's `parse_nmea_txt`), segmented by PELT (`lib/changepoint.ts`).

## Setup

```bash
cd web
npm install
cp .env.example .env.local   # fill in the keys
npm run dev                  # http://localhost:3000
```

Apply `supabase/migrations/20261021120000_shaft_labels.sql` once in the
Supabase SQL editor before saving labels.

## Tests

```bash
npm test
# after changing the Python or TS segmenter, regenerate the parity fixture:
../venv/bin/python scripts/make_baro_fixture.py
# after changing gps_stay or lib/gpsStay.ts (needs ../../gps_stay, optionally ../../nahori):
../venv/bin/python scripts/make_gps_stay_fixture.py
```

## Deploy (Vercel)

Import the repo, set **Root Directory** to `web`, and add the variables from
`.env.example` to the project's environment.
