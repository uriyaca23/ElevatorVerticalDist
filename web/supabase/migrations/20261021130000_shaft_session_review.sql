-- Web-app review state per Skydex session, kept apart from shaft_sessions so
-- the phone app's upserts never touch it.
--   approved_*: labelling is finished and the session can be used for research
--               (labels are read-only while approved).
--   deleted_*:  hidden from the web app. Soft delete: the session row, its
--               labels and the sensor files are kept and can be restored by
--               clearing deleted_at.
create table if not exists public.shaft_session_review (
  session_id  bigint primary key references public.shaft_sessions(id) on delete cascade,
  approved_at timestamptz,
  approved_by text,
  deleted_at  timestamptz,
  deleted_by  text,
  updated_at  timestamptz not null default now()
);

-- No anon/authenticated policies: only the web app's server (service_role) reads or writes.
alter table public.shaft_session_review enable row level security;
