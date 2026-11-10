-- Ground-truth elevator ride intervals drawn in the Shaft Lab web app.
-- One row per ride; time not covered by a row is "outside". Column names
-- mirror the research repo's gt.csv (start_ms, end_ms, type, height_diff_m).
create table if not exists public.shaft_labels (
  id            bigserial primary key,
  session_id    bigint not null references public.shaft_sessions(id) on delete cascade,
  start_ms      bigint not null,
  end_ms        bigint not null,
  type          text   not null check (type in ('up', 'down')),
  height_diff_m double precision,
  description   text,
  labeled_by    text,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now(),
  check (end_ms > start_ms)
);

create index if not exists shaft_labels_session_id_idx on public.shaft_labels (session_id);

-- No anon/authenticated policies: only the web app's server (service_role) reads or writes.
alter table public.shaft_labels enable row level security;
