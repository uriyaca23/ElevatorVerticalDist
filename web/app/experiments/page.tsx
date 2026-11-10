import { Suspense } from "react";
import { AppHeader } from "@/components/AppHeader";
import { ExperimentsTable } from "@/components/ExperimentsTable";
import { listSessions } from "@/lib/sessions";

export default function ExperimentsPage() {
  return (
    <>
      <AppHeader>Experiments</AppHeader>
      <main className="mx-auto w-full max-w-7xl flex-1 p-4 sm:p-6">
        <Suspense fallback={<p className="text-sm text-neutral-500">Loading experiments…</p>}>
          <Sessions />
        </Suspense>
      </main>
    </>
  );
}

async function Sessions() {
  const result = await listSessions().then(
    (sessions) => ({ sessions }),
    (e: unknown) => ({ error: e instanceof Error ? e.message : String(e) }),
  );
  if ("error" in result) return <p className="text-sm text-red-600">Could not load sessions: {result.error}</p>;
  return <ExperimentsTable sessions={result.sessions} />;
}
