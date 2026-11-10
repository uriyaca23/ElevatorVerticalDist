import { notFound } from "next/navigation";
import { Suspense } from "react";
import { AppHeader } from "@/components/AppHeader";
import { DeleteExperimentButton } from "@/components/DeleteExperimentButton";
import { ExperimentWorkspace } from "@/components/ExperimentWorkspace";
import { GramushkaViewer } from "@/components/GramushkaViewer";
import { MetadataSummary } from "@/components/MetadataSummary";
import { getSession } from "@/lib/sessions";

export default function ExperimentPage({ params }: PageProps<"/experiments/[id]">) {
  return (
    <Suspense
      fallback={
        <>
          <AppHeader>Experiments</AppHeader>
          <p className="p-6 text-sm text-neutral-500">Loading experiment…</p>
        </>
      }
    >
      <Experiment params={params} />
    </Suspense>
  );
}

async function Experiment({ params }: { params: PageProps<"/experiments/[id]">["params"] }) {
  const id = Number((await params).id);
  if (!Number.isInteger(id)) notFound();
  const session = await getSession(id);
  if (!session) notFound();
  return (
    <>
      <AppHeader>
        Experiments / #{session.id} · {session.username} · {session.session_tag}
      </AppHeader>
      <main className="mx-auto w-full max-w-[1600px] flex-1 space-y-4 p-4 sm:p-6">
        <MetadataSummary
          session={session}
          actions={
            <DeleteExperimentButton
              sessionId={session.id}
              name={session.recording_settings?.exp_name ?? `${session.username} · ${session.session_tag}`}
            />
          }
        />
        <GramushkaViewer sessionId={session.id} />
        <ExperimentWorkspace sessionId={session.id} timeZone={session.recording_settings?.timezone ?? "Asia/Jerusalem"} />
      </main>
    </>
  );
}
