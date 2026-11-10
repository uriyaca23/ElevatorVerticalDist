import "server-only";
import { supabase } from "@/lib/clients/supabase";
import type { SessionReview } from "@/lib/types";

const EMPTY: SessionReview = { approved_at: null, approved_by: null, deleted_at: null };

// Before the migration is applied (or without the service key) the table is
// unreadable; every session then reads as "not approved, not deleted".
// Reads use retry(false): a missing table answers 503 (schema-cache reload),
// which supabase-js would otherwise retry with backoff for 10–30 s.
export async function listReviews(): Promise<Map<number, SessionReview>> {
  const out = new Map<number, SessionReview>();
  const { data, error } = await supabase().from("shaft_session_review").select("session_id,approved_at,approved_by,deleted_at").retry(false);
  if (error) return out;
  for (const { session_id, ...r } of data as (SessionReview & { session_id: number })[]) out.set(session_id, r);
  return out;
}

export async function getReview(sessionId: number): Promise<SessionReview> {
  const { data, error } = await supabase()
    .from("shaft_session_review")
    .select("approved_at,approved_by,deleted_at")
    .eq("session_id", sessionId)
    .maybeSingle()
    .retry(false);
  if (error || !data) return EMPTY;
  return data as SessionReview;
}

async function upsert(sessionId: number, patch: Record<string, string | null>): Promise<SessionReview> {
  const { error } = await supabase()
    .from("shaft_session_review")
    .upsert({ session_id: sessionId, ...patch, updated_at: new Date().toISOString() }, { onConflict: "session_id" });
  if (error) throw new Error(`shaft_session_review: ${error.message}`);
  return getReview(sessionId);
}

export function setApproved(sessionId: number, approved: boolean, user: string): Promise<SessionReview> {
  return upsert(sessionId, approved
    ? { approved_at: new Date().toISOString(), approved_by: user }
    : { approved_at: null, approved_by: null });
}

export function markDeleted(sessionId: number, user: string): Promise<SessionReview> {
  return upsert(sessionId, { deleted_at: new Date().toISOString(), deleted_by: user });
}
