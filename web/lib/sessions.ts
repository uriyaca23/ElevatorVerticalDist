import "server-only";
import { supabase } from "@/lib/clients/supabase";
import { getReview, listReviews } from "@/lib/review";
import type { SessionListItem, SessionRow } from "@/lib/types";

export async function listSessions(): Promise<SessionListItem[]> {
  const db = supabase();
  const { data, error } = await db.from("shaft_sessions").select("*").order("start_time", { ascending: false });
  if (error) throw new Error(`shaft_sessions: ${error.message}`);
  const [counts, reviews] = await Promise.all([labelCounts(), listReviews()]);
  return (data as SessionRow[])
    .filter((s) => !reviews.get(s.id)?.deleted_at)
    .map((s) => ({ ...s, label_count: counts.get(s.id) ?? 0, approved_at: reviews.get(s.id)?.approved_at ?? null }));
}

/** A session that exists and has not been deleted in the web app, else null. */
export async function getSession(id: number): Promise<SessionRow | null> {
  const [{ data, error }, review] = await Promise.all([
    supabase().from("shaft_sessions").select("*").eq("id", id).maybeSingle(),
    getReview(id),
  ]);
  if (error) throw new Error(`shaft_sessions: ${error.message}`);
  if (review.deleted_at) return null;
  return data as SessionRow | null;
}

// Before the shaft_labels migration is applied (or without the service key)
// the table is unreadable — the list still renders, just without counts.
async function labelCounts(): Promise<Map<number, number>> {
  const counts = new Map<number, number>();
  const { data, error } = await supabase().from("shaft_labels").select("session_id").retry(false);
  if (error) return counts;
  for (const { session_id } of data as { session_id: number }[]) {
    counts.set(session_id, (counts.get(session_id) ?? 0) + 1);
  }
  return counts;
}
