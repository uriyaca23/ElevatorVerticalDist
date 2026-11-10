import "server-only";
import { cert, getApps, initializeApp, type App } from "firebase-admin/app";
import { getStorage } from "firebase-admin/storage";

function app(): App {
  const existing = getApps()[0];
  if (existing) return existing;
  const raw = process.env.FIREBASE_SERVICE_ACCOUNT;
  if (!raw) throw new Error("FIREBASE_SERVICE_ACCOUNT must be set (service-account JSON)");
  return initializeApp({ credential: cert(JSON.parse(raw)) });
}

/** The Skydex recordings bucket (client reads are denied by storage.rules; admin SDK only). */
export function bucket() {
  const name = process.env.FIREBASE_BUCKET;
  if (!name) throw new Error("FIREBASE_BUCKET must be set");
  return getStorage(app()).bucket(name);
}
