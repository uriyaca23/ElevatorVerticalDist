import { createHmac, timingSafeEqual } from "node:crypto";

// Shared lab login. Hardcoded on purpose: this is a single team account, not
// per-user auth. Checked server-side only.
const USERNAME = "shaftAdmin";
const PASSWORD = "shaft12345";

export const SESSION_COOKIE = "shaft_session";
export const SESSION_MAX_AGE_SEC = 60 * 60 * 24 * 7;

function secret(): string {
  const s = process.env.AUTH_SECRET;
  if (!s) throw new Error("AUTH_SECRET must be set");
  return s;
}

function sign(payload: string): string {
  return createHmac("sha256", secret()).update(payload).digest("base64url");
}

function safeEqual(a: string, b: string): boolean {
  const ab = Buffer.from(a);
  const bb = Buffer.from(b);
  return ab.length === bb.length && timingSafeEqual(ab, bb);
}

export function checkCredentials(username: string, password: string): boolean {
  return safeEqual(username, USERNAME) && safeEqual(password, PASSWORD);
}

/** Cookie value `<username>.<expiry epoch s>.<hmac>`. */
export function createSessionToken(username: string): string {
  const payload = `${username}.${Math.floor(Date.now() / 1000) + SESSION_MAX_AGE_SEC}`;
  return `${payload}.${sign(payload)}`;
}

/** Returns the username for a valid, unexpired token, else null. */
export function verifySessionToken(token: string | undefined): string | null {
  if (!token) return null;
  const i = token.lastIndexOf(".");
  if (i < 0) return null;
  const payload = token.slice(0, i);
  if (!safeEqual(token.slice(i + 1), sign(payload))) return null;
  const [username, exp] = payload.split(".");
  if (!username || !(Number(exp) > Date.now() / 1000)) return null;
  return username;
}
