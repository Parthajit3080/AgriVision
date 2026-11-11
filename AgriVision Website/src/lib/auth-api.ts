const API = process.env.NEXT_PUBLIC_AGRIVISION_API ?? (process.env.NODE_ENV === "development" ? "http://localhost:8000" : "");

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

let csrfToken: string | null = null;

async function getCsrfToken(): Promise<string> {
  const response = await fetch(`${API}/api/auth/csrf`, { credentials: "include", cache: "no-store" });
  if (!response.ok) throw new ApiError("Could not start a secure session. Please try again.", response.status);
  const body = await response.json();
  if (typeof body.csrf_token !== "string") throw new ApiError("Could not start a secure session. Please try again.", response.status);
  csrfToken = body.csrf_token;
  return body.csrf_token as string;
}

export async function apiRequest<T = any>(path: string, options: RequestInit = {}, requestOptions: { csrf?: boolean } = {}): Promise<T> {
  const method = (options.method ?? "GET").toUpperCase();
  const headers = new Headers(options.headers);
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  if (requestOptions.csrf !== false && !new Set(["GET", "HEAD", "OPTIONS"]).has(method)) {
    const token = await getCsrfToken();
    headers.set("X-CSRF-Token", token);
  }
  const response = await fetch(`${API}/api${path}`, {
    ...options,
    method,
    headers,
    credentials: "include",
    cache: "no-store",
  });
  if (response.status === 204) return undefined as T;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 403 && String(body.detail ?? "").toLowerCase().includes("refresh")) csrfToken = null;
    throw new ApiError(body.detail ?? "Something went wrong. Please try again.", response.status);
  }
  return body as T;
}

export function friendlyError(error: unknown) {
  return error instanceof Error ? error.message : "Something went wrong. Please try again.";
}
