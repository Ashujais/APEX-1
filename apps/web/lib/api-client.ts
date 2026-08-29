export const API_URL = process.env.NEXT_PUBLIC_APEX_API_URL ?? 'http://localhost:8000';

let accessToken: string | null = null;

export function setAccessToken(token: string | null) {
  accessToken = token;
}

export function hasAccessToken() {
  return accessToken !== null;
}

export async function restoreBrowserSession(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/v1/auth/browser/refresh`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      accessToken = null;
      return false;
    }
    const payload = (await response.json()) as { access_token: string };
    accessToken = payload.access_token;
    return true;
  } catch {
    accessToken = null;
    return false;
  }
}

export async function apiFetch(path: string, init: RequestInit = {}, retry = true) {
  const headers = new Headers(init.headers);
  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`);
  if (init.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers,
    credentials: 'include',
  });
  if (response.status === 401 && retry && (await restoreBrowserSession())) {
    return apiFetch(path, init, false);
  }
  return response;
}
