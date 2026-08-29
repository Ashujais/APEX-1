'use client';

import { ArrowLeft, KeyRound, Loader2, ShieldCheck } from 'lucide-react';
import { FormEvent, useState } from 'react';
import { useRouter } from 'next/navigation';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { API_URL, setAccessToken } from '@/lib/api-client';

type Mode = 'login' | 'register';

export default function AuthPage() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>('login');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === 'register') {
        const registration = await fetch(`${API_URL}/v1/auth/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, email, password }),
        });
        const registrationBody = (await registration.json()) as {
          detail?: string;
          development_verification_token?: string | null;
        };
        if (!registration.ok) throw new Error(registrationBody.detail ?? 'Registration failed');
        if (!registrationBody.development_verification_token) {
          throw new Error('Account created. Configure email delivery to receive the verification link.');
        }
        const verification = await fetch(`${API_URL}/v1/auth/verify-email`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token: registrationBody.development_verification_token }),
        });
        if (!verification.ok) throw new Error('Local email verification failed');
      }

      const login = await fetch(`${API_URL}/v1/auth/login`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      const loginBody = (await login.json()) as { access_token?: string; detail?: string };
      if (!login.ok || !loginBody.access_token) throw new Error(loginBody.detail ?? 'Sign in failed');
      setAccessToken(loginBody.access_token);
      router.push('/');
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Authentication failed');
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="grid min-h-dvh place-items-center bg-background px-4 py-12">
      <div className="w-full max-w-md">
        <Button variant="ghost" className="mb-6 gap-2" onClick={() => router.push('/')}>
          <ArrowLeft /> Back to workspace
        </Button>
        <section className="rounded-2xl border border-border bg-card p-6 shadow-[0_24px_70px_rgb(15_23_42/8%)] sm:p-8">
          <div className="flex items-center gap-3">
            <div className="apex-mark" aria-hidden="true">A1</div>
            <div>
              <h1 className="text-lg font-semibold">{mode === 'login' ? 'Open your workspace' : 'Create local account'}</h1>
              <p className="text-xs text-muted-foreground">APEX-1 development environment</p>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-2 rounded-xl bg-muted p-1">
            <Button variant={mode === 'login' ? 'secondary' : 'ghost'} onClick={() => setMode('login')}>Sign in</Button>
            <Button variant={mode === 'register' ? 'secondary' : 'ghost'} onClick={() => setMode('register')}>Register</Button>
          </div>

          <form className="mt-6 space-y-4" onSubmit={submit}>
            {mode === 'register' && (
              <label className="block text-xs font-medium">
                Name
                <Input className="mt-1.5" value={name} onChange={(event) => setName(event.target.value)} required minLength={1} maxLength={120} autoComplete="name" />
              </label>
            )}
            <label className="block text-xs font-medium">
              Email
              <Input className="mt-1.5" type="email" value={email} onChange={(event) => setEmail(event.target.value)} required autoComplete="email" />
            </label>
            <label className="block text-xs font-medium">
              Password
              <Input className="mt-1.5" type="password" value={password} onChange={(event) => setPassword(event.target.value)} required minLength={12} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} />
              {mode === 'register' && <span className="mt-1 block font-normal text-muted-foreground">Use at least 12 characters.</span>}
            </label>
            {error && <p role="alert" className="rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 text-xs text-destructive">{error}</p>}
            <Button className="h-10 w-full gap-2" type="submit" disabled={busy}>
              {busy ? <Loader2 className="animate-spin" /> : <KeyRound />}
              {mode === 'login' ? 'Sign in securely' : 'Create and verify account'}
            </Button>
          </form>

          <div className="mt-6 flex items-start gap-2 border-t border-border pt-4 text-[11px] leading-4 text-muted-foreground">
            <ShieldCheck className="mt-0.5 size-3.5 shrink-0" />
            Refresh credentials stay in an HTTP-only same-site cookie. Production still requires configured email delivery and TLS.
          </div>
        </section>
      </div>
    </main>
  );
}
