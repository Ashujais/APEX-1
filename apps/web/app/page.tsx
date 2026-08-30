'use client';

import {
  Bot,
  ChevronDown,
  CircleDot,
  Database,
  FileText,
  FolderKanban,
  Library,
  Mic,
  MoreHorizontal,
  Paperclip,
  PanelLeft,
  Plus,
  Search,
  Send,
  Settings,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import Link from 'next/link';
import { type SyntheticEvent, useEffect, useState } from 'react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { apiFetch, hasAccessToken, restoreBrowserSession } from '@/lib/api-client';

const navigation = [
  { label: 'New chat', icon: Plus, active: true },
  { label: 'Search', icon: Search },
  { label: 'Projects', icon: FolderKanban },
  { label: 'Agents', icon: Bot },
  { label: 'Library', icon: Library },
];

const recentChats = [
  'Training run diagnostics',
  'RAG architecture review',
  'Tokenizer vocabulary notes',
];

type ApiStatus = 'checking' | 'signed-in' | 'signed-out' | 'offline';
type LiveMessage = { id: string; role: 'user' | 'assistant'; content: string };

export default function Home() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>(() =>
    hasAccessToken() ? 'signed-in' : 'checking',
  );
  const [prompt, setPrompt] = useState('');
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [liveMessages, setLiveMessages] = useState<LiveMessage[]>([]);
  const [sending, setSending] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (hasAccessToken()) return;
    restoreBrowserSession()
      .then((restored) => setApiStatus(restored ? 'signed-in' : 'signed-out'))
      .catch(() => setApiStatus('offline'));
  }, []);

  async function sendMessage(event: SyntheticEvent<HTMLFormElement, SubmitEvent>) {
    event.preventDefault();
    const text = prompt.trim();
    if (!text || sending) return;
    if (apiStatus !== 'signed-in') {
      setNotice('Sign in to send a request through the authenticated APEX API.');
      return;
    }

    const userMessage: LiveMessage = { id: crypto.randomUUID(), role: 'user', content: text };
    const assistantId = crypto.randomUUID();
    setLiveMessages((current) => [
      ...current,
      userMessage,
      { id: assistantId, role: 'assistant', content: '' },
    ]);
    setPrompt('');
    setNotice(null);
    setSending(true);

    try {
      let activeConversation = conversationId;
      if (!activeConversation) {
        const created = await apiFetch('/v1/conversations', {
          method: 'POST',
          body: JSON.stringify({ title: text.slice(0, 80), model_id: 'apex-dev' }),
        });
        if (!created.ok) throw new Error('Could not create the conversation');
        activeConversation = ((await created.json()) as { id: string }).id;
        setConversationId(activeConversation);
      }

      const response = await apiFetch(`/v1/conversations/${activeConversation}/stream`, {
        method: 'POST',
        body: JSON.stringify({ prompt: text }),
      });
      if (!response.ok || !response.body) throw new Error('The streaming API is unavailable');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() ?? '';
        for (const block of blocks) {
          if (!block.startsWith('event: delta')) continue;
          const dataLine = block.split('\n').find((line) => line.startsWith('data: '));
          if (!dataLine) continue;
          const payload = JSON.parse(dataLine.slice(6)) as { text: string };
          setLiveMessages((current) =>
            current.map((liveMessage) =>
              liveMessage.id === assistantId
                ? { ...liveMessage, content: liveMessage.content + payload.text }
                : liveMessage,
            ),
          );
        }
        if (done) break;
      }
    } catch (problem) {
      setLiveMessages(
        replaceMessageContent(
          assistantId,
          `Request failed: ${problem instanceof Error ? problem.message : 'Request failed'}`,
        ),
      );
    } finally {
      setSending(false);
    }
  }

  return (
    <main className="apex-shell">
      <aside className="apex-sidebar">
        <div className="flex h-16 items-center justify-between border-b border-sidebar-border px-4">
          <div className="flex items-center gap-2.5">
            <div className="apex-mark" aria-hidden="true">A1</div>
            <div>
              <p className="text-sm font-semibold tracking-[-0.02em]">APEX-1</p>
              <p className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Research system</p>
            </div>
          </div>
          <Button variant="ghost" size="icon-sm" aria-label="Collapse sidebar"><PanelLeft /></Button>
        </div>

        <nav className="px-3 py-4" aria-label="Primary navigation">
          <div className="space-y-1">
            {navigation.map(({ label, icon: Icon, active }) => (
              <Button key={label} variant={active ? 'secondary' : 'ghost'} className="h-9 w-full justify-start gap-2.5 px-3">
                <Icon />{label}
              </Button>
            ))}
          </div>
          <div className="mt-7 px-3">
            <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Recent</p>
          </div>
          <div className="mt-2 space-y-0.5">
            {recentChats.map((chat) => (
              <Button key={chat} variant="ghost" className="h-8 w-full justify-start overflow-hidden px-3 text-xs font-normal text-muted-foreground">
                <span className="truncate">{chat}</span>
              </Button>
            ))}
          </div>
        </nav>

        <div className="mt-auto border-t border-sidebar-border p-3">
          <div className="rounded-xl border border-border bg-card/70 p-3">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium">Local runtime</span>
              <Badge variant="outline" className="gap-1 border-emerald-500/30 text-emerald-600 dark:text-emerald-400"><CircleDot className="size-2.5" /> {statusLabel(apiStatus)}</Badge>
            </div>
            <p className="mt-2 text-[11px] leading-4 text-muted-foreground">CPU development mode · no discrete GPU detected</p>
          </div>
          <Button variant="ghost" className="mt-2 h-9 w-full justify-start gap-2.5 px-3"><Settings /> Settings</Button>
        </div>
      </aside>

      <section className="apex-workspace">
        <header className="flex h-16 items-center justify-between border-b border-border/80 px-4 sm:px-6">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="truncate text-sm font-semibold">Architecture copilot</h1>
              <Badge variant="secondary" className="hidden sm:inline-flex">Local</Badge>
            </div>
            <p className="mt-0.5 text-[11px] text-muted-foreground">Project · APEX core</p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" className="hidden gap-2 sm:inline-flex" onClick={() => { window.location.href = '/auth'; }}><ShieldCheck /> {apiStatus === 'signed-in' ? 'Session active' : 'Sign in'}</Button>
            <Button variant="ghost" size="icon" aria-label="Conversation menu"><MoreHorizontal /></Button>
          </div>
        </header>

        <div className="apex-chat-scroll">
          <div className="mx-auto w-full max-w-3xl px-4 py-8 sm:px-8 sm:py-12">
            <div className="mb-8 flex items-center gap-3">
              <div className="apex-orbit" aria-hidden="true"><Sparkles className="size-4" /></div>
              <div>
                <p className="text-xs font-medium">APEX Standard</p>
                <p className="text-[11px] text-muted-foreground">Development runtime · verified capabilities only</p>
              </div>
            </div>

            <article className="space-y-7" aria-label="Conversation">
              <div className="ml-auto max-w-[86%] rounded-2xl rounded-tr-md bg-primary px-4 py-3 text-sm leading-6 text-primary-foreground">
                Review the first implementation milestone and tell me what this machine can realistically run.
              </div>

              <div className="grid grid-cols-[28px_minmax(0,1fr)] gap-3">
                <div className="apex-response-mark" aria-hidden="true">A1</div>
                <div className="space-y-4 text-sm leading-6">
                  <p>
                    This environment is ready for the platform foundation and CPU-scale pipeline validation. I detected a 2-core Intel CPU, 3.8 GB RAM, and integrated graphics, so the honest starting point is a tiny Transformer test configuration—not a production model training run.
                  </p>
                  <div className="grid gap-3 sm:grid-cols-3">
                    <CapabilityCard label="Platform" value="Ready" detail="Web + API foundation" />
                    <CapabilityCard label="Model lab" value="Limited" detail="Tiny CPU smoke tests" />
                    <CapabilityCard label="Frontier training" value="Blocked" detail="GPU cluster required" />
                  </div>
                  <div className="flex flex-wrap gap-2 pt-1">
                    <Badge variant="outline"><Database /> Local data</Badge>
                    <Badge variant="outline"><ShieldCheck /> Tenant scoped</Badge>
                    <Badge variant="outline"><FileText /> Status documented</Badge>
                  </div>
                </div>
              </div>

              {liveMessages.map((message) =>
                message.role === 'user' ? (
                  <div key={message.id} className="ml-auto max-w-[86%] rounded-2xl rounded-tr-md bg-primary px-4 py-3 text-sm leading-6 text-primary-foreground">
                    {message.content}
                  </div>
                ) : (
                  <div key={message.id} className="grid grid-cols-[28px_minmax(0,1fr)] gap-3">
                    <div className="apex-response-mark" aria-hidden="true">A1</div>
                    <div className="text-sm leading-6">
                      {message.content || <span className="text-muted-foreground">Streaming verified development output…</span>}
                    </div>
                  </div>
                ),
              )}
            </article>
          </div>
        </div>

        <div className="apex-composer-wrap">
          <div className="mx-auto w-full max-w-3xl px-4 pb-4 sm:px-8 sm:pb-6">
            <form className="apex-composer" onSubmit={sendMessage}>
              <Textarea aria-label="Message APEX-1" placeholder={apiStatus === 'signed-in' ? 'Test the authenticated APEX pipeline…' : 'Sign in to test the authenticated pipeline…'} value={prompt} onChange={(event) => setPrompt(event.target.value)} className="min-h-12 resize-none border-0 bg-transparent px-3 py-3 shadow-none focus-visible:ring-0" />
              <div className="flex items-center justify-between gap-3 px-2 pb-2">
                <div className="flex items-center gap-1">
                  <Button type="button" variant="ghost" size="icon-sm" aria-label="Attach files"><Paperclip /></Button>
                  <Button type="button" variant="ghost" size="icon-sm" aria-label="Start voice input" disabled><Mic /></Button>
                  <Button type="button" variant="ghost" size="sm" className="gap-1.5 text-xs">APEX Standard <ChevronDown /></Button>
                </div>
                <Button type="submit" size="icon" aria-label="Send message" className="rounded-xl" disabled={sending}><Send /></Button>
              </div>
            </form>
            {notice && <output className="mt-2 block text-center text-[11px] text-destructive">{notice} <Link className="underline underline-offset-2" href="/auth">Open sign in</Link></output>}
            <p className="mt-2 text-center text-[10px] text-muted-foreground">Experimental research platform. Outputs may be incorrect; verified states are labeled explicitly.</p>
          </div>
        </div>
      </section>
    </main>
  );
}

function statusLabel(status: ApiStatus) {
  if (status === 'signed-in') return 'API linked';
  if (status === 'signed-out') return 'Sign-in required';
  if (status === 'offline') return 'API offline';
  return 'Checking';
}

function replaceMessageContent(messageId: string, content: string) {
  return (current: LiveMessage[]) =>
    current.map((item) => (item.id === messageId ? { ...item, content } : item));
}

function CapabilityCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="apex-capability-card">
      <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">{label}</p>
      <p className="mt-2 text-sm font-semibold">{value}</p>
      <p className="mt-0.5 text-[11px] leading-4 text-muted-foreground">{detail}</p>
    </div>
  );
}
