import { createPinia, setActivePinia } from 'pinia';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useAuthStore } from './auth';
import { useChatStore } from './chat';
import { useSessionStore } from './sessions';
import api from '@/utils/api';

vi.mock('@/utils/api', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
  },
  // The store now builds its stream URL through this. Mocked as the identity function,
  // which is what an unconfigured VITE_API_BASE_URL produces — so every existing
  // assertion in this file still sees the same '/chat/stream' it always did.
  apiUrl: (path: string) => path,
  API_BASE_URL: '',
}));

type PendingRead = {
  resolve: (value: ReadableStreamReadResult<Uint8Array>) => void;
  reject: (reason?: unknown) => void;
};

const flushPromises = () => new Promise((resolve) => setTimeout(resolve, 0));

/** A timetable answer as `_settle_answer_blocks` builds it: prose, marker, record. */
const SETTLED_TIMETABLE = 'دي جدولها:\n\n<!--record-block-->\n**الأحد**\n1) الكيمياء · 07:45–08:30';
const WEEK_BLOCK = {
  kind: 'timetable',
  index: 0,
  language: 'ar',
  data: {
    periods: [{ number: 1, starts_at: '07:45', ends_at: '08:30' }],
    days: [{ day: 'sunday', label: 'الأحد', slots: [{ period: 1, subject: 'الكيمياء' }] }],
  },
};

const createLocalStorageMock = () => {
  const store = new Map<string, string>();
  return {
    getItem: vi.fn((key: string) => store.get(key) || null),
    setItem: vi.fn((key: string, value: string) => {
      store.set(key, value);
    }),
    removeItem: vi.fn((key: string) => {
      store.delete(key);
    }),
    clear: vi.fn(() => {
      store.clear();
    }),
  };
};

const createAbortError = () => {
  if (typeof DOMException !== 'undefined') {
    return new DOMException('The operation was aborted.', 'AbortError');
  }
  const error = new Error('The operation was aborted.');
  error.name = 'AbortError';
  return error;
};

const createControlledSseFetch = () => {
  const encoder = new TextEncoder();
  const chunks: Uint8Array[] = [];
  const pendingReads: PendingRead[] = [];
  let closed = false;

  const reader = {
    read: vi.fn(() => {
      if (chunks.length) {
        return Promise.resolve({ done: false, value: chunks.shift() });
      }
      if (closed) {
        return Promise.resolve({ done: true, value: undefined });
      }
      return new Promise<ReadableStreamReadResult<Uint8Array>>((resolve, reject) => {
        pendingReads.push({ resolve, reject });
      });
    }),
  };

  const resolveNextRead = (value: ReadableStreamReadResult<Uint8Array>) => {
    const pending = pendingReads.shift();
    if (pending) {
      pending.resolve(value);
    } else if (!value.done && value.value) {
      chunks.push(value.value);
    }
  };

  const fetchMock = vi.fn((_url: string, init?: RequestInit) => {
    init?.signal?.addEventListener('abort', () => {
      closed = true;
      const abortError = createAbortError();
      pendingReads.splice(0).forEach((pending) => pending.reject(abortError));
    });

    return Promise.resolve({
      ok: true,
      status: 200,
      body: {
        getReader: () => reader,
      },
    } as unknown as Response);
  });

  return {
    fetchMock,
    pushEvent(event: object) {
      resolveNextRead({
        done: false,
        value: encoder.encode(`data: ${JSON.stringify(event)}\n\n`),
      });
    },
    // The `[DONE]` sentinel is not JSON, so it cannot go through `pushEvent`. It marks
    // the answer as complete while the connection stays OPEN — which is exactly the
    // window the server uses to confirm the turn is stored, and the state the composer
    // must already be released in.
    pushDone() {
      resolveNextRead({ done: false, value: encoder.encode('data: [DONE]\n\n') });
    },
    close() {
      closed = true;
      resolveNextRead({ done: true, value: undefined });
    },
  };
};

/** A message as `/sessions/{id}` returns it: stored, and so carrying its row id. */
const serverPageMessage = (id: number, content: string, extra: Record<string, unknown> = {}) => ({
  id,
  type: id % 2 === 1 ? 'human' : 'ai',
  content,
  timestamp: '2026-09-14T00:00:00',
  ...extra,
});

const setupStores = () => {
  setActivePinia(createPinia());

  const authStore = useAuthStore();
  authStore.token = 'test-token';
  authStore.currentUser = { username: 'tester', role: 'user' };

  const chatStore = useChatStore();
  chatStore.setViewedSession('session_current', []);

  return {
    authStore,
    chatStore,
    sessionStore: useSessionStore(),
  };
};

describe('chat store streaming sessions', () => {
  beforeEach(() => {
    // Restoring puts spies back; it no longer empties a `vi.fn()`'s call history (Vitest 3+),
    // so the shared `api` mock is cleared too — a test that counts calls starts from zero.
    vi.restoreAllMocks();
    vi.clearAllMocks();
    vi.stubGlobal('localStorage', createLocalStorageMock());
    vi.stubGlobal('alert', vi.fn());
    vi.stubGlobal('confirm', vi.fn(() => true));
  });

  it('clears account-scoped chat state when the authenticated workspace changes', () => {
    const { chatStore } = setupStores();
    const previousSessionId = chatStore.sessionId;

    chatStore.messagesBySession.session_current = [
      { text: "Previous account's message", isUser: true },
    ];
    chatStore.messages = chatStore.messagesBySession.session_current;
    chatStore.userInput = 'Unsent draft';
    chatStore.activeNav = 'settings';
    chatStore.pendingHitlBySession.session_current = {
      prompt: 'Please provide more information',
      options: [],
    };

    chatStore.resetWorkspace();

    expect(chatStore.messages).toEqual([]);
    expect(chatStore.messagesBySession).toEqual({});
    expect(chatStore.userInput).toBe('');
    expect(chatStore.activeNav).toBe('newChat');
    expect(chatStore.pendingHitlBySession).toEqual({});
    expect(chatStore.sessionId).not.toBe(previousSessionId);
  });

  it('creates a local history session with the user message and thinking placeholder immediately', async () => {
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);
    const { chatStore, sessionStore } = setupStores();

    chatStore.userInput = 'Help me summarize the document';
    const sendPromise = chatStore.handleSend();
    await flushPromises();

    expect(sessionStore.sessions[0]).toMatchObject({
      session_id: 'session_current',
      isStreaming: true,
    });
    expect(chatStore.messagesBySession.session_current).toHaveLength(2);
    expect(chatStore.messagesBySession.session_current[0]).toMatchObject({
      text: 'Help me summarize the document',
      isUser: true,
    });
    expect(chatStore.messagesBySession.session_current[1]).toMatchObject({
      text: '',
      isUser: false,
      isThinking: true,
    });

    stream.close();
    await sendPromise;
  });

  it('keeps streaming chunks on the originating session after viewing another history session', async () => {
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);
    vi.mocked(api.get).mockResolvedValue({
      data: {
        messages: [
          {
            type: 'human',
            content: 'Old question',
            timestamp: '2026-07-08T00:00:00',
          },
          {
            type: 'ai',
            content: 'Old answer',
            timestamp: '2026-07-08T00:00:01',
          },
        ],
      },
    });

    const { chatStore } = setupStores();
    chatStore.userInput = 'New question';
    const sendPromise = chatStore.handleSend();
    await flushPromises();

    await chatStore.loadSession('session_old');
    expect(chatStore.sessionId).toBe('session_old');
    expect(chatStore.messages.map((msg) => msg.text)).toEqual(['Old question', 'Old answer']);

    stream.pushEvent({ type: 'rag_step', step: { label: 'Retrieving', group: null } });
    await flushPromises();

    stream.pushEvent({ type: 'content', content: 'Answering' });
    await flushPromises();

    expect(chatStore.messagesBySession.session_current[1]).toMatchObject({
      text: 'Answering',
      isThinking: false,
    });
    expect(chatStore.messagesBySession.session_current[1].ragSteps?.[0]).toMatchObject({
      label: 'Retrieving',
    });
    expect(chatStore.messages.map((msg) => msg.text)).toEqual(['Old question', 'Old answer']);

    vi.mocked(api.get).mockClear();
    await chatStore.loadSession('session_current');

    expect(api.get).not.toHaveBeenCalled();
    expect(chatStore.sessionId).toBe('session_current');
    expect(chatStore.messages[1]).toMatchObject({
      text: 'Answering',
      isThinking: false,
    });

    stream.close();
    await sendPromise;
  });

  it('writes abort state only to the streaming session', async () => {
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);
    vi.mocked(api.get).mockResolvedValue({
      data: {
        messages: [
          {
            type: 'human',
            content: 'Another session',
            timestamp: '2026-07-08T00:00:00',
          },
        ],
      },
    });

    const { chatStore } = setupStores();
    chatStore.userInput = 'A question that will be stopped';
    const sendPromise = chatStore.handleSend();
    await flushPromises();

    await chatStore.loadSession('session_other');
    chatStore.handleStop();
    await sendPromise;

    expect(chatStore.messagesBySession.session_current[1]).toMatchObject({
      text: '(Response stopped)',
      isThinking: false,
    });
    expect(chatStore.messagesBySession.session_other.map((msg) => msg.text)).toEqual([
      'Another session',
    ]);
    expect(chatStore.sessionId).toBe('session_other');
    expect(chatStore.isLoading).toBe(false);
    expect(chatStore.streamingSessionId).toBeNull();
  });

  it('releases the composer at [DONE] rather than at connection close', async () => {
    // The server keeps the stream open past `[DONE]` until the turn is stored. Waiting
    // for the close left the send button disabled after the last word had rendered.
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);
    const { chatStore } = setupStores();

    chatStore.userInput = 'Who teaches her maths?';
    const sendPromise = chatStore.handleSend();
    await flushPromises();

    stream.pushEvent({ type: 'content', content: 'Mr Hany teaches her maths.' });
    await flushPromises();
    expect(chatStore.isLoading).toBe(true);

    stream.pushDone();
    await flushPromises();

    // Released, with the connection still open and no `done` read yet.
    expect(chatStore.isLoading).toBe(false);
    expect(chatStore.streamingSessionId).toBeNull();
    expect(chatStore.messagesBySession.session_current[1]).toMatchObject({
      text: 'Mr Hany teaches her maths.',
      isThinking: false,
    });

    // Events that arrive in the window after `[DONE]` are still applied.
    stream.pushEvent({ type: 'trace', rag_trace: { turn_forced_tool: 'get_student_teachers' } });
    await flushPromises();
    expect(chatStore.messagesBySession.session_current[1].ragTrace).toMatchObject({
      turn_forced_tool: 'get_student_teachers',
    });

    stream.close();
    await sendPromise;
    expect(chatStore.isLoading).toBe(false);
  });

  it('keys the turn on the rows the server stored it under', async () => {
    // The `stored` event follows `[DONE]`. From then on this tab's copy and the server's
    // are one message, which is what reopening the chat relies on.
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);
    const { chatStore } = setupStores();

    chatStore.userInput = 'When does the bus leave?';
    const sendPromise = chatStore.handleSend();
    await flushPromises();
    stream.pushEvent({ type: 'content', content: 'At 07:30.' });
    stream.pushDone();
    stream.pushEvent({ type: 'stored', message_ids: [41, 42] });
    stream.close();
    await sendPromise;

    const [question, answer] = chatStore.messagesBySession.session_current;
    expect(question).toMatchObject({ id: 41, isUser: true });
    expect(answer).toMatchObject({ id: 42, text: 'At 07:30.' });
    expect(question.unconfirmed).toBeUndefined();
    expect(answer.unconfirmed).toBeUndefined();
  });

  it('keeps a turn whose save has not landed when the chat is reopened, then knows it once stored', async () => {
    // The composer is released at `[DONE]`; the save follows by milliseconds. A parent who
    // switches chats and back inside that window used to see the answer they had just
    // read disappear, replaced by a page from the server that did not hold it yet.
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);
    const { chatStore } = setupStores();
    const older = [serverPageMessage(31, 'Earlier question'), serverPageMessage(32, 'Earlier answer')];

    chatStore.userInput = 'When does the bus leave?';
    const sendPromise = chatStore.handleSend();
    await flushPromises();
    stream.pushEvent({ type: 'content', content: 'At 07:30.' });
    stream.pushDone();
    await flushPromises();
    expect(chatStore.isLoading).toBe(false);

    vi.mocked(api.get).mockResolvedValueOnce({ data: { messages: older, has_more: false } });
    await chatStore.loadSession('session_current');
    expect(chatStore.messages.map((msg) => msg.text)).toEqual([
      'Earlier question',
      'Earlier answer',
      'When does the bus leave?',
      'At 07:30.',
    ]);

    stream.pushEvent({ type: 'stored', message_ids: [41, 42] });
    stream.close();
    await sendPromise;
    expect(chatStore.messages[3]).toMatchObject({ id: 42 });

    // The server now returns the turn; the reconciled list holds it once, not twice.
    vi.mocked(api.get).mockResolvedValueOnce({
      data: {
        messages: [...older, serverPageMessage(41, 'When does the bus leave?'), serverPageMessage(42, 'At 07:30.')],
        has_more: false,
      },
    });
    await chatStore.loadSession('session_current');
    expect(chatStore.messages.map((msg) => msg.id)).toEqual([31, 32, 41, 42]);
  });

  it('yields to the server for a turn the stream never confirmed', async () => {
    // Stop was pressed. The server stores what had reached the parent, marked as
    // interrupted; this tab's copy must not appear beside it on reopen.
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);
    const { chatStore } = setupStores();

    chatStore.userInput = 'What are the fees?';
    const sendPromise = chatStore.handleSend();
    await flushPromises();
    stream.pushEvent({ type: 'content', content: 'The fees for Year 1 are ' });
    await flushPromises();
    chatStore.handleStop();
    await sendPromise;
    expect(chatStore.messages[1].text).toBe('The fees for Year 1 are \n\n_(Response was stopped)_');

    vi.mocked(api.get).mockResolvedValueOnce({
      data: {
        messages: [
          serverPageMessage(51, 'What are the fees?'),
          serverPageMessage(52, 'The fees for Year 1 are ', { rag_trace: { turn_interrupted: true } }),
        ],
        has_more: false,
      },
    });
    await chatStore.loadSession('session_current');

    expect(chatStore.messages).toHaveLength(2);
    expect(chatStore.messages[1]).toMatchObject({
      id: 52,
      text: 'The fees for Year 1 are \n\n_(Response was stopped)_',
    });
  });

  it('uploads a voice note, queues it with its transcript, and sends the two together', async () => {
    // The transcript is the message; the note goes with it so the server keeps them as one
    // turn and the recording comes back on every device.
    vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:local-note'), revokeObjectURL: vi.fn() });
    vi.mocked(api.post).mockResolvedValueOnce({
      data: {
        id: 'note-1', kind: 'voice', url: '/chat/attachments/note-1', content_type: 'audio/webm',
        byte_size: 1200, duration_ms: 4200, transcript: 'إمتى الباص بييجي؟', transcript_status: 'ok',
      },
    });
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);
    const { chatStore } = setupStores();

    const attachment = await chatStore.attachVoiceNote(new Blob(['audio'], { type: 'audio/webm' }), 4.2);
    expect(attachment.transcript_status).toBe('ok');
    expect(api.post).toHaveBeenCalledWith('/chat/attachments', expect.any(FormData), expect.anything());
    expect(chatStore.pendingVoice).toMatchObject({ url: 'blob:local-note', attachmentId: 'note-1', duration: 4.2 });

    chatStore.userInput = attachment.transcript!;
    const sendPromise = chatStore.handleSend();
    await flushPromises();
    stream.close();
    await sendPromise;

    const body = JSON.parse((stream.fetchMock.mock.calls[0][1] as RequestInit).body as string);
    expect(body).toMatchObject({ message: 'إمتى الباص بييجي؟', attachment_id: 'note-1' });
    expect(chatStore.messagesBySession.session_current[0]).toMatchObject({
      isUser: true,
      text: 'إمتى الباص بييجي؟',
      voice: { attachmentId: 'note-1', url: 'blob:local-note' },
    });
    expect(chatStore.pendingVoice).toBeNull();
  });

  it('does not queue a note nobody could transcribe', async () => {
    vi.mocked(api.post).mockResolvedValueOnce({
      data: {
        id: 'note-2', kind: 'voice', url: '/chat/attachments/note-2', content_type: 'audio/webm',
        byte_size: 1200, duration_ms: 4200, transcript: null, transcript_status: 'unavailable',
      },
    });
    const { chatStore } = setupStores();

    const attachment = await chatStore.attachVoiceNote(new Blob(['audio']), 4.2);

    expect(attachment.transcript_status).toBe('unavailable');
    expect(chatStore.pendingVoice).toBeNull();
  });

  it('restores a voice note from the server copy of the message', () => {
    const { chatStore } = setupStores();
    const [question] = chatStore.mapServerMessages([
      serverPageMessage(41, 'إمتى الباص بييجي؟', {
        attachment: {
          id: 'note-1', kind: 'voice', url: '/chat/attachments/note-1', content_type: 'audio/webm',
          byte_size: 1200, duration_ms: 4200, transcript: 'إمتى الباص بييجي؟', transcript_status: 'ok',
        },
      }),
    ]);

    expect(question.voice).toEqual({
      url: '/chat/attachments/note-1',
      duration: 4.2,
      mimeType: 'audio/webm',
      attachmentId: 'note-1',
      transcript: 'إمتى الباص بييجي؟',
    });
    expect(question.text).toBe('إمتى الباص بييجي؟');
  });

  it('does not let a drained stream clear a newer request state', async () => {
    // Releasing the composer early means a next message can start while the previous
    // connection is still draining. That stream's teardown must not clear the state the
    // newer request owns, or the send button would re-enable mid-answer.
    const first = createControlledSseFetch();
    vi.stubGlobal('fetch', first.fetchMock);
    const { chatStore } = setupStores();

    chatStore.userInput = 'What is her timetable?';
    const firstSend = chatStore.handleSend();
    await flushPromises();
    first.pushDone();
    await flushPromises();
    expect(chatStore.isLoading).toBe(false);

    const second = createControlledSseFetch();
    vi.stubGlobal('fetch', second.fetchMock);
    chatStore.userInput = 'And who teaches her?';
    const secondSend = chatStore.handleSend();
    await flushPromises();
    expect(chatStore.isLoading).toBe(true);

    // The first connection only closes now, after the second is already streaming.
    first.close();
    await firstSend;
    expect(chatStore.isLoading).toBe(true);
    expect(chatStore.streamingSessionId).toBe('session_current');

    second.pushDone();
    second.close();
    await secondSend;
    expect(chatStore.isLoading).toBe(false);
  });

  it('turns hitl_request events into a pending HITL prompt', async () => {
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);
    const { chatStore } = setupStores();

    chatStore.userInput = "What are this character's attributes?";
    const sendPromise = chatStore.handleSend();
    await flushPromises();

    stream.pushEvent({
      type: 'trace',
      rag_trace: {
        retrieval_status: 'needs_clarification',
        route: 'clarify',
        hitl_prompt: 'Please provide the character name',
        hitl_options: ['Danjin', 'Dan Heng'],
      },
    });
    await flushPromises();

    stream.pushEvent({
      type: 'hitl_request',
      hitl: {
        id: 'hitl-1',
        prompt: 'Please provide the character name',
        options: ['Danjin', 'Dan Heng'],
        route: 'clarify',
        retrieval_status: 'needs_clarification',
        original_question: "What are this character's attributes?",
      },
    });
    stream.close();
    await sendPromise;

    expect(chatStore.messagesBySession.session_current[1]).toMatchObject({
      isThinking: false,
      isHitlRequest: true,
      hitlPrompt: 'Please provide the character name',
      hitlOptions: ['Danjin', 'Dan Heng'],
    });
    expect(chatStore.pendingHitlBySession.session_current).toMatchObject({
      prompt: 'Please provide the character name',
      options: ['Danjin', 'Dan Heng'],
    });
    expect(chatStore.inputPlaceholder).toBe('Type your own answer, or pick an option above and send...');
  });

  it('marks the next user message as a HITL answer and clears pending state after content streams', async () => {
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);
    const { chatStore } = setupStores();
    chatStore.pendingHitlBySession.session_current = {
      id: 'hitl-1',
      prompt: 'Please provide the character name',
      options: ['Danjin'],
    };

    chatStore.userInput = 'Danjin';
    const sendPromise = chatStore.handleSend();
    await flushPromises();

    expect(chatStore.messagesBySession.session_current[0]).toMatchObject({
      text: 'Danjin',
      isUser: true,
      isHitlAnswer: true,
    });
    expect(chatStore.messagesBySession.session_current[1]).toMatchObject({
      isUser: false,
      hitlResumeText: 'Danjin',
    });
    expect(chatStore.pendingHitlBySession.session_current).toBeUndefined();

    stream.pushEvent({ type: 'content', content: 'Danjin has the Nihility element.' });
    stream.close();
    await sendPromise;

    expect(chatStore.messagesBySession.session_current[1]).toMatchObject({
      text: 'Danjin has the Nihility element.',
      isThinking: false,
      hitlResumeText: 'Danjin',
    });
    expect(chatStore.pendingHitlBySession.session_current).toBeUndefined();
  });

  it('shows a message as typed when the server reads it as a new question, not an answer', async () => {
    // Asked "which child?", the parent asks about the bus instead. The message was
    // marked as an answer at send time; the server's `turn` event says otherwise.
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);
    const { chatStore } = setupStores();
    chatStore.pendingHitlBySession.session_current = {
      id: 'which-child',
      prompt: 'Which child do you mean?',
      options: ['Fatma — Year 11', 'Fatma — Year 9'],
      route: 'child_select',
    };

    chatStore.userInput = 'When does the bus come?';
    const sendPromise = chatStore.handleSend();
    await flushPromises();
    expect(chatStore.messagesBySession.session_current[0].isHitlAnswer).toBe(true);

    stream.pushEvent({ type: 'turn', answers_clarification: false });
    stream.pushEvent({ type: 'content', content: 'At 07:30.' });
    stream.close();
    await sendPromise;

    const [question, answer] = chatStore.messagesBySession.session_current;
    expect(question).toMatchObject({ text: 'When does the bus come?', isUser: true, isHitlAnswer: false });
    expect(answer.hitlResumeText).toBeUndefined();
    expect(answer.text).toBe('At 07:30.');
  });

  it('shows a stored question that replaced a clarification as a message on reload', () => {
    const { chatStore } = setupStores();
    const messages = chatStore.mapServerMessages([
      { type: 'human', content: 'What are her subjects?' },
      {
        type: 'ai',
        content: 'Which child do you mean?',
        rag_trace: { retrieval_status: 'needs_child_choice', route: 'child_select', hitl_prompt: 'Which child do you mean?' },
      },
      { type: 'human', content: 'When does the bus come?' },
      { type: 'ai', content: 'At 07:30.', rag_trace: { turn_clarification: 'replaced' } },
    ]);

    // `needs_child_choice` is not a retrieval clarification, so message 1 is not a HITL
    // request here; the rule under test is the one on messages 2 and 3.
    expect(messages[2]).toMatchObject({ isUser: true, isHitlAnswer: false });
    expect(messages[3].hitlResumeText).toBeUndefined();

    const retrieval = chatStore.mapServerMessages([
      { type: 'human', content: 'What are the fees?' },
      { type: 'ai', content: 'Which year group?', rag_trace: { retrieval_status: 'needs_clarification', route: 'clarify' } },
      { type: 'human', content: 'Who is the principal?' },
      { type: 'ai', content: 'Mr Hany.', rag_trace: { turn_clarification: 'replaced' } },
    ]);
    expect(retrieval[2]).toMatchObject({ text: 'Who is the principal?', isHitlAnswer: false });
    expect(retrieval[3].hitlResumeText).toBeUndefined();
  });

  it('maps persisted HITL answer turns as continuation state instead of normal chat turns', () => {
    const { chatStore } = setupStores();

    const messages = chatStore.mapServerMessages([
      { type: 'human', content: "What are this character's attributes?" },
      {
        type: 'ai',
        content: 'Please provide the character name',
        rag_trace: {
          retrieval_status: 'needs_clarification',
          route: 'clarify',
          hitl_prompt: 'Please provide the character name',
        },
      },
      { type: 'human', content: 'Danjin' },
      { type: 'ai', content: 'Danjin has the Nihility element.' },
    ]);

    expect(messages[1]).toMatchObject({ isHitlRequest: true });
    expect(messages[2]).toMatchObject({ isHitlAnswer: true });
    expect(messages[3]).toMatchObject({
      text: 'Danjin has the Nihility element.',
      hitlResumeText: 'Danjin',
    });
  });

  // An answer can be streamed and then fail verification against the evidence it
  // claimed — a fee figure that is in none of the retrieved chunks. The server sends
  // `content_replace`, and the whole point is that it REPLACES: a correction appended
  // underneath would leave the unverified number on screen next to the retraction,
  // which is worse than either alone.
  it('replaces a streamed answer when the server retracts an unverified figure', async () => {
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);

    const { chatStore } = setupStores();
    chatStore.userInput = 'مصاريف ابني كام';
    const sendPromise = chatStore.handleSend();
    await flushPromises();

    stream.pushEvent({ type: 'content', content: 'مصاريف ابنك 45 ألف جنيه' });
    await flushPromises();
    expect(chatStore.messagesBySession.session_current[1]).toMatchObject({
      text: 'مصاريف ابنك 45 ألف جنيه',
    });

    stream.pushEvent({
      type: 'content_replace',
      content: 'معلش، مقدرتش أتأكد من الأرقام دي من مستندات المدرسة.',
    });
    stream.close();
    await sendPromise;

    const bubble = chatStore.messagesBySession.session_current[1];
    expect(bubble.text).toBe('معلش، مقدرتش أتأكد من الأرقام دي من مستندات المدرسة.');
    expect(bubble.text).not.toContain('45');
    expect(bubble.isThinking).toBe(false);
  });

  // A timetable reaches the bubble as text with a record marker in it, and — one event
  // earlier — as data naming that marker. The data is what the bubble draws; a kind this
  // client cannot draw is left out, and its marker prints as the markdown it came with.
  it('holds the records the stream sends ahead of the text that places them', async () => {
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);

    const { chatStore } = setupStores();
    chatStore.userInput = 'جدول بنتي';
    const sendPromise = chatStore.handleSend();
    await flushPromises();

    stream.pushEvent({ type: 'content', content: 'دي جدولها:' });
    stream.pushEvent({
      type: 'answer_blocks',
      answer_blocks: [WEEK_BLOCK, { kind: 'attendance', index: 1, data: { rows: [] } }],
    });
    stream.pushEvent({ type: 'content_replace', content: SETTLED_TIMETABLE });
    stream.close();
    await sendPromise;

    const bubble = chatStore.messagesBySession.session_current[1];
    expect(bubble.text).toBe(SETTLED_TIMETABLE);
    expect(bubble.answerBlocks).toMatchObject([{ kind: 'timetable', index: 0, language: 'ar' }]);
  });

  it('leaves a clean answer alone when no retraction arrives', async () => {
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);

    const { chatStore } = setupStores();
    chatStore.userInput = 'مصاريف ابني كام';
    const sendPromise = chatStore.handleSend();
    await flushPromises();

    stream.pushEvent({ type: 'content', content: 'رسوم الصف الأول ' });
    stream.pushEvent({ type: 'content', content: '30,000 جنيه. [1]' });
    stream.close();
    await sendPromise;

    expect(chatStore.messagesBySession.session_current[1].text).toBe(
      'رسوم الصف الأول 30,000 جنيه. [1]',
    );
  });
});

describe('chat store conversation paging', () => {
  beforeEach(() => {
    // Restoring puts spies back; it no longer empties a `vi.fn()`'s call history (Vitest 3+),
    // so the shared `api` mock is cleared too — a test that counts calls starts from zero.
    vi.restoreAllMocks();
    vi.clearAllMocks();
    vi.stubGlobal('localStorage', createLocalStorageMock());
    vi.stubGlobal('alert', vi.fn());
    vi.stubGlobal('confirm', vi.fn(() => true));
  });

  const serverMessage = (id: number, content: string, extra: Record<string, unknown> = {}) => ({
    id,
    type: id % 2 === 1 ? 'human' : 'ai',
    content,
    timestamp: '2026-07-08T00:00:00',
    ...extra,
  });

  it('opens a conversation with one batch and remembers where it stopped', async () => {
    vi.mocked(api.get).mockResolvedValue({
      data: {
        messages: [serverMessage(41, 'Recent question'), serverMessage(42, 'Recent answer')],
        has_more: true,
      },
    });

    const { chatStore } = setupStores();
    await chatStore.loadSession('session_long');

    expect(api.get).toHaveBeenCalledWith('/sessions/session_long?limit=15');
    expect(chatStore.messages.map((msg) => msg.text)).toEqual([
      'Recent question',
      'Recent answer',
    ]);
    expect(chatStore.pagingBySession.session_long).toEqual({
      oldestId: 41,
      hasMore: true,
      loadingOlder: false,
    });
    expect(chatStore.canLoadOlderMessages).toBe(true);
  });

  it('prepends the batch before the oldest message held', async () => {
    vi.mocked(api.get).mockResolvedValueOnce({
      data: {
        messages: [serverMessage(41, 'Recent question'), serverMessage(42, 'Recent answer')],
        has_more: true,
      },
    });

    const { chatStore } = setupStores();
    await chatStore.loadSession('session_long');

    vi.mocked(api.get).mockResolvedValueOnce({
      data: {
        messages: [serverMessage(39, 'Older question'), serverMessage(40, 'Older answer')],
        has_more: false,
      },
    });
    await chatStore.loadOlderMessages('session_long');

    expect(api.get).toHaveBeenLastCalledWith('/sessions/session_long?limit=15&before=41');
    expect(chatStore.messages.map((msg) => msg.text)).toEqual([
      'Older question',
      'Older answer',
      'Recent question',
      'Recent answer',
    ]);
    expect(chatStore.pagingBySession.session_long).toEqual({
      oldestId: 39,
      hasMore: false,
      loadingOlder: false,
    });
    expect(chatStore.canLoadOlderMessages).toBe(false);
  });

  it('does not fetch again once the start of the conversation is reached', async () => {
    vi.mocked(api.get).mockResolvedValue({
      data: { messages: [serverMessage(1, 'The very first message')], has_more: false },
    });

    const { chatStore } = setupStores();
    await chatStore.loadSession('session_short');
    vi.mocked(api.get).mockClear();

    await chatStore.loadOlderMessages('session_short');

    expect(api.get).not.toHaveBeenCalled();
  });

  it('joins a clarification exchange split across two batches', async () => {
    // The reply was mapped by an earlier fetch that could not see the request above it.
    vi.mocked(api.get).mockResolvedValueOnce({
      data: {
        messages: [
          serverMessage(43, 'Danjin'),
          serverMessage(44, 'Danjin has the Nihility element.'),
        ],
        has_more: true,
      },
    });

    const { chatStore } = setupStores();
    await chatStore.loadSession('session_hitl');
    expect(chatStore.messages[0]).toMatchObject({ isUser: true, isHitlAnswer: false });

    vi.mocked(api.get).mockResolvedValueOnce({
      data: {
        messages: [
          serverMessage(41, 'Which character?'),
          serverMessage(42, 'Please provide the character name', {
            rag_trace: {
              retrieval_status: 'needs_clarification',
              route: 'clarify',
              hitl_prompt: 'Please provide the character name',
            },
          }),
        ],
        has_more: false,
      },
    });
    await chatStore.loadOlderMessages('session_hitl');

    expect(chatStore.messages[1]).toMatchObject({ isHitlRequest: true });
    expect(chatStore.messages[2]).toMatchObject({ text: 'Danjin', isHitlAnswer: true });
    expect(chatStore.messages[3]).toMatchObject({ hitlResumeText: 'Danjin' });
  });

  it('keeps a clarification still awaiting an answer after paging back', async () => {
    vi.mocked(api.get).mockResolvedValueOnce({
      data: {
        messages: [
          serverMessage(43, 'What about fees?'),
          serverMessage(44, 'Which year group?', {
            rag_trace: {
              retrieval_status: 'needs_clarification',
              route: 'clarify',
              hitl_prompt: 'Which year group?',
            },
          }),
        ],
        has_more: true,
      },
    });

    const { chatStore } = setupStores();
    await chatStore.loadSession('session_pending');
    expect(chatStore.currentPendingHitl?.prompt).toBe('Which year group?');

    vi.mocked(api.get).mockResolvedValueOnce({
      data: { messages: [serverMessage(41, 'Hello'), serverMessage(42, 'Hi there')], has_more: false },
    });
    await chatStore.loadOlderMessages('session_pending');

    expect(chatStore.currentPendingHitl?.prompt).toBe('Which year group?');
  });

  it('leaves paging alone for a session that is still streaming', async () => {
    const stream = createControlledSseFetch();
    vi.stubGlobal('fetch', stream.fetchMock);

    const { chatStore } = setupStores();
    chatStore.userInput = 'A question';
    const sendPromise = chatStore.handleSend();
    await flushPromises();

    vi.mocked(api.get).mockClear();
    await chatStore.loadSession('session_current');

    expect(api.get).not.toHaveBeenCalled();
    expect(chatStore.canLoadOlderMessages).toBe(false);

    stream.close();
    await sendPromise;
  });

  it('draws the tables of a reloaded conversation again', () => {
    const { chatStore } = setupStores();
    const [answer, older] = chatStore.mapServerMessages([
      serverMessage(2, SETTLED_TIMETABLE, { rag_trace: { answer_blocks: [WEEK_BLOCK] } }),
      // Stored before blocks existed: no data, so its record prints as markdown.
      serverMessage(4, SETTLED_TIMETABLE),
    ]);
    expect(answer.answerBlocks).toMatchObject([{ kind: 'timetable', index: 0 }]);
    expect(older.answerBlocks).toEqual([]);
  });

  it('forgets the scroll-back after the conversation is cleared', async () => {
    vi.mocked(api.get).mockResolvedValue({
      data: { messages: [serverMessage(41, 'Question')], has_more: true },
    });
    vi.mocked(api.delete).mockResolvedValue({ data: { message: 'Session deleted successfully' } });

    const { chatStore } = setupStores();
    await chatStore.loadSession('session_long');
    expect(chatStore.canLoadOlderMessages).toBe(true);

    await chatStore.handleClearChat();

    expect(chatStore.pagingBySession.session_long).toBeUndefined();
    expect(chatStore.canLoadOlderMessages).toBe(false);
  });

  it('clears a conversation on the server too, and starts a fresh one', async () => {
    // Clearing only the screen left the conversation stored: it came back on reopen and
    // the assistant kept reading it as history.
    vi.mocked(api.get).mockResolvedValue({
      data: { messages: [serverMessage(41, 'Question'), serverMessage(42, 'Answer')], has_more: false },
    });
    vi.mocked(api.delete).mockResolvedValue({ data: { message: 'Session deleted successfully' } });

    const { chatStore, sessionStore } = setupStores();
    sessionStore.sessions = [{ session_id: 'session_long', title: 'Q', message_count: 2, updated_at: 'now' }];
    await chatStore.loadSession('session_long');

    await chatStore.handleClearChat();

    expect(api.delete).toHaveBeenCalledWith('/sessions/session_long');
    expect(chatStore.messagesBySession.session_long).toBeUndefined();
    expect(sessionStore.sessions.find((s) => s.session_id === 'session_long')).toBeUndefined();
    expect(chatStore.sessionId).not.toBe('session_long');
    expect(chatStore.messages).toEqual([]);
  });

  it('drops a conversation the server never saw without asking it to delete anything', async () => {
    const { chatStore } = setupStores();
    chatStore.messagesBySession.session_current = [{ text: 'typed, never sent', isUser: true }];
    chatStore.messages = chatStore.messagesBySession.session_current;

    await chatStore.handleClearChat();

    expect(api.delete).not.toHaveBeenCalled();
    expect(chatStore.messages).toEqual([]);
  });

  it('keeps the conversation when the server refuses to delete it', async () => {
    vi.mocked(api.get).mockResolvedValue({
      data: { messages: [serverMessage(41, 'Question')], has_more: false },
    });
    vi.mocked(api.delete).mockRejectedValue(new Error('Network Error'));

    const { chatStore } = setupStores();
    await chatStore.loadSession('session_long');

    await chatStore.handleClearChat();

    expect(chatStore.sessionId).toBe('session_long');
    expect(chatStore.messages.map((msg) => msg.text)).toEqual(['Question']);
  });
});
