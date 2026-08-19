import { defineStore } from 'pinia';
import { useAuthStore } from './auth';
import { useSessionStore } from './sessions';
import api from '@/utils/api';
import type { Message, RagStep, GroupedRagStep, HitlRequest, RagTrace, SessionPaging } from '@/types/chat';

// One scroll-back. Opening a chat loads the last screenful; older batches arrive as the
// user scrolls up to them, so a conversation with a thousand messages opens as fast as
// one with ten.
const PAGE_SIZE = 15;

export const useChatStore = defineStore('chat', {
  state: () => ({
    messages: [] as Message[],
    messagesBySession: {} as Record<string, Message[]>,
    userInput: '',
    isLoading: false,
    activeNav: 'newChat' as 'newChat' | 'history' | 'settings',
    sessionId: 'session_' + Date.now(),
    streamingSessionId: null as string | null,
    abortController: null as AbortController | null,
    pendingHitlBySession: {} as Record<string, HitlRequest | null>,
    // Where each session's scroll-back has got to. Absent means nothing was loaded from
    // the server — a brand new chat has no history to page through.
    pagingBySession: {} as Record<string, SessionPaging>,
  }),

  getters: {
    isViewingStreamingSession(state): boolean {
      return state.isLoading && state.streamingSessionId === state.sessionId;
    },

    isInputLocked(state): boolean {
      return state.isLoading && state.streamingSessionId !== state.sessionId;
    },

    currentPendingHitl(state): HitlRequest | null {
      return state.pendingHitlBySession[state.sessionId] || null;
    },

    canLoadOlderMessages(state): boolean {
      const paging = state.pagingBySession[state.sessionId];
      return !!paging && paging.hasMore && !paging.loadingOlder && paging.oldestId !== null;
    },

    isLoadingOlderMessages(state): boolean {
      return !!state.pagingBySession[state.sessionId]?.loadingOlder;
    },

    inputPlaceholder(state): string {
      const pendingHitl = state.pendingHitlBySession[state.sessionId];
      if (pendingHitl) {
        return 'Type your own answer, or pick an option above and send...';
      }
      return 'Say something to Agent... (Shift+Enter for a new line)';
    },
  },

  actions: {
    resetWorkspace() {
      if (this.abortController) {
        this.abortController.abort();
      }
      this.$reset();
    },

    ensureSessionMessages(sessionId: string): Message[] {
      if (!this.messagesBySession[sessionId]) {
        this.messagesBySession[sessionId] = [];
      }
      return this.messagesBySession[sessionId];
    },

    isHitlTrace(trace?: RagTrace | null): boolean {
      if (!trace) return false;
      return trace.retrieval_status === 'needs_clarification'
        || trace.retrieval_status === 'needs_scope_selection'
        || trace.route === 'clarify'
        || trace.route === 'scope_select';
    },

    normalizeHitlRequest(hitl: any, trace?: RagTrace | null): HitlRequest {
      const prompt = String(hitl?.prompt || trace?.hitl_prompt || 'Please provide one more key detail so I can continue the search.');
      const rawOptions = hitl?.options || trace?.hitl_options || [];
      const options = Array.isArray(rawOptions)
        ? rawOptions.map((item) => String(item).trim()).filter(Boolean)
        : [];
      return {
        id: hitl?.id,
        prompt,
        options,
        route: hitl?.route || trace?.route,
        retrieval_status: hitl?.retrieval_status || trace?.retrieval_status,
        original_question: hitl?.original_question,
      };
    },

    formatHitlText(hitl: HitlRequest): string {
      const options = hitl.options || [];
      if (!options.length) return hitl.prompt;
      return `${hitl.prompt}\n\nOptions:\n${options.map((item) => `- ${item}`).join('\n')}`;
    },

    derivePendingHitl(messages: Message[]): HitlRequest | null {
      const lastMessage = messages[messages.length - 1];
      if (!lastMessage || lastMessage.isUser || !this.isHitlTrace(lastMessage.ragTrace)) {
        return null;
      }
      return this.normalizeHitlRequest(
        {
          prompt: lastMessage.hitlPrompt || lastMessage.ragTrace?.hitl_prompt || lastMessage.text,
          options: lastMessage.hitlOptions || lastMessage.ragTrace?.hitl_options || [],
        },
        lastMessage.ragTrace
      );
    },

    syncPendingHitlFromMessages(sessionId: string) {
      const pendingHitl = this.derivePendingHitl(this.ensureSessionMessages(sessionId));
      if (pendingHitl) {
        this.pendingHitlBySession[sessionId] = pendingHitl;
      } else {
        delete this.pendingHitlBySession[sessionId];
      }
    },

    selectHitlOption(option: string) {
      this.userInput = option;
    },

    setViewedSession(sessionId: string, messages?: Message[]) {
      if (messages) {
        this.messagesBySession[sessionId] = messages;
        this.syncPendingHitlFromMessages(sessionId);
      }
      this.sessionId = sessionId;
      this.messages = this.ensureSessionMessages(sessionId);
      if (!messages) {
        this.syncPendingHitlFromMessages(sessionId);
      }
      this.activeNav = 'newChat';
    },

    createSessionId(): string {
      let nextId = 'session_' + Date.now();
      while (this.messagesBySession[nextId]) {
        nextId = 'session_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
      }
      return nextId;
    },

    getLocalSessionTitle(sessionId: string, messages: Message[]): string {
      const firstUserMessage = messages.find((msg) => msg.isUser && msg.text.trim());
      if (!firstUserMessage) return sessionId;
      const title = firstUserMessage.text.trim();
      return title.length > 10 ? title.substring(0, 10) + '...' : title;
    },

    mapServerMessages(messages: any[]): Message[] {
      let awaitingHitlAnswer = false;
      let hitlResumeText: string | undefined;

      return (messages || []).map((msg: any) => {
        const ragTrace = msg.rag_trace || null;
        const isUser = msg.type === 'human';
        const isHitlRequest = !isUser && this.isHitlTrace(ragTrace);
        const isHitlAnswer = isUser && awaitingHitlAnswer;
        const resumeTextForMessage = !isUser && !isHitlRequest ? hitlResumeText : undefined;

        if (isHitlRequest) {
          awaitingHitlAnswer = true;
          hitlResumeText = undefined;
        } else if (isHitlAnswer) {
          awaitingHitlAnswer = false;
          hitlResumeText = msg.content;
        } else if (!isUser) {
          hitlResumeText = undefined;
        }

        return {
          text: msg.content,
          isUser,
          isHitlRequest,
          isHitlAnswer,
          hitlPrompt: isHitlRequest ? ragTrace?.hitl_prompt || msg.content : undefined,
          hitlOptions: isHitlRequest ? ragTrace?.hitl_options || [] : undefined,
          hitlResumeText: resumeTextForMessage,
          ragTrace,
          // Reloading a past session restores its images too: the backend persists
          // them on the trace, so they survive a page refresh.
          assets: ragTrace?.assets || [],
        };
      });
    },

    mergeCachedSessionsIntoHistory() {
      const sessionStore = useSessionStore();
      const sessions = sessionStore.sessions.map((session) => ({
        ...session,
        isStreaming: this.isLoading && session.session_id === this.streamingSessionId,
      }));

      Object.entries(this.messagesBySession).forEach(([sessionId, messages]) => {
        if (!messages.length) return;

        const existingIndex = sessions.findIndex((session) => session.session_id === sessionId);
        const existing = existingIndex >= 0 ? sessions[existingIndex] : null;
        const localSession = {
          session_id: sessionId,
          title: existing?.title || this.getLocalSessionTitle(sessionId, messages),
          message_count: Math.max(existing?.message_count || 0, messages.length),
          updated_at: existing?.updated_at || new Date().toISOString(),
          isStreaming: this.isLoading && sessionId === this.streamingSessionId,
        };

        if (existingIndex >= 0) {
          sessions[existingIndex] = { ...existing, ...localSession };
        } else {
          sessions.unshift(localSession);
        }
      });

      sessionStore.sessions = sessions;
    },

    appendRagStepToGroups(prev: GroupedRagStep[], step: RagStep): GroupedRagStep[] {
      const groups = prev ? [...prev] : [];
      const g = step.group || null;
      const groupLabel = step.group_label || g;
      
      if (g) {
        const idx = groups.findIndex((grp) => grp.group === g);
        if (idx >= 0) {
          const existing = groups[idx];
          const updated: GroupedRagStep = {
            group: existing.group,
            label: existing.label || groupLabel,
            steps: [...existing.steps, step],
            collapsed: existing.collapsed,
          };
          groups[idx] = updated;
          return groups;
        }
        return [...groups, { group: g, label: groupLabel, steps: [step], collapsed: true }];
      }

      const last = groups.length > 0 ? groups[groups.length - 1] : null;
      if (last && last.group === null) {
        const updated = { ...last, steps: [...last.steps, step] };
        groups[groups.length - 1] = updated;
        return groups;
      }
      return [...groups, { group: null, label: null, steps: [step], collapsed: false }];
    },

    groupRagSteps(steps: RagStep[]): GroupedRagStep[] {
      if (!steps || !steps.length) return [];
      return steps.reduce((groups: GroupedRagStep[], step) => this.appendRagStepToGroups(groups, step), []);
    },

    toggleStepGroup(msgIndex: number, groupIndex: number) {
      const msg = this.messages[msgIndex];
      if (!msg || !msg._groupedSteps || !msg._groupedSteps[groupIndex]) return;
      msg._groupedSteps[groupIndex].collapsed = !msg._groupedSteps[groupIndex].collapsed;
    },

    handleNewChat() {
      const sessionId = this.createSessionId();
      this.messagesBySession[sessionId] = [];
      delete this.pendingHitlBySession[sessionId];
      delete this.pagingBySession[sessionId];
      this.setViewedSession(sessionId);
      const sessionStore = useSessionStore();
      sessionStore.showHistorySidebar = false;
    },

    handleClearChat() {
      if (this.streamingSessionId === this.sessionId) {
        alert('This chat is still generating a response. Stop it or wait for it to finish before clearing.');
        return;
      }
      if (confirm('Clear the current conversation? Meow?')) {
        this.messagesBySession[this.sessionId] = [];
        this.messages = this.messagesBySession[this.sessionId];
        delete this.pendingHitlBySession[this.sessionId];
        // Otherwise scrolling up would pull the cleared conversation back in.
        delete this.pagingBySession[this.sessionId];
      }
    },

    recordPaging(sessionId: string, serverMessages: any[], hasMore: boolean) {
      const oldest = serverMessages[0];
      this.pagingBySession[sessionId] = {
        // The cursor for the next batch: everything older than the oldest message held.
        oldestId: typeof oldest?.id === 'number' ? oldest.id : null,
        hasMore: !!hasMore,
        loadingOlder: false,
      };
    },

    async loadSession(sessionId: string) {
      const sessionStore = useSessionStore();
      const cachedMessages = this.messagesBySession[sessionId];

      this.setViewedSession(sessionId, cachedMessages || []);
      sessionStore.showHistorySidebar = false;

      if (sessionId === this.streamingSessionId) {
        this.mergeCachedSessionsIntoHistory();
        return;
      }

      try {
        // The newest batch only. What came before it is fetched when scrolled to.
        const response = await api.get(
          `/sessions/${encodeURIComponent(sessionId)}?limit=${PAGE_SIZE}`
        );
        const data = response.data;
        const serverMessages = data.messages || [];
        const loadedMessages = this.mapServerMessages(serverMessages);
        this.messagesBySession[sessionId] = loadedMessages;
        this.recordPaging(sessionId, serverMessages, data.has_more);
        this.syncPendingHitlFromMessages(sessionId);
        if (this.sessionId === sessionId) {
          this.messages = loadedMessages;
        }
        this.mergeCachedSessionsIntoHistory();
      } catch (error: any) {
        const errMsg = error.response?.data?.detail || error.message || 'Failed to load session';
        if (!cachedMessages && this.sessionId === sessionId) {
          this.messages = [];
        }
        throw new Error(errMsg);
      }
    },

    /**
     * Prepend the batch immediately older than what is on screen.
     *
     * Paging by the oldest message's id rather than by an offset: a turn finishing while
     * someone reads back does not shift the window, so no batch repeats or goes missing.
     */
    async loadOlderMessages(requestedSessionId?: string) {
      const sessionId = requestedSessionId || this.sessionId;
      const paging = this.pagingBySession[sessionId];
      if (!paging || !paging.hasMore || paging.loadingOlder || paging.oldestId === null) return;

      paging.loadingOlder = true;
      try {
        const response = await api.get(
          `/sessions/${encodeURIComponent(sessionId)}?limit=${PAGE_SIZE}&before=${paging.oldestId}`
        );
        const data = response.data;
        const serverMessages = data.messages || [];
        const older = this.mapServerMessages(serverMessages);

        const existing = this.ensureSessionMessages(sessionId);
        this.stitchHitlAcrossBatches(older, existing);
        const combined = [...older, ...existing];
        this.messagesBySession[sessionId] = combined;
        if (this.sessionId === sessionId) {
          this.messages = combined;
        }

        this.recordPaging(sessionId, serverMessages, data.has_more);
        // Nothing came back, so there is nothing older however the server counted it.
        if (!older.length) {
          this.pagingBySession[sessionId] = { oldestId: paging.oldestId, hasMore: false, loadingOlder: false };
        }
      } catch (error: any) {
        const current = this.pagingBySession[sessionId];
        if (current) current.loadingOlder = false;
        throw new Error(error.response?.data?.detail || error.message || 'Failed to load older messages');
      }
    },

    /**
     * Repair a clarification exchange split across a batch boundary.
     *
     * `mapServerMessages` reads a conversation forwards: a clarification request marks
     * the reply that follows it as the answer, and those two are hidden as a pair. When
     * the request is the last message of an older batch, the reply was mapped in an
     * earlier fetch that could not have known — so it would show as an ordinary message
     * and the exchange would render twice.
     */
    stitchHitlAcrossBatches(older: Message[], existing: Message[]) {
      const request = older[older.length - 1];
      const answer = existing[0];
      if (!request?.isHitlRequest || !answer?.isUser || answer.isHitlAnswer) return;

      answer.isHitlAnswer = true;
      const nextReply = existing[1];
      if (nextReply && !nextReply.isUser && !nextReply.isHitlRequest && !nextReply.hitlResumeText) {
        nextReply.hitlResumeText = answer.text;
      }
    },

    handleStop() {
      if (this.abortController) {
        this.abortController.abort();
      }
    },

    async handleSend() {
      const authStore = useAuthStore();
      const sessionStore = useSessionStore();

      if (!authStore.isAuthenticated) {
        alert('Please log in first');
        return;
      }

      const text = this.userInput.trim();
      if (!text) return;
      if (this.isLoading) {
        alert('A response is already being generated. Wait for it to finish, or go back to that session to stop it.');
        return;
      }

      const requestSessionId = this.sessionId;
      const requestMessages = this.ensureSessionMessages(requestSessionId);
      const pendingHitlAtSend = this.pendingHitlBySession[requestSessionId] || null;
      if (this.sessionId === requestSessionId) {
        this.messages = requestMessages;
      }

      requestMessages.push({
        text: text,
        isUser: true,
        isHitlAnswer: !!pendingHitlAtSend,
      });
      if (pendingHitlAtSend) {
        delete this.pendingHitlBySession[requestSessionId];
      }

      if (requestMessages.length === 1) {
        const tempTitle = this.getLocalSessionTitle(requestSessionId, requestMessages);
        const existingSession = sessionStore.sessions.find((s) => s.session_id === requestSessionId);
        if (existingSession) {
          existingSession.title = existingSession.title || tempTitle;
          existingSession.message_count = requestMessages.length;
          existingSession.updated_at = new Date().toISOString();
          existingSession.isStreaming = true;
        } else {
          sessionStore.sessions.unshift({
            session_id: requestSessionId,
            title: tempTitle,
            message_count: requestMessages.length,
            updated_at: new Date().toISOString(),
            isStreaming: true,
          });
        }
      }

      this.userInput = '';
      this.isLoading = true;
      this.streamingSessionId = requestSessionId;

      requestMessages.push({
        text: '',
        isUser: false,
        isThinking: true,
        thinkingStartedAt: Date.now(),
        hitlResumeText: pendingHitlAtSend ? text : undefined,
        ragTrace: null,
        assets: [],
        ragSteps: [],
        _groupedSteps: [],
      });
      const botMsgIdx = requestMessages.length - 1;
      this.mergeCachedSessionsIntoHistory();

      this.abortController = new AbortController();
      let receivedHitlRequest = false;
      let streamHadError = false;

      try {
        const response = await fetch('/chat/stream', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${authStore.token}`,
          },
          body: JSON.stringify({
            message: text,
            session_id: requestSessionId,
          }),
          signal: this.abortController.signal,
        });

        if (!response.ok) {
          if (response.status === 401) {
            authStore.handleLogout();
            throw new Error('Your session has expired, please log in again');
          }
          throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error('Unable to read the response stream');

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          let eventEndIndex;
          while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
            const eventStr = buffer.slice(0, eventEndIndex);
            buffer = buffer.slice(eventEndIndex + 2);

            if (eventStr.startsWith('data: ')) {
              const dataStr = eventStr.slice(6);
              if (dataStr === '[DONE]') continue;
              try {
                const data = JSON.parse(dataStr);
                if (data.type === 'content') {
                  const botMsg = requestMessages[botMsgIdx];
                  if (!botMsg) continue;
                  if (botMsg.isThinking) {
                    botMsg.isThinking = false;
                  }
                  if (botMsg.isHitlRequest) {
                    continue;
                  }
                  botMsg.text += data.content;
                } else if (data.type === 'assets') {
                  // Its own event, ahead of the trace, so images can render without
                  // depending on the trace payload's shape.
                  const botMsg = requestMessages[botMsgIdx];
                  if (botMsg) {
                    botMsg.assets = data.assets || [];
                  }
                } else if (data.type === 'trace') {
                  const botMsg = requestMessages[botMsgIdx];
                  if (botMsg) {
                    botMsg.ragTrace = data.rag_trace;
                  }
                } else if (data.type === 'hitl_request') {
                  const botMsg = requestMessages[botMsgIdx];
                  if (!botMsg) continue;
                  const hitl = this.normalizeHitlRequest(data.hitl, botMsg.ragTrace);
                  receivedHitlRequest = true;
                  this.pendingHitlBySession[requestSessionId] = hitl;
                  botMsg.isThinking = false;
                  botMsg.isHitlRequest = true;
                  botMsg.hitlPrompt = hitl.prompt;
                  botMsg.hitlOptions = hitl.options || [];
                  botMsg.text = this.formatHitlText(hitl);
                } else if (data.type === 'rag_step') {
                  const msg = requestMessages[botMsgIdx];
                  if (!msg) continue;
                  if (!msg.ragSteps) msg.ragSteps = [];
                  msg.ragSteps.push(data.step);
                  msg._groupedSteps = this.appendRagStepToGroups(msg._groupedSteps || [], data.step);
                } else if (data.type === 'session_title') {
                  const s = sessionStore.sessions.find(
                    (item) => item.session_id === data.session_id
                  );
                  if (s) {
                    s.title = data.title;
                    s.updated_at = new Date().toISOString();
                    s.message_count = requestMessages.length;
                    s.isStreaming = data.session_id === this.streamingSessionId;
                  } else {
                    sessionStore.sessions.unshift({
                      session_id: data.session_id,
                      title: data.title,
                      message_count: requestMessages.length,
                      updated_at: new Date().toISOString(),
                      isStreaming: data.session_id === this.streamingSessionId,
                    });
                  }
                } else if (data.type === 'error') {
                  streamHadError = true;
                  const botMsg = requestMessages[botMsgIdx];
                  if (!botMsg) continue;
                  botMsg.isThinking = false;
                  botMsg.text += `\n[Error: ${data.content}]`;
                }
              } catch (e) {
                console.warn('SSE parse error:', e);
              }
            }
          }
        }
      } catch (error: any) {
        streamHadError = true;
        const botMsg = requestMessages[botMsgIdx];
        if (!botMsg) return;
        if (error.name === 'AbortError') {
          botMsg.isThinking = false;
          if (!botMsg.text) {
            botMsg.text = '(Response stopped)';
          } else {
            botMsg.text += '\n\n_(Response was stopped)_';
          }
        } else {
          botMsg.isThinking = false;
          botMsg.text = `Meow... something went wrong: ${error.message}`;
        }
      } finally {
        if (streamHadError && pendingHitlAtSend && !receivedHitlRequest) {
          this.pendingHitlBySession[requestSessionId] = pendingHitlAtSend;
        }
        this.isLoading = false;
        this.streamingSessionId = null;
        this.abortController = null;
        this.mergeCachedSessionsIntoHistory();
      }
    },
  },
});
