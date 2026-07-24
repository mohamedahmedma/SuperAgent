import { defineStore } from 'pinia';
import api from '@/utils/api';
import type { ChatSession } from '@/types/chat';

export const useSessionStore = defineStore('sessions', {
  state: () => ({
    sessions: [] as ChatSession[],
    showHistorySidebar: false,
  }),

  actions: {
    async fetchSessions() {
      try {
        const response = await api.get('/sessions');
        this.sessions = response.data.sessions || [];
      } catch (error: any) {
        const errMsg = error.response?.data?.detail || error.message || 'Failed to load conversation history';
        throw new Error(errMsg);
      }
    },

    async deleteSession(sessionId: string) {
      try {
        const response = await api.delete(`/sessions/${encodeURIComponent(sessionId)}`);
        this.sessions = this.sessions.filter((s) => s.session_id !== sessionId);
        return response.data.message || 'Session deleted';
      } catch (error: any) {
        const errMsg = error.response?.data?.detail || error.message || 'Failed to delete session';
        throw new Error(errMsg);
      }
    },
  },
});
