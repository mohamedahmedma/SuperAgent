import { defineStore } from 'pinia';
import identityApi, {
  ACCESS_TOKEN_KEY,
  REFRESH_TOKEN_KEY,
} from '@/utils/identityApi';
import type { CurrentUser, UserRole } from '@/types/user';

/**
 * Authentication now talks to the identity service, not the chat backend.
 *
 * The backend no longer has `/auth/*` routes — it only verifies the tokens it is
 * handed. Login, registration and refresh all go to a separate origin, which is why
 * every call here uses `identityApi` rather than the shared `api` instance.
 */
export const useAuthStore = defineStore('auth', {
  state: () => ({
    token: localStorage.getItem(ACCESS_TOKEN_KEY) || '',
    refreshToken: localStorage.getItem(REFRESH_TOKEN_KEY) || '',
    currentUser: null as CurrentUser | null,
    authMode: 'login' as 'login' | 'register',
    authForm: {
      username: '',
      password: '',
      role: 'user' as 'user' | 'admin',
      admin_code: '',
    },
    authLoading: false,
  }),

  getters: {
    isAuthenticated(): boolean {
      return !!this.token && !!this.currentUser;
    },
    isAdmin(): boolean {
      return this.currentUser?.role === 'admin';
    },
    /**
     * Whether to offer the student-records features at all. Not an authorisation
     * check — the records facade makes that decision from the signed token — just a
     * hint about which UI is worth showing.
     */
    isParent(): boolean {
      return !!this.currentUser?.guardianId;
    },
  },

  actions: {
    async fetchMe() {
      if (!this.token) return;
      try {
        const response = await identityApi.get('/v1/auth/me', {
          headers: { Authorization: `Bearer ${this.token}` },
        });
        this.currentUser = {
          username: response.data.username,
          role: response.data.role as UserRole,
          guardianId: response.data.guardian_id ?? null,
          displayName: response.data.display_name || '',
        };
      } catch (error) {
        this.handleLogout();
        throw error;
      }
    },

    async handleAuthSubmit() {
      if (this.authLoading) return;
      const username = this.authForm.username.trim();
      const password = this.authForm.password.trim();
      if (!username || !password) {
        throw new Error('Username and password cannot be empty');
      }

      this.authLoading = true;
      try {
        const endpoint =
          this.authMode === 'login' ? '/v1/auth/login' : '/v1/auth/register';
        const payload: Record<string, unknown> = { username, password };
        if (this.authMode === 'register') {
          payload.role = this.authForm.role;
          payload.admin_code = this.authForm.admin_code || null;
        }

        const { data } = await identityApi.post(endpoint, payload);
        this.applySession(data);

        // Reset password fields
        this.authForm.password = '';
        this.authForm.admin_code = '';
      } catch (error: any) {
        // The identity service returns `detail` as an object with a machine-readable
        // `code`; the old backend returned a bare string. Both are handled so a
        // mid-migration deployment cannot render "[object Object]" at a parent.
        const detail = error.response?.data?.detail;
        const message =
          (typeof detail === 'string' ? detail : detail?.message) ||
          error.message ||
          'Authentication failed';
        throw new Error(message);
      } finally {
        this.authLoading = false;
      }
    },

    applySession(data: any) {
      this.token = data.access_token;
      this.refreshToken = data.refresh_token || '';
      this.currentUser = {
        username: data.username,
        role: data.role as UserRole,
        guardianId: data.guardian_id ?? null,
        displayName: data.display_name || '',
      };
      localStorage.setItem(ACCESS_TOKEN_KEY, this.token);
      if (this.refreshToken) {
        localStorage.setItem(REFRESH_TOKEN_KEY, this.refreshToken);
      }
    },

    async handleLogout() {
      // Tell identity to revoke the refresh token. Best effort: the local session is
      // cleared either way, because a user who clicked "log out" must end up logged
      // out even if the network call fails.
      const refreshToken = this.refreshToken;
      this.token = '';
      this.refreshToken = '';
      this.currentUser = null;
      localStorage.removeItem(ACCESS_TOKEN_KEY);
      localStorage.removeItem(REFRESH_TOKEN_KEY);

      if (refreshToken) {
        try {
          await identityApi.post('/v1/auth/logout', { refresh_token: refreshToken });
        } catch {
          // Already gone, or unreachable. Nothing useful to do.
        }
      }
    },
  },
});
